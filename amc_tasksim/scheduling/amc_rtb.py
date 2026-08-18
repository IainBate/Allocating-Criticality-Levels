"""AMC-rtb schedulability test for Adaptive Mixed-Criticality scheduling.

Implements the AMC-rtb response-time analysis from "Analysis-Runtime
Co-design for Adaptive Mixed-Criticality Scheduling" (RTAS 2022),
Sections III-B and equations (1)-(2).

Also provides the standard fixed-priority response-time analysis for
comparison (the "non-trivial AMC" filter from the spec).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from amc_tasksim.generation.taskset import TaskSet


def _warn_no_convergence(task_index: int, quantity: str, max_iterations: int) -> None:
    warnings.warn(
        f"{quantity} for task {task_index} did not converge in {max_iterations} "
        f"iterations and did not exceed its deadline; result may be inaccurate",
        stacklevel=3,
    )


@dataclass
class ResponseTimeResult:
    """Results of AMC-rtb response-time analysis.

    Attributes:
        r_lo: Per-task LO-criticality response time (Ri(LO)).
        r_hi: Per-task HI-criticality response time (Ri(HI)).
        schedulable_lo: Whether each task meets its deadline in LO mode.
        schedulable_hi: Whether each HI-criticality task meets its deadline in HI mode.
        overall_schedulable: True if all tasks are schedulable.
    """

    r_lo: list[int] = field(default_factory=list)
    r_hi: list[int] = field(default_factory=list)
    schedulable_lo: list[bool] = field(default_factory=list)
    schedulable_hi: list[bool] = field(default_factory=list)
    overall_schedulable: bool = False


def _num_jobs(task_T: int, time: int) -> int:
    """Number of releases of a periodic task with period T in (0, time].

    Matches the AMC-rtb formula: ceil(time / T). Computed with integer
    arithmetic so the result is exact for large response times.
    """
    if time <= 0:
        return 0
    return -(-int(time) // int(task_T))


def amc_rtb(taskset: TaskSet, max_iterations: int = 10000) -> ResponseTimeResult:
    """Run AMC-rtb response-time analysis on a task set.

    Computes Ri(LO) via standard fixed-priority response-time analysis
    (equation 1 in the AMC-RH paper):

        R_i(LO) = C_i(LO) + sum over j in hp(i) of:
            ceil(R_i(LO) / T_j) * C_j(LO)

    where hp(i) is the set of tasks with higher priority than i.

    Computes Ri(HI) for HI-criticality tasks via equation (2):

        R_i(HI) = C_i(HI) + sum over j in hp_HI(i) of:
            ceil(R_i(HI) / T_j) * C_j(HI)
        + sum over j in hp_LO(i) of:
            min(ceil(R_i(HI) / T_j), ceil(R_i(LO) / T_j) + 1) * C_j(LO)

    where hp_HI(i) = higher-priority HI-criticality tasks,
          hp_LO(i) = higher-priority LO-criticality tasks.

    Args:
        taskset: Task set with priorities already assigned.
        max_iterations: Maximum fixed-point iterations.

    Returns:
        ResponseTimeResult with per-task response times and schedulability.
    """
    n = taskset.n
    r_lo = [0] * n
    r_hi = [0] * n

    for i in range(n):
        # --- Ri(LO) via fixed-point iteration ---
        # The iteration is monotonically increasing, so it either converges to
        # the response time or passes D_i, at which point the task is
        # unschedulable and there is nothing more to learn from continuing.
        w = taskset.C_lo[i]
        for _ in range(max_iterations):
            interference = 0
            for j in range(n):
                if taskset.priority[j] >= taskset.priority[i]:
                    continue  # j not in hp(i) (equal or higher priority number = not higher priority)
                interference += _num_jobs(taskset.T[j], w) * taskset.C_lo[j]
            w_new = taskset.C_lo[i] + interference
            if w_new == w:
                break
            w = w_new
            if w > taskset.D[i]:
                break
        else:
            _warn_no_convergence(i, "Ri(LO)", max_iterations)
        r_lo[i] = w

        # --- Ri(HI) for HI-criticality tasks ---
        if taskset.criticality[i] == "HI":
            w = taskset.C_hi[i]
            for _ in range(max_iterations):
                interference = 0
                for j in range(n):
                    if taskset.priority[j] >= taskset.priority[i]:
                        continue  # j not in hp(i)
                    if taskset.criticality[j] == "HI":
                        # HI-criticality interference at C_j(HI)
                        interference += _num_jobs(taskset.T[j], w) * taskset.C_hi[j]
                    else:
                        # LO-criticality releases are bounded by Ri(LO), per (2)
                        num_hi = _num_jobs(taskset.T[j], w)
                        num_lo = _num_jobs(taskset.T[j], r_lo[i])
                        interference += min(num_hi, num_lo) * taskset.C_lo[j]
                w_new = taskset.C_hi[i] + interference
                if w_new == w:
                    break
                w = w_new
                if w > taskset.D[i]:
                    break
            else:
                _warn_no_convergence(i, "Ri(HI)", max_iterations)
            r_hi[i] = w
        else:
            r_hi[i] = r_lo[i]  # LO-criticality tasks don't have HI response time

    # Determine schedulability
    schedulable_lo = [r_lo[i] <= taskset.D[i] for i in range(n)]
    schedulable_hi = [
        (r_hi[i] <= taskset.D[i]) if taskset.criticality[i] == "HI" else True
        for i in range(n)
    ]
    overall = all(schedulable_lo) and all(schedulable_hi)

    return ResponseTimeResult(
        r_lo=r_lo,
        r_hi=r_hi,
        schedulable_lo=schedulable_lo,
        schedulable_hi=schedulable_hi,
        overall_schedulable=overall,
    )


def busy_period_bound(taskset: TaskSet, max_iterations: int = 10000) -> Optional[int]:
    """Longest possible normal-mode busy period, over all priority levels.

    The fixed point of ``L = sum_j ceil(L / T_j) * C_j(LO)`` taken over every
    task, i.e. the priority level-n (lowest priority) busy period. No interval
    of continuous processor activity in normal mode can exceed it.

    This is what bounds how far back a simulation has to be warmed up before
    its run-queue state is guaranteed to have forgotten whatever preceded it:
    the busy period containing any instant ``t`` started at or before ``t``, so
    an idle instant must occur within ``L`` of ``t``.

    Returns:
        The bound, or None if the iteration does not converge (which means the
        LO-criticality utilisation is at or above 1 and busy periods are
        unbounded).
    """
    L = sum(taskset.C_lo)
    if L <= 0:
        return 0
    for _ in range(max_iterations):
        nxt = sum(_num_jobs(taskset.T[j], L) * taskset.C_lo[j] for j in range(taskset.n))
        if nxt == L:
            return int(L)
        L = nxt
    return None


def normal_mode_schedulable(taskset: TaskSet) -> bool:
    """Whether every task meets its deadline when all jobs comply with C_i(LO).

    This is requirement R1 of the AMC-RH paper. It is the precondition for
    treating an interval with no HI-criticality behaviour as uneventful: if it
    holds, such an interval contributes no deadline misses, no mode changes and
    no abandoned jobs, only job releases.
    """
    if not taskset.priority:
        from amc_tasksim.scheduling.priority import assign_deadline_monotonic

        assign_deadline_monotonic(taskset)
    return all(amc_rtb(taskset).schedulable_lo)


def amc_rtb_single(taskset: TaskSet, i: int, hp: set[int]) -> bool:
    """AMC-rtb schedulability of one task, given the set of higher-priority tasks.

    Equations (1) and (2) depend on hp(i) only as a set, not on the relative
    order within it, which is what makes Audsley's algorithm applicable.

    Args:
        taskset: Task set (priorities are not consulted).
        i: Index of the task under test.
        hp: Indices of the tasks with higher priority than task i.

    Returns:
        True if task i meets its deadline in both modes.
    """
    D_i = taskset.D[i]

    # Ri(LO), equation (1).
    w = taskset.C_lo[i]
    while True:
        nxt = taskset.C_lo[i] + sum(
            _num_jobs(taskset.T[j], w) * taskset.C_lo[j] for j in hp
        )
        if nxt == w:
            break
        w = nxt
        if w > D_i:
            return False
    r_lo = w

    if taskset.criticality[i] != "HI":
        return True

    # Ri(HI), equation (2).
    hp_hi = [j for j in hp if taskset.criticality[j] == "HI"]
    hp_lo = [j for j in hp if taskset.criticality[j] != "HI"]
    lo_interference = sum(
        _num_jobs(taskset.T[k], r_lo) * taskset.C_lo[k] for k in hp_lo
    )

    w = taskset.C_hi[i]
    while True:
        nxt = (
            taskset.C_hi[i]
            + sum(_num_jobs(taskset.T[j], w) * taskset.C_hi[j] for j in hp_hi)
            + lo_interference
        )
        if nxt == w:
            return True
        w = nxt
        if w > D_i:
            return False


def fpps_schedulable(taskset: TaskSet) -> bool:
    """Exact fixed-priority schedulability ignoring criticality.

    Every task is assumed to execute for max(C_i(LO), C_i(HI)), and priorities
    are deadline monotonic, which is optimal for constrained deadlines under
    single-criticality FPPS. This is the baseline the papers use to decide
    whether a task set actually needs a mixed-criticality scheme.
    """
    order = sorted(range(taskset.n), key=lambda i: (taskset.D[i], i))
    c_max = [max(taskset.C_lo[i], taskset.C_hi[i]) for i in range(taskset.n)]

    for rank, i in enumerate(order):
        hp = order[:rank]
        w = c_max[i]
        while True:
            nxt = c_max[i] + sum(_num_jobs(taskset.T[j], w) * c_max[j] for j in hp)
            if nxt == w:
                break
            w = nxt
            if w > taskset.D[i]:
                return False
    return True


def _fp_response_time_max(taskset: TaskSet, i: int) -> float:
    """Compute worst-case response time under fixed-priority with max execution times.

    Used by the non-trivial AMC filter: assumes every task executes at
    max(C_i(LO), C_i(HI)).
    """
    C_lo = taskset.C_lo
    C_hi = taskset.C_hi
    T = taskset.T
    priority = taskset.priority
    D = taskset.D

    def c_max(idx: int) -> int:
        return max(C_lo[idx], C_hi[idx])

    w = c_max(i)
    for _ in range(10000):
        interference = 0
        for j in range(len(T)):
            if priority[j] >= priority[i]:
                continue
            interference += _num_jobs(T[j], w) * c_max(j)
        w_new = c_max(i) + interference
        if w_new == w:
            break
        w = w_new
        if w > D[i]:
            break
    return w


def is_nontrivial_amc_taskset(taskset: TaskSet, use_opa: bool = True) -> bool:
    """Whether a task set is one the papers would have kept.

    Section V-C of the AMC-RH paper: "we required that the task sets chosen had
    at least one task that was unschedulable according to exact analysis of
    fixed priority preemptive scheduling (i.e. ignoring criticality), but were
    nevertheless schedulable according to the AMC-rtb test."

    So the task set must (a) fail single-criticality FPPS at
    max(C_i(LO), C_i(HI)) -- otherwise it needs no mixed-criticality scheme at
    all -- and (b) pass AMC-rtb.

    Args:
        taskset: Task set. Priorities are assigned if absent.
        use_opa: Assign priorities with Audsley's algorithm, as the papers do.
            With False, whatever ordering the task set already carries is used.

    Returns:
        True if the task set belongs to the papers' population.
    """
    from amc_tasksim.scheduling.priority import assign_audsley_opa, assign_deadline_monotonic

    # (a) must not already be schedulable ignoring criticality
    if fpps_schedulable(taskset):
        return False

    # (b) must be schedulable under AMC-rtb
    if use_opa:
        return assign_audsley_opa(taskset)
    if not taskset.priority:
        assign_deadline_monotonic(taskset)
    return amc_rtb(taskset).overall_schedulable
