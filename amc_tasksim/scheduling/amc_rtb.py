"""AMC-rtb schedulability test for Adaptive Mixed-Criticality scheduling.

Implements the AMC-rtb response-time analysis from "Analysis-Runtime
Co-design for Adaptive Mixed-Criticality Scheduling" (RTAS 2022),
Sections III-B and equations (1)-(2).

Also provides the standard fixed-priority response-time analysis for
comparison (the "non-trivial AMC" filter from the spec).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from amc_tasksim.generation.taskset import TaskSet


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

    r_lo: list[float] = field(default_factory=list)
    r_hi: list[float] = field(default_factory=list)
    schedulable_lo: list[bool] = field(default_factory=list)
    schedulable_hi: list[bool] = field(default_factory=list)
    overall_schedulable: bool = False


def _num_jobs(task_T: int, time: float) -> int:
    """Number of releases of a periodic task with period T in (0, time].

    Matches the AMC-rtb formula: ceil(time / T).
    """
    if time < 0:
        return 0
    return max(0, math.ceil(time / task_T))


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
    r_lo = [0.0] * n
    r_hi = [0.0] * n

    for i in range(n):
        # --- Ri(LO) via fixed-point iteration ---
        w = taskset.C_lo[i]
        for _ in range(max_iterations):
            interference = 0.0
            for j in range(n):
                if taskset.priority[j] >= taskset.priority[i]:
                    continue  # j not in hp(i) (equal or higher priority number = not higher priority)
                num = _num_jobs(taskset.T[j], w)
                interference += num * taskset.C_lo[j]
            w_new = taskset.C_lo[i] + interference
            if w_new == w or w_new > taskset.D[i] * 2:  # overflow guard
                break
            w = w_new
        r_lo[i] = w

        # --- Ri(HI) for HI-criticality tasks ---
        if taskset.criticality[i] == "HI":
            w = taskset.C_hi[i]
            for _ in range(max_iterations):
                interference = 0.0
                for j in range(n):
                    if taskset.priority[j] >= taskset.priority[i]:
                        continue  # j not in hp(i)
                    num = _num_jobs(taskset.T[j], w)
                    if taskset.criticality[j] == "HI":
                        # HI-criticality interference at C_j(HI)
                        interference += num * taskset.C_hi[j]
                    else:
                        # LO-criticality interference capped at R_j(LO)
                        num_hi = num
                        num_lo = math.ceil(r_lo[i] / taskset.T[j])
                        interference += min(num_hi, num_lo) * taskset.C_lo[j]
                w_new = taskset.C_hi[i] + interference
                if w_new == w or w_new > taskset.D[i] * 2:
                    break
                w = w_new
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
        interference = 0.0
        for j in range(len(T)):
            if priority[j] >= priority[i]:
                continue
            num = max(0, math.ceil(w / T[j]))
            interference += num * c_max(j)
        w_new = c_max(i) + interference
        if w_new == w or w_new > D[i] * 2:
            break
        w = w_new
    return w


def is_nontrivial_amc_taskset(taskset: TaskSet) -> bool:
    """Check if a task set genuinely needs the mixed-criticality scheme.

    A task set is "non-trivial" if:
    (a) It is schedulable under AMC-rtb, AND
    (b) At least one HI-criticality task is NOT schedulable under
        plain fixed-priority analysis assuming every task executes at
        max(C_i(LO), C_i(HI)).

    This follows the methodology in the AMC-RH paper Section V-C.

    Args:
        taskset: Task set with priorities assigned.

    Returns:
        True if the task set is non-trivial (needs AMC to be schedulable).
    """
    from amc_tasksim.scheduling.priority import assign_deadline_monotonic

    # Ensure priorities are assigned
    if not taskset.priority:
        assign_deadline_monotonic(taskset)

    # Check (a): schedulable under AMC-rtb
    result = amc_rtb(taskset)
    if not result.overall_schedulable:
        return False

    # Check (b): at least one HI task unschedulable under plain FP
    for i in range(taskset.n):
        if taskset.criticality[i] == "HI":
            fp_rt = _fp_response_time_max(taskset, i)
            if fp_rt > taskset.D[i]:
                return True

    return False
