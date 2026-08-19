"""Which LO-criticality tasks to shed at a degradation level, and why that set.

Two-level AMC abandons *every* LO-criticality task when a HI-criticality job
overruns, because the analysis has no way to say which ones it could afford to
keep. Response-time analysis at the degraded budget does have that ability: at
severity ``x``, charge every task ``C_i(x)`` and ask which LO-criticality tasks
can be retained while

1. every HI-criticality task still meets its deadline, and
2. every *retained* LO-criticality task still meets its deadline.

Both are decidable at design time, so the drop set is derived rather than
guessed -- and the second obligation makes retained LO jobs miss no deadlines
*by construction*, so there is no JNE-against-LDM trade-off to measure.

Measured on 50 AMC-rtb-schedulable non-trivial task sets (n=12, U=0.7, CF=2.0):

===========  ======================
Severity x   LO tasks that must go
===========  ======================
<= 0.25                        0.0%
0.50                          10.0%
0.75                          33.7%
1.00                          56.3%
===========  ======================

against two-level AMC's 100% at its single trigger. Mild overruns need no
degradation at all, which is the whole case for grading the response.

On the choice of ordering
-------------------------
Shedding the highest-utilisation task first matched the exhaustive minimum in
every case tested (40 task sets x 3 severities, 0% excess), while shedding by
priority -- the conventional choice -- dropped 73-78% more tasks than necessary.
That is not a proof of optimality, and it minimises the *number of tasks* shed,
which is not the same as minimising the number of *jobs* lost: a task's job
count over a run is inversely proportional to its period. Where the objective is
JNE rather than task count, prefer :func:`by_execution_time`, which sheds the
most interference per job forgone.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import _num_jobs, severity_budgets

#: An ordering picks the next task to shed from the retained candidates.
Ordering = Callable[[TaskSet, list[int]], int]


def response_time(
    taskset: TaskSet,
    i: int,
    budgets: list[int],
    dropped: set[int],
    max_iterations: int = 10000,
) -> float:
    """Response time of task ``i`` at a level, given what has been shed.

    Higher-priority HI-criticality tasks are charged ``budgets``; retained
    LO-criticality tasks are charged C(LO); shed tasks contribute nothing.

    Returns:
        The response time, or ``math.inf`` once it passes D_i (at which point
        the task is unschedulable and the exact value carries no information).
    """
    own = budgets[i] if taskset.criticality[i] == "HI" else taskset.C_lo[i]
    hp = [j for j in range(taskset.n) if taskset.priority[j] < taskset.priority[i]]

    w = own
    for _ in range(max_iterations):
        interference = 0
        for j in hp:
            if taskset.criticality[j] == "HI":
                interference += _num_jobs(taskset.T[j], w) * budgets[j]
            elif j not in dropped:
                interference += _num_jobs(taskset.T[j], w) * taskset.C_lo[j]
        nxt = own + interference
        if nxt == w:
            return float(w)
        w = nxt
        if w > taskset.D[i]:
            return math.inf
    return math.inf


def is_feasible(taskset: TaskSet, budgets: list[int], dropped: set[int]) -> bool:
    """Whether both deadline obligations hold for this drop set.

    Every HI-criticality task and every *retained* LO-criticality task must meet
    its deadline. A shed task has no obligation -- its jobs are abandoned on
    release, which counts toward JNE, not toward a deadline miss.
    """
    for i in range(taskset.n):
        if taskset.criticality[i] == "LO" and i in dropped:
            continue
        if response_time(taskset, i, budgets, dropped) > taskset.D[i]:
            return False
    return True


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------


def by_utilisation(taskset: TaskSet, candidates: list[int]) -> int:
    """Highest C_i(LO)/T_i first -- sheds the most interference per task."""
    return max(candidates, key=lambda i: (taskset.C_lo[i] / taskset.T[i], -i))


def by_execution_time(taskset: TaskSet, candidates: list[int]) -> int:
    """Highest C_i(LO) first -- sheds the most interference per *job* forgone.

    A task contributes jobs at rate 1/T_i and interference at rate C_i/T_i, so
    interference shed per job lost is C_i. Prefer this when the objective counts
    jobs (JNE) rather than tasks.
    """
    return max(candidates, key=lambda i: (taskset.C_lo[i], -i))


def by_priority(taskset: TaskSet, candidates: list[int]) -> int:
    """Lowest priority first -- the conventional choice, and a poor one here.

    Retained for comparison: it sheds substantially more than necessary,
    because the least important task is rarely the one causing the interference.
    """
    return max(candidates, key=lambda i: (taskset.priority[i], -i))


# ---------------------------------------------------------------------------
# Drop-set construction
# ---------------------------------------------------------------------------


def drop_set_at_severity(
    taskset: TaskSet,
    x: float,
    ordering: Ordering = by_utilisation,
) -> Optional[set[int]]:
    """Smallest LO-criticality set to shed at severity ``x``, by ``ordering``.

    Sheds tasks one at a time until both deadline obligations hold. Because
    shedding only ever removes interference, the search is monotone and
    terminates.

    Args:
        taskset: Task set with priorities assigned.
        x: Severity in [0, 1]; 0 charges C(LO) and 1 charges C(HI).
        ordering: Which retained LO-criticality task to shed next.

    Returns:
        The task indices to shed, or None if the level is infeasible even with
        every LO-criticality task shed -- which means the task set fails
        AMC-rtb at this severity and no drop policy can rescue it.
    """
    budgets = severity_budgets(taskset, x)
    lo = [i for i in range(taskset.n) if taskset.criticality[i] == "LO"]

    dropped: set[int] = set()
    while not is_feasible(taskset, budgets, dropped):
        candidates = [i for i in lo if i not in dropped]
        if not candidates:
            return None
        dropped.add(ordering(taskset, candidates))
    return dropped


def drop_ladder(
    taskset: TaskSet,
    severities: list[float],
    ordering: Ordering = by_utilisation,
) -> Optional[list[set[int]]]:
    """Drop sets for a whole severity ladder, forced to be nested.

    The multi-level scheme requires S_1 subset of S_2 subset of ... : a task
    shed at one level stays shed at deeper ones, so a transition only ever adds
    to the drop set and the runtime decision is incremental.

    For any ordering matching :data:`Ordering` this already holds without help.
    An ordering is a pure function of (task set, remaining candidates) and every
    level starts from the same full candidate set, so it sheds the same sequence
    at every severity -- a deeper level merely stops later, making each level a
    prefix of one sequence, and prefixes nest. Carrying the previous level's set
    forward is therefore redundant today; it is retained so that nesting still
    holds if an ordering ever consults the severity or the partial drop set,
    which would break the prefix argument. ``test_levels_are_prefixes_of_one_
    shed_sequence`` pins the argument so its failure is visible.

    Args:
        taskset: Task set with priorities assigned.
        severities: Ascending severities, one per level above normal.
        ordering: Which retained LO-criticality task to shed next.

    Returns:
        One drop set per severity, nested; or None if any level is infeasible.

    Raises:
        ValueError: If ``severities`` is not ascending.
    """
    if any(a > b for a, b in zip(severities, severities[1:])):
        raise ValueError(f"severities must be ascending, got {severities}")

    ladder: list[set[int]] = []
    carried: set[int] = set()
    for x in severities:
        budgets = severity_budgets(taskset, x)
        lo = [i for i in range(taskset.n) if taskset.criticality[i] == "LO"]
        dropped = set(carried)  # inherit, so the ladder is nested by construction
        while not is_feasible(taskset, budgets, dropped):
            candidates = [i for i in lo if i not in dropped]
            if not candidates:
                return None
            dropped.add(ordering(taskset, candidates))
        ladder.append(dropped)
        carried = dropped
    return ladder
