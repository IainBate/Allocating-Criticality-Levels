"""Priority assignment for AMC task sets.

Two schemes:

- **Deadline monotonic** (DMPA), optimal for constrained-deadline task sets
  under single-criticality fixed-priority preemptive scheduling.
- **Audsley's Optimal Priority Assignment** (OPA), which is what both papers
  use. Deadline monotonic is *not* optimal for AMC-rtb; OPA is, and the
  difference in the fraction of task sets deemed schedulable is large at high
  utilisation.
"""

from __future__ import annotations

from typing import Optional

from amc_tasksim.generation.taskset import TaskSet


def assign_deadline_monotonic(taskset: TaskSet) -> TaskSet:
    """Assign priorities by Deadline Monotonic Priority Assignment (DMPA).

    Lower deadline = higher priority (assigned a smaller priority number).
    Ties are broken by task index.

    Args:
        taskset: Task set whose priority list is to be filled.

    Returns:
        The same taskset, with priorities assigned.
    """
    indices = sorted(range(taskset.n), key=lambda i: (taskset.D[i], i))
    priority = [0] * taskset.n
    for rank, idx in enumerate(indices):
        priority[idx] = rank
    taskset.priority = priority
    return taskset


def assign_audsley_opa(taskset: TaskSet, test=None) -> bool:
    """Assign priorities by Audsley's Optimal Priority Assignment.

    Working from the lowest priority level upwards, the algorithm looks for any
    unassigned task that is schedulable at that level given that every other
    unassigned task has higher priority. If some task fits at every level, the
    resulting assignment is optimal with respect to the test; if no task fits at
    some level, no fixed-priority assignment makes the task set schedulable.

    This is valid because the AMC-rtb test depends on the higher-priority tasks
    only as a set, not on their relative order.

    Args:
        taskset: Task set to assign priorities to. On success `taskset.priority`
            holds the assignment; on failure it holds a deadline-monotonic
            ordering so the task set can still be simulated.
        test: Single-task schedulability test `(taskset, i, hp) -> bool`, taking
            the index of the task under test and the set of higher-priority task
            indices. Defaults to the AMC-rtb test.

    Returns:
        True if a complete assignment was found.
    """
    if test is None:
        from amc_tasksim.scheduling.amc_rtb import amc_rtb_single

        test = amc_rtb_single

    n = taskset.n
    unassigned = set(range(n))
    priority: list[Optional[int]] = [None] * n

    for level in range(n - 1, -1, -1):
        for candidate in sorted(unassigned):
            hp = unassigned - {candidate}
            if test(taskset, candidate, hp):
                priority[candidate] = level
                unassigned.discard(candidate)
                break
        else:
            # No task is schedulable at this priority level.
            assign_deadline_monotonic(taskset)
            return False

    taskset.priority = [p if p is not None else 0 for p in priority]
    return True
