"""Priority assignment for AMC task sets.

Supports Deadline Monotonic Priority Assignment (DMPA), the optimal
fixed-priority assignment for constrained-deadline systems.
"""

from __future__ import annotations

from amc_tasksim.generation.taskset import TaskSet


def assign_deadline_monotonic(taskset: TaskSet) -> TaskSet:
    """Assign priorities by Deadline Monotonic Priority Assignment (DMPA).

    Lower deadline = higher priority (assigned as a smaller priority number).
    Ties are broken by task index (lower index = higher priority).

    Args:
        taskset: Task set with priority list to be filled.

    Returns:
        The same taskset with priority assigned.
    """
    # Create index list sorted by (deadline, index)
    indices = list(range(taskset.n))
    indices.sort(key=lambda i: (taskset.D[i], i))

    # Assign priority: first in sorted order gets priority 0 (highest)
    priority = [0] * taskset.n
    for rank, idx in enumerate(indices):
        priority[idx] = rank

    taskset.priority = priority
    return taskset
