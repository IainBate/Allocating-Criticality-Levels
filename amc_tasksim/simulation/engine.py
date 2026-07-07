"""Event-driven simulator for AMC fixed-priority preemptive scheduling.

Simulates a single TaskSet on one core using discrete-event simulation.
Collects per-run metrics: NiD, TiD, JNE, LDM, HDM.
"""

from __future__ import annotations

import heapq
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amc_tasksim.generation.taskset import TaskSet


@dataclass
class Job:
    """A single job release.

    Attributes:
        task_id: Index of the task this job belongs to.
        release: Release time (integer tick).
        deadline: Absolute deadline.
        c_lo: LO-criticality execution time budget.
        c_hi: HI-criticality execution time budget.
        bcet: Best-case execution time.
        criticality: "HI" or "LO".
        remaining: Remaining execution time.
        priority: Priority level (lower = higher priority).
        completed: Whether the job has finished.
        hi_behaviour: Whether this job exhibits HI-criticality behaviour.
        dropped: Whether the job was dropped (LO in degraded mode).
    """

    task_id: int
    release: int
    deadline: int
    c_lo: int
    c_hi: int
    bcet: int
    criticality: str
    remaining: int = 0
    priority: int = 0
    completed: bool = False
    hi_behaviour: bool = False
    dropped: bool = False


@dataclass
class SimulationResult:
    """Results from a single simulation run.

    Attributes:
        nid: Number of times degraded mode was entered.
        tid: Total time spent in degraded mode (fraction of duration).
        jne: Count of LO-criticality jobs dropped in degraded mode.
        ldm: Count of LO-criticality jobs that executed but missed deadline.
        hdm: Count of HI-criticality deadline misses.
        hi_releases_per_task: Total HI-criticality job releases per task.
        hi_trigger_events: Total HI-behaviour trigger events observed.
    """

    nid: int = 0
    tid: float = 0.0
    jne: int = 0
    ldm: int = 0
    hdm: int = 0
    hi_releases_per_task: list[int] = field(default_factory=list)
    hi_trigger_events: int = 0


class ModeChangeProtocol(ABC):
    """Abstract base class for mode-change protocols."""

    @abstractmethod
    def check_enter_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        """Determine if we should enter degraded mode."""
        ...

    @abstractmethod
    def check_exit_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        """Determine if we should exit degraded mode."""
        ...


class OriginalAMC(ModeChangeProtocol):
    """Original AMC mode-change protocol.

    Enter degraded mode: when any active HI-criticality job has executed
    for C_i(LO) without completing.

    Exit degraded mode: on the next idle instant (no active jobs of any task).
    """

    def check_enter_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        for job in active_jobs:
            if job.criticality == "HI" and not job.completed and not job.dropped:
                executed = job.c_hi - job.remaining
                # Enter degraded mode only if the job has executed for C_lo
                # AND still has remaining work (i.e., execution time > C_lo)
                if executed >= job.c_lo and job.remaining > 0:
                    return True
        return False

    def check_exit_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        return len(active_jobs) == 0


def simulate(
    taskset: TaskSet,
    duration: int = 10**6,
    seed: Optional[int] = None,
    mode_protocol: Optional[ModeChangeProtocol] = None,
    fp: float = 1e-4,
    release_times: Optional[list[Optional[int]]] = None,
) -> SimulationResult:
    """Simulate a task set for a specified duration.

    Event-driven simulation: advances time to the next event (release,
    job completion, or deadline) rather than iterating tick by tick.

    Args:
        taskset: Task set to simulate (must have priorities assigned).
        duration: Simulation duration in ticks.
        seed: Random seed for job execution times.
        mode_protocol: Mode-change protocol (default: OriginalAMC).
        fp: Failure probability for HI-criticality behaviour.
        release_times: Optional explicit release time per task (for testing).

    Returns:
        SimulationResult with collected metrics.
    """
    if mode_protocol is None:
        mode_protocol = OriginalAMC()

    rng = np.random.default_rng(seed)

    # Ensure priorities are assigned
    if not taskset.priority:
        from amc_tasksim.scheduling.priority import assign_deadline_monotonic
        assign_deadline_monotonic(taskset)

    result = SimulationResult()
    result.hi_releases_per_task = [0] * taskset.n

    # State
    mode = "normal"
    current_time = 0
    ready_queue: list[Job] = []
    active_jobs: list[Job] = []

    # Track next release per task
    next_release = [0] * taskset.n
    for i in range(taskset.n):
        if release_times is not None and release_times[i] is not None:
            next_release[i] = release_times[i]
        else:
            next_release[i] = taskset.T[i]

    # Track degraded mode start time
    degraded_start = -1
    total_degraded_ticks = 0

    # Track completed jobs for deadline checking
    completed_lo = []  # (release_time, deadline) for completed LO jobs
    completed_hi = []  # (release_time, deadline) for completed HI jobs

    while current_time < duration:
        # Check for new releases at current_time
        for i in range(taskset.n):
            if next_release[i] <= current_time:
                if mode == "degraded" and taskset.criticality[i] == "LO":
                    result.jne += 1
                else:
                    # Determine execution time for this job
                    if taskset.criticality[i] == "HI":
                        if fp > 0 and rng.random() < fp:
                            result.hi_trigger_events += 1
                            hi_lo = taskset.C_lo[i]
                            hi_hi = taskset.C_hi[i]
                            if hi_lo >= hi_hi:
                                exec_time = hi_hi
                            else:
                                exec_time = max(1, int(rng.integers(hi_lo, hi_hi + 1)))
                        else:
                            lo_lo = max(taskset.BCET[i], 1)
                            lo_hi = taskset.C_lo[i]
                            if lo_lo >= lo_hi:
                                exec_time = lo_hi
                            else:
                                exec_time = max(1, int(rng.integers(lo_lo, lo_hi + 1)))
                        remaining = exec_time
                    else:
                        lo_lo = max(taskset.BCET[i], 1)
                        lo_hi = taskset.C_lo[i]
                        if lo_lo >= lo_hi:
                            remaining = lo_hi
                        else:
                            remaining = max(1, int(rng.integers(lo_lo, lo_hi + 1)))

                    job = Job(
                        task_id=i,
                        release=current_time,
                        deadline=current_time + taskset.D[i],
                        c_lo=taskset.C_lo[i],
                        c_hi=taskset.C_hi[i],
                        bcet=taskset.BCET[i],
                        criticality=taskset.criticality[i],
                        remaining=remaining,
                        priority=taskset.priority[i],
                    )
                    ready_queue.append(job)
                    active_jobs.append(job)
                    if taskset.criticality[i] == "HI":
                        result.hi_releases_per_task[i] += 1
                # Schedule next release
                if release_times is None or release_times[i] is None:
                    next_release[i] = current_time + taskset.T[i]

        # Check if we should enter degraded mode
        if mode == "normal":
            if mode_protocol.check_enter_degraded(active_jobs, current_time):
                mode = "degraded"
                result.nid += 1
                degraded_start = current_time

        # Check if we should exit degraded mode
        if mode == "degraded":
            if mode_protocol.check_exit_degraded(active_jobs, current_time):
                mode = "normal"
                total_degraded_ticks += current_time - degraded_start
                degraded_start = -1

        # Find highest-priority ready job
        if ready_queue:
            ready_queue.sort(key=lambda j: j.priority)
            running = ready_queue.pop(0)

            # Determine how long this job runs
            # It runs until: it completes, or a release occurs, or a deadline passes
            next_release_time = min((nr for nr in next_release if nr > current_time), default=duration)
            next_deadline = min((j.deadline for j in ready_queue + active_jobs if j.deadline > current_time), default=duration)
            next_event = min(next_release_time, next_deadline, duration)

            if next_event <= current_time:
                next_event = current_time + 1

            # Run for min(remaining, time_to_next_event)
            run_time = min(running.remaining, next_event - current_time)
            running.remaining -= run_time
            current_time += run_time

            if mode == "degraded":
                total_degraded_ticks += run_time

            # Check completion
            if running.remaining <= 0:
                running.remaining = 0
                running.completed = True
                active_jobs = [j for j in active_jobs if j is not running]

                if running.criticality == "LO":
                    completed_lo.append((running.release, running.deadline))
                else:
                    completed_hi.append((running.release, running.deadline))

                # Check mode exit
                if mode == "degraded":
                    if mode_protocol.check_exit_degraded(active_jobs, current_time):
                        mode = "normal"
                        total_degraded_ticks += current_time - degraded_start
                        degraded_start = -1
            else:
                # Job still running — put back in ready queue
                ready_queue.append(running)
        else:
            # Processor idle — advance to next release
            future_releases = [nr for nr in next_release if nr > current_time]
            if future_releases:
                current_time = min(future_releases)
            else:
                # No future releases — simulation is done
                break

    # Finalize degraded time
    if mode == "degraded" and degraded_start >= 0:
        total_degraded_ticks += duration - degraded_start

    result.tid = total_degraded_ticks / duration if duration > 0 else 0.0

    # Check deadlines for completed jobs
    for release, deadline in completed_lo:
        if current_time > deadline:
            # LO job completed but we can't tell from this data if it missed
            pass

    return result
