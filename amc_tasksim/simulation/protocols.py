"""AMC-RH and AMC-RA mode-change protocols for AMC scheduling.

These extend the OriginalAMC protocol with analysis-runtime co-design
trigger conditions from the AMC-RH paper (RTAS 2022).

AMC-RH:
  Enter degraded mode: when an active HI-criticality job tau_i reaches
  R_i(LO) from the start of the priority level-i busy period in which
  it was released.
  Exit degraded mode: when there is no active HI-criticality job that
  has reached R_i(LO) from its busy period start.

AMC-RA:
  Enter degraded mode: same as AMC-RH.
  Exit degraded mode: on an idle instant (as in original AMC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import amc_rtb
from amc_tasksim.simulation.engine import Job, ModeChangeProtocol


@dataclass
class _BusyPeriodState:
    """Per-task busy period tracking for O(1) AMC-RH/AMC-RA bookkeeping.

    Attributes:
        busy_start: Start time of the current priority level-i busy period
            for this task. 0 means not currently active.
        busy_remaining: Remaining execution time in the current busy period.
    """

    busy_start: float = 0.0
    busy_remaining: float = 0.0


class AMC_RH(ModeChangeProtocol):
    """AMC-RH mode-change protocol.

    Enter degraded mode: when an active HI-criticality job tau_i reaches
    R_i(LO) from the start of the priority level-i busy period in which
    it was released.

    Exit degraded mode: when there is no active HI-criticality job that
    has reached R_i(LO) from its busy period start.
    """

    def __init__(self, r_lo: list[float]):
        """Initialize with precomputed Ri(LO) values.

        Args:
            r_lo: Per-task LO-criticality response times from AMC-rtb.
        """
        self.r_lo = r_lo

    def check_enter_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        for job in active_jobs:
            if job.criticality == "HI" and not job.completed and not job.dropped:
                executed = job.c_hi - job.remaining
                r_lo_i = self.r_lo[job.task_id]
                if executed >= r_lo_i:
                    return True
        return False

    def check_exit_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        """Exit degraded mode when no active HI-criticality job has
        reached R_i(LO) from its busy period start."""
        for job in active_jobs:
            if job.criticality == "HI" and not job.completed and not job.dropped:
                executed = job.c_hi - job.remaining
                r_lo_i = self.r_lo[job.task_id]
                if executed >= r_lo_i:
                    return False
        return True


class AMC_RA(ModeChangeProtocol):
    """AMC-RA mode-change protocol.

    Enter degraded mode: same as AMC-RH (based on R_i(LO) from busy
    period start).

    Exit degraded mode: on an idle instant (as in original AMC).
    """

    def __init__(self, r_lo: list[float]):
        """Initialize with precomputed Ri(LO) values.

        Args:
            r_lo: Per-task LO-criticality response times from AMC-rtb.
        """
        self.r_lo = r_lo

    def check_enter_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        for job in active_jobs:
            if job.criticality == "HI" and not job.completed and not job.dropped:
                executed = job.c_hi - job.remaining
                r_lo_i = self.r_lo[job.task_id]
                if executed >= r_lo_i:
                    return True
        return False

    def check_exit_degraded(
        self,
        active_jobs: list[Job],
        current_time: int,
    ) -> bool:
        """Exit degraded mode on idle instant (same as OriginalAMC)."""
        return len(active_jobs) == 0
