"""Mode-change protocols for AMC scheduling.

The protocol classes themselves live in :mod:`amc_tasksim.simulation.engine`,
alongside the scheduler state they operate on. This module re-exports them so
that ``from amc_tasksim.simulation.protocols import AMC_RH, AMC_RA`` keeps
working, and documents the three schemes in one place.

===============  ==========================================  ==========================================
Scheme           Enter degraded mode                         Exit degraded mode
===============  ==========================================  ==========================================
OriginalAMC      a HI job has executed for C_i(LO) without   idle instant
                 signalling completion
AMC-RA           an active HI job reaches R_i(LO) after the  idle instant
                 start of its priority level-i busy period
AMC-RH           as AMC-RA                                   a HI job completes and no active HI job
                                                             has reached R_k(LO) from its busy period
===============  ==========================================  ==========================================

AMC-RH and AMC-RA are specifications S1-S5 of "Analysis-Runtime Co-design for
Adaptive Mixed-Criticality Scheduling" (RTAS 2022), Section IV-B. Both take the
per-task R_i(LO) values produced by :func:`amc_tasksim.scheduling.amc_rtb.amc_rtb`.
"""

from __future__ import annotations

from amc_tasksim.simulation.engine import (
    AMC_RA,
    AMC_RH,
    ModeChangeProtocol,
    OriginalAMC,
    SchedulerState,
)

__all__ = [
    "AMC_RA",
    "AMC_RH",
    "ModeChangeProtocol",
    "OriginalAMC",
    "SchedulerState",
]
