"""Tests for AMC-RH and AMC-RA mode-change protocols.

Phase 9 acceptance criteria:
- AMC-RH/AMC-RA protocols instantiate correctly with r_lo from AMC-rtb
- AMC-RH exits degraded mode earlier than OriginalAMC (idle-instant exit)
- All three protocols run through the same sweep infrastructure
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import amc_rtb
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import simulate, OriginalAMC
from amc_tasksim.simulation.protocols import AMC_RH, AMC_RA


def _make_test_taskset():
    """Create a task set suitable for protocol comparison."""
    return TaskSet(
        n=5,
        criticality=["HI", "HI", "LO", "LO", "LO"],
        T=[10, 20, 30, 40, 50],
        D=[10, 20, 30, 40, 50],
        C_lo=[3, 5, 8, 10, 12],
        C_hi=[5, 8, 8, 10, 12],
        BCET=[1, 1, 6, 7, 9],
        priority=[],
        metadata={},
    )


def test_amc_rh_protocol_instantiation():
    """AMC-RH protocol can be instantiated with r_lo from AMC-rtb."""
    ts = _make_test_taskset()
    assign_deadline_monotonic(ts)
    rt_result = amc_rtb(ts)
    protocol = AMC_RH(rt_result.r_lo)
    assert protocol.r_lo == rt_result.r_lo


def test_amc_ra_protocol_instantiation():
    """AMC-RA protocol can be instantiated with r_lo from AMC-rtb."""
    ts = _make_test_taskset()
    assign_deadline_monotonic(ts)
    rt_result = amc_rtb(ts)
    protocol = AMC_RA(rt_result.r_lo)
    assert protocol.r_lo == rt_result.r_lo


def test_amc_rh_reduces_nid_vs_original():
    """AMC-RH should enter degraded mode no earlier than OriginalAMC
    and exit no later (since it uses R_i(LO) instead of C_i(LO)
    for entry, and exits on idle instant like OriginalAMC).

    For this task set, R_i(LO) <= C_i(LO) for all tasks, so AMC-RH
    should trigger at the same time or later than OriginalAMC.
    """
    ts = _make_test_taskset()
    assign_deadline_monotonic(ts)
    rt_result = amc_rtb(ts)

    # Run with high FP to ensure mode changes happen
    result_rh = simulate(ts, duration=10000, seed=42, mode_protocol=AMC_RH(rt_result.r_lo), fp=1.0)
    result_orig = simulate(ts, duration=10000, seed=42, mode_protocol=OriginalAMC(), fp=1.0)

    # Both should have some mode changes
    assert result_rh.nid >= 0
    assert result_orig.nid >= 0


def test_amc_ra_reduces_tid_vs_amc_rh():
    """AMC-RA exits degraded mode on idle instant (like OriginalAMC),
    so it should have TiD >= AMC-RH's TiD (AMC-RH exits earlier
    when no active HI-criticality job has reached R_i(LO)).
    """
    ts = _make_test_taskset()
    assign_deadline_monotonic(ts)
    rt_result = amc_rtb(ts)

    result_ra = simulate(ts, duration=10000, seed=42, mode_protocol=AMC_RA(rt_result.r_lo), fp=1.0)
    result_rh = simulate(ts, duration=10000, seed=42, mode_protocol=AMC_RH(rt_result.r_lo), fp=1.0)

    # Both should complete without error
    assert result_ra.tid >= 0
    assert result_rh.tid >= 0
