"""Tests for AMC-rtb response-time analysis.

Phase 4 acceptance criteria:
- Reproduce the worked 3-task example from AMC-RH Appendix A:
  R3(LO) = 10, R3(HI) = 19 (tau3 unschedulable).
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import amc_rtb, is_nontrivial_amc_taskset, ResponseTimeResult
from amc_tasksim.scheduling.priority import assign_deadline_monotonic


def _make_taskset():
    """Build the Appendix A example task set from the AMC-RH paper.

    tau1: C_lo=1, T=2, D=2, LO
    tau2: C_lo=1, C_hi=5, T=10, D=10, HI
    tau3: C_lo=C_hi=4, T=100, D=18, HI
    """
    return TaskSet(
        n=3,
        criticality=["LO", "HI", "HI"],
        T=[2, 10, 100],
        D=[2, 10, 18],
        C_lo=[1, 1, 4],
        C_hi=[1, 5, 4],
        BCET=[1, 1, 4],
        priority=[],
        metadata={"source": "AMC-RH Appendix A"},
    )


# ---------------------------------------------------------------------------
# Worked example from AMC-RH Appendix A
# ---------------------------------------------------------------------------

def test_appendix_a_response_times():
    """Reproduce the Appendix A example: R3(LO)=10, R3(HI)=19."""
    ts = _make_taskset()
    assign_deadline_monotonic(ts)
    result = amc_rtb(ts)

    # tau3 is index 2 (lowest priority, D=18)
    assert result.r_lo[2] == 10, f"R3(LO)={result.r_lo[2]}, expected 10"
    assert result.r_hi[2] == 19, f"R3(HI)={result.r_hi[2]}, expected 19"


def test_appendix_a_unschedulable():
    """tau3 should be unschedulable: R3(HI)=19 > D3=18."""
    ts = _make_taskset()
    assign_deadline_monotonic(ts)
    result = amc_rtb(ts)

    assert not result.schedulable_hi[2], (
        f"tau3 should be unschedulable: R3(HI)={result.r_hi[2]} > D3=18"
    )
    assert not result.overall_schedulable


def test_appendix_a_priority_order():
    """DMPA should assign: tau1 (priority 0) > tau2 (priority 1) > tau3 (priority 2)."""
    ts = _make_taskset()
    assign_deadline_monotonic(ts)
    assert ts.priority == [0, 1, 2], f"Priority order {ts.priority}, expected [0, 1, 2]"


def test_appendix_a_tau1_tau2_schedulable():
    """tau1 and tau2 should be schedulable."""
    ts = _make_taskset()
    assign_deadline_monotonic(ts)
    result = amc_rtb(ts)

    assert result.schedulable_lo[0], f"tau1 LO response time {result.r_lo[0]} > D1=2"
    assert result.schedulable_lo[1], f"tau2 LO response time {result.r_lo[1]} > D2=10"
    assert result.schedulable_hi[1], f"tau2 HI response time {result.r_hi[1]} > D2=10"


# ---------------------------------------------------------------------------
# Schedulable task set
# ---------------------------------------------------------------------------

def test_schedulable_taskset():
    """A simple schedulable task set should pass."""
    ts = TaskSet(
        n=2,
        criticality=["HI", "HI"],
        T=[10, 20],
        D=[10, 20],
        C_lo=[3, 5],
        C_hi=[5, 8],
        BCET=[2, 4],
        priority=[],
        metadata={},
    )
    assign_deadline_monotonic(ts)
    result = amc_rtb(ts)
    assert result.overall_schedulable


# ---------------------------------------------------------------------------
# Non-trivial AMC filter
# ---------------------------------------------------------------------------

def test_nontrivial_filter():
    """Appendix A example should be non-trivial (schedulable under AMC, not under plain FP)."""
    ts = _make_taskset()
    assign_deadline_monotonic(ts)
    # The Appendix A example has R3(HI)=19 > D3=18, so it's NOT schedulable under AMC-rtb
    # Therefore is_nontrivial should return False (it fails condition (a))
    assert not is_nontrivial_amc_taskset(ts)


def test_nontrivial_filter_schedulable_amc():
    """A task set that is schedulable under AMC but not plain FP should be non-trivial."""
    # Modify tau3's deadline to 20 (so AMC-rtb passes: R3(HI)=19 <= 20)
    ts = _make_taskset()
    ts.D = [2, 10, 20]
    ts.priority = []
    assign_deadline_monotonic(ts)
    result = amc_rtb(ts)
    assert result.overall_schedulable, "Modified task set should be schedulable under AMC-rtb"
    assert is_nontrivial_amc_taskset(ts), "Should be non-trivial: schedulable under AMC, not under plain FP"
