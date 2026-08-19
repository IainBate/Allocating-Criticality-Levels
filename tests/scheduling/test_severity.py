"""Properties of the severity trigger ladder.

The multi-level scheme replaces the two-level trigger with a ladder of
thresholds R_i(x), one per degradation level, where x is a severity in [0, 1].
Three properties make that ladder safe, and they are the whole safety argument,
so they are tested over generated task sets rather than asserted in a document:

A. ``R_i(0) == R_i(LO)``  -- level 1 coincides with the AMC-RH trigger, so
   HI-criticality protection begins no later than under two-level AMC.
B. ``x1 <= x2  =>  R_i(x1) <= R_i(x2)``  -- the ladder is ordered and never
   crosses, so "deeper level" always means "later trigger".
C. ``R_i(x) >= R_i(LO)`` for every x -- no level fires before AMC-RH's trigger,
   so the multi-level drop set is at every instant a subset of AMC-RH's and
   JNE cannot exceed it.

Property C is what the earlier fraction-of-R_i(LO) design violated by
construction, and violating it is what made intermediate levels fire under
ordinary load with no fault present.
"""

from __future__ import annotations

import math

import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import (
    amc_rtb,
    busy_period_bound,
    normal_mode_schedulable,
    response_times_at_budget,
    severity_budgets,
    severity_trigger,
)
from amc_tasksim.scheduling.priority import assign_deadline_monotonic

# Severities spanning the range, including both endpoints and a fine step near
# zero where the thresholds are closest together and crossings would show first.
SEVERITIES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def population(count: int = 40, **kwargs) -> list[TaskSet]:
    """R1-schedulable task sets, the population the multi-level scheme targets."""
    params = dict(n=10, CP=0.5, U=0.7, CF=2.0)
    params.update(kwargs)
    out: list[TaskSet] = []
    for seed in range(3000):
        ts = generate_taskset(rng_seed=seed, **params)
        assign_deadline_monotonic(ts)
        if normal_mode_schedulable(ts) and busy_period_bound(ts) is not None:
            out.append(ts)
        if len(out) == count:
            break
    assert len(out) == count, f"only found {len(out)} suitable task sets"
    return out


@pytest.fixture(scope="module")
def tasksets() -> list[TaskSet]:
    return population()


# ---------------------------------------------------------------------------
# severity_budgets
# ---------------------------------------------------------------------------


def test_budgets_interpolate_between_the_two_criticality_levels(tasksets):
    for ts in tasksets:
        assert severity_budgets(ts, 0.0) == ts.C_lo
        assert severity_budgets(ts, 1.0) == [
            max(ts.C_lo[i], ts.C_hi[i]) for i in range(ts.n)
        ]


def test_budgets_never_fall_below_the_lo_budget(tasksets):
    """Clamping matters: rounding must not produce a budget under C_i(LO).

    A budget below C_i(LO) would give a threshold below R_i(LO), which is
    exactly the property-C violation the ladder exists to avoid.
    """
    for ts in tasksets:
        for x in SEVERITIES:
            budgets = severity_budgets(ts, x)
            assert all(budgets[i] >= ts.C_lo[i] for i in range(ts.n))


def test_budgets_are_monotone_in_severity(tasksets):
    for ts in tasksets:
        for lo, hi in zip(SEVERITIES, SEVERITIES[1:]):
            a, b = severity_budgets(ts, lo), severity_budgets(ts, hi)
            assert all(a[i] <= b[i] for i in range(ts.n))


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_severity_outside_the_unit_interval_is_rejected(bad):
    ts = population(1)[0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        severity_budgets(ts, bad)


# ---------------------------------------------------------------------------
# Property A: level 1 is the AMC-RH trigger
# ---------------------------------------------------------------------------


def test_property_a_severity_zero_reproduces_r_lo(tasksets):
    """R_i(0) must equal R_i(LO) exactly, not approximately."""
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        at_zero = severity_trigger(ts, 0.0)
        assert [int(v) for v in at_zero] == r_lo, f"seed {ts.metadata['seed']}"


# ---------------------------------------------------------------------------
# Property B: the ladder is ordered
# ---------------------------------------------------------------------------


def test_property_b_thresholds_are_monotone_in_severity(tasksets):
    for ts in tasksets:
        rows = [severity_trigger(ts, x) for x in SEVERITIES]
        for (xa, a), (xb, b) in zip(
            zip(SEVERITIES, rows), zip(SEVERITIES[1:], rows[1:])
        ):
            for i in range(ts.n):
                assert a[i] <= b[i], (
                    f"ladder crosses on task {i} between x={xa} and x={xb}: "
                    f"{a[i]} > {b[i]}"
                )


def test_unreachable_levels_are_infinite_not_truncated():
    """The saturation that makes property B hold.

    The AMC-rtb recurrence stops iterating once w exceeds D_i. Reusing that
    truncated value as a trigger time lets a larger budget stop sooner and
    report a *smaller* threshold, which breaks monotonicity. A level a task
    cannot reach is +inf -- a trigger that never fires.
    """
    # C_hi is large enough that the HI task's recurrence blows past its deadline.
    ts = TaskSet(
        n=2,
        criticality=["HI", "LO"],
        T=[100, 100],
        D=[100, 100],
        C_lo=[10, 10],
        C_hi=[95, 10],
        BCET=[10, 10],
    )
    assign_deadline_monotonic(ts)

    assert all(math.isfinite(v) for v in severity_trigger(ts, 0.0))
    top = severity_trigger(ts, 1.0)
    assert any(math.isinf(v) for v in top), "expected an unreachable level"
    # Infinite, so still monotone against every finite lower level.
    for x in SEVERITIES:
        lower = severity_trigger(ts, x)
        assert all(lower[i] <= top[i] for i in range(ts.n))


def test_unreachable_levels_occur_in_the_real_population(tasksets):
    """Saturation is a common case, not a corner one, so it must be handled.

    The effective number of levels therefore varies per task: a nominal k=5
    ladder may be shallower for some tasks, which is why the effective level
    count has to be reported alongside the nominal one.
    """
    total = unreachable = 0
    for ts in tasksets:
        for x in SEVERITIES:
            for v in severity_trigger(ts, x):
                total += 1
                unreachable += math.isinf(v)
    assert unreachable > 0, "expected some unreachable levels in this population"
    assert unreachable / total < 0.5, "population is too degenerate to be useful"


# ---------------------------------------------------------------------------
# Property C: no level precedes the AMC-RH trigger
# ---------------------------------------------------------------------------


def test_property_c_no_threshold_precedes_r_lo(tasksets):
    """The property the fraction-of-R_lo design violated by construction."""
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        for x in SEVERITIES:
            row = severity_trigger(ts, x)
            for i in range(ts.n):
                assert row[i] >= r_lo[i], (
                    f"task {i} at x={x}: threshold {row[i]} precedes "
                    f"R_i(LO)={r_lo[i]}"
                )


def test_fraction_of_r_lo_would_violate_property_c(tasksets):
    """Guards the rationale, so the rejected design cannot quietly return.

    Documenting *why* the ladder interpolates the budget rather than the
    deadline is only useful if the alternative's defect is pinned down.
    """
    for ts in tasksets[:5]:
        r_lo = amc_rtb(ts).r_lo
        for frac in (0.5, 0.9):
            scaled = [r * frac for r in r_lo]
            assert any(scaled[i] < r_lo[i] for i in range(ts.n) if r_lo[i] > 0)


# ---------------------------------------------------------------------------
# response_times_at_budget directly
# ---------------------------------------------------------------------------


def test_charging_c_lo_reproduces_the_standard_analysis(tasksets):
    for ts in tasksets:
        assert [int(v) for v in response_times_at_budget(ts, ts.C_lo)] == amc_rtb(ts).r_lo


def test_higher_budgets_never_lower_a_threshold(tasksets):
    """Monotonicity in the budget vector itself, independent of the severity map."""
    for ts in tasksets[:15]:
        base = response_times_at_budget(ts, ts.C_lo)
        bumped = response_times_at_budget(ts, [c + 1 for c in ts.C_lo])
        for i in range(ts.n):
            assert base[i] <= bumped[i]


@pytest.mark.parametrize("U", [0.5, 0.6, 0.8])
def test_properties_hold_across_utilisations(U):
    """The three properties are structural, so they must not depend on load."""
    for ts in population(count=8, U=U):
        r_lo = amc_rtb(ts).r_lo
        rows = [severity_trigger(ts, x) for x in SEVERITIES]
        assert [int(v) for v in rows[0]] == r_lo                      # A
        for a, b in zip(rows, rows[1:]):                              # B
            assert all(a[i] <= b[i] for i in range(ts.n))
        for row in rows:                                              # C
            assert all(row[i] >= r_lo[i] for i in range(ts.n))
