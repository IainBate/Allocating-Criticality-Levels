"""Tests for the cascade-exit opportunity measurement (exit_opportunity.py).

Small, fast checks that the grid wiring and stats are sane -- not a
re-verification of the underlying diagnostic, which is tested directly in
tests/simulation/test_multilevel.py (_natural_level and the overdegraded_*
accounting).
"""

from __future__ import annotations

import warnings

import pytest

from amc_tasksim.experiments.exit_opportunity import (
    EarlyExitCell,
    HoldoffPoint,
    HysteresisCell,
    OverdegradedCell,
    early_exit_trial,
    hold_off_sweep,
    hysteresis_sweep,
    overdegraded_opportunity,
)


def test_overdegraded_opportunity_produces_cells_with_sane_stats():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = overdegraded_opportunity(
            U_values=(0.7,), n_tasksets=6, seeds=(0, 1), duration=20_000,
        )
    assert cells, "no cells produced"
    ops = {c.operating_point for c in cells}
    assert ops == {"A", "B"}
    policies = {c.policy for c in cells}
    assert "shed_early" in policies
    for cell in cells:
        assert isinstance(cell, OverdegradedCell)
        assert cell.n_tasksets >= 2
        assert 0.0 <= cell.overdegraded_pct_mean <= 100.0
        assert 0.0 <= cell.overdegraded_level_pct_mean <= 100.0
        assert 0.0 <= cell.overdegraded_jne_pct_mean <= 100.0
        assert cell.overdegraded_pct_se >= 0.0
        assert cell.overdegraded_jne_pct_se >= 0.0
        assert cell.mean_events_per_run >= 0.0
        assert 0.0 <= cell.full_exit_pct_mean <= cell.overdegraded_jne_pct_mean + 1e-9
        assert cell.cascade_headroom_pct_mean >= -1e-9
        if cell.policy == "shed_early":
            # k=2: no intermediate level exists, so nothing is partial-only.
            assert cell.cascade_headroom_pct_mean < 1e-6
        assert cell.summary()  # does not raise, is non-empty


def test_early_exit_trial_produces_paired_cells():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = early_exit_trial(
            U_values=(0.7,), n_tasksets=6, seeds=(0, 1), duration=20_000,
        )
    assert cells, "no cells produced"
    ops = {c.operating_point for c in cells}
    assert ops == {"A", "B"}
    for cell in cells:
        assert isinstance(cell, EarlyExitCell)
        assert cell.n_tasksets >= 2
        assert cell.service_ratio.n == cell.n_tasksets
        assert cell.level_trans.n == cell.n_tasksets
        assert cell.tid.n == cell.n_tasksets
        assert cell.wasted_cpu_pct.n == cell.n_tasksets
        assert cell.summary()
    # Not asserted as a hard invariant: amc_rh admits LO work earlier, but an
    # admitted job can (via priority interference) pull a later HI job's busy
    # period earlier, triggering an additional escalation idle-exit would not
    # have had -- see safety_proof.md Corollary 2's scope note. Safety (HI
    # deadlines, no late LO completion) does not depend on this either way;
    # net service ratio and oscillation are empirical questions for the trial,
    # not proven directions, so this smoke test only checks structure.


def test_hold_off_sweep_frac_suppressed_increases_monotonically_with_hold_off():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points = hold_off_sweep(
            hold_offs=[0, 10, 100, 1000, 10**9],
            U=0.9, regime="implicit", n_tasksets=6, seeds=(0, 1), duration=50_000,
        )
    assert len(points) == 5
    for p in points:
        assert isinstance(p, HoldoffPoint)
        assert p.n_gaps > 0, "no exit-to-reentry gaps observed; check the population/regime"
        assert 0.0 <= p.frac_suppressed <= 1.0
        assert 0 <= p.lo_given_back <= p.lo_admitted_in_any_gap
        assert 0.0 <= p.lo_given_back_frac <= 1.0
    # Monotonic in hold_off: a longer hold-off suppresses at least as many gaps.
    for a, b in zip(points, points[1:]):
        assert b.frac_suppressed >= a.frac_suppressed
        assert b.lo_given_back >= a.lo_given_back
    # An effectively-infinite hold_off suppresses every gap and gives back
    # every job admitted in a gap, by construction.
    assert points[-1].frac_suppressed == 1.0
    assert points[-1].lo_given_back == points[-1].lo_admitted_in_any_gap


def test_hold_off_sweep_rejects_unknown_regime():
    with pytest.raises(ValueError, match="regime"):
        hold_off_sweep(hold_offs=[10], U=0.7, regime="bogus", n_tasksets=2)


def test_hysteresis_sweep_produces_cells_and_hold_off_zero_matches_amc_rh():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = hysteresis_sweep(
            hold_offs=[0, 100], U_values=(0.7,), regimes=("tight",),
            n_tasksets=6, seeds=(0, 1), duration=20_000,
        )
        early = early_exit_trial(
            U_values=(0.7,), n_tasksets=6, seeds=(0, 1), duration=20_000,
        )
    assert len(cells) == 2  # one per hold_off, single (U, regime)
    for c in cells:
        assert isinstance(c, HysteresisCell)
        assert c.n_tasksets >= 2
        assert c.summary()

    # H=0 is exactly exit_policy="amc_rh" -- cross-check against early_exit_trial's
    # own (U=0.7, tight, B) cell, same population/seed/duration parameters.
    h0 = next(c for c in cells if c.hold_off == 0)
    amc_rh_cell = next(
        c for c in early if c.U == 0.7 and c.regime == "tight" and c.operating_point == "B"
    )
    assert h0.service_ratio.mean_diff == pytest.approx(amc_rh_cell.service_ratio.mean_diff)
    assert h0.level_trans.mean_diff == pytest.approx(amc_rh_cell.level_trans.mean_diff)
