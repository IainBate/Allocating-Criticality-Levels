"""Tests for the revised protocol's reusable stages (multilevel_protocol.py).

Small, fast checks that the ladder builders, feasibility counting and the
regime map wiring behave as documented -- not a re-verification of the
underlying analysis, which is tested in tests/scheduling/test_drop_sets.py
and tests/simulation/test_multilevel.py.
"""

from __future__ import annotations

import warnings

import pytest

from amc_tasksim.experiments.multilevel_protocol import (
    BoundedGainResult,
    FeasibilityResult,
    bounded_gain,
    build_tight_population,
    feasibility_fraction,
    progressive_ladder,
    regime_map,
    shed_early_ladder,
)
from amc_tasksim.experiments.sweep import build_population
from amc_tasksim.scheduling.amc_rtb import amc_rtb


@pytest.fixture(scope="module")
def tasksets():
    ts, _ = build_population(12, U=0.7, seed=0, n=10, CP=0.5, CF=2.0)
    return ts


def test_shed_early_ladder_is_a_single_uniform_level(tasksets):
    for ts in tasksets:
        lad = shed_early_ladder(ts)
        if lad is None:
            continue
        assert lad.k == 2
        assert lad.severities == [0.0]


def test_progressive_ladder_can_shed_less_at_the_shallowest_rung(tasksets):
    """The structural pilot's headline finding: progressive != shed-early when it exists."""
    saw_strictly_less = False
    for ts in tasksets:
        se = shed_early_ladder(ts)
        pg = progressive_ladder(ts, [0.0, 0.25])
        if se is None or pg is None:
            continue
        assert pg.drop_sets[0] <= se.drop_sets[0]
        if len(pg.drop_sets[0]) < len(se.drop_sets[0]):
            saw_strictly_less = True
    assert saw_strictly_less, "no task set showed grading; the pilot's finding did not reproduce"


def test_conservative_never_sheds_less_than_termination(tasksets):
    for ts in tasksets:
        a = shed_early_ladder(ts, require_lo_deadlines=True)
        b = shed_early_ladder(ts, require_lo_deadlines=False)
        assert b is not None, "the termination point must always be feasible"
        if a is not None:
            assert b.drop_sets[0] <= a.drop_sets[0]


def test_feasibility_fraction_matches_direct_construction(tasksets):
    result = feasibility_fraction(tasksets, require_lo_deadlines=False, severities=(0.0, 0.25))
    assert isinstance(result, FeasibilityResult)
    assert result.n == len(tasksets)
    direct = sum(1 for ts in tasksets if shed_early_ladder(ts) is not None)
    assert result.shed_early == pytest.approx(direct / len(tasksets))
    assert 0.0 <= result.shed_early <= 1.0
    assert result.progressive is not None
    assert 0.0 <= result.progressive <= 1.0


def test_feasibility_progressive_is_none_when_severities_omitted(tasksets):
    result = feasibility_fraction(tasksets, require_lo_deadlines=False)
    assert result.progressive is None
    assert result.mean_shed_pct_progressive_l1 is None


def test_build_tight_population_preserves_normal_mode_schedulability():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tasksets = build_tight_population(6, U=0.7, seed=0, alpha=0.3, n=10, CP=0.5, CF=2.0)
    assert len(tasksets) == 6
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        for i in range(ts.n):
            assert r_lo[i] <= ts.D[i] <= ts.T[i]


def test_regime_map_produces_paired_cells_for_both_operating_points():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = regime_map(
            U_values=(0.7,), n_tasksets=6, seeds=(0, 1), duration=20_000,
        )
    assert cells, "no cells produced"
    ops = {c.operating_point for c in cells}
    assert ops == {"A", "B"}
    policies = {c.policy for c in cells}
    assert "shed_early" in policies
    for cell in cells:
        assert cell.n_tasksets > 0
        assert cell.vs_amc_ra.n == cell.n_tasksets


def test_bounded_gain_default_never_beats_the_optimum():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bounded_gain(
            require_lo_deadlines=False, n_tasksets=3, n_tasks=6, U=0.7,
            seeds=(0, 1), duration=20_000,
        )
    if result is None:
        pytest.skip("no feasible task sets at this tiny population")
    assert isinstance(result, BoundedGainResult)
    assert result.default_vs_best.mean_diff >= -1e-9, (
        "the exhaustive optimum must be at least as good as the default"
    )
    assert result.mean_best_shed_size <= result.mean_default_shed_size + 1e-9
