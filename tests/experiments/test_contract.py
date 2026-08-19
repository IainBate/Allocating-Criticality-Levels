"""Tests for the measurement contract.

The contract's job is to make an unsound comparison impossible to express, so
most of these check that it *refuses* things rather than that it computes them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from amc_tasksim.experiments.contract import (
    CANONICAL,
    CONFIRM_SEED_BLOCK,
    EFFECT_FLOOR,
    SEED_BLOCK,
    aggregate_by_taskset,
    paired_compare,
    required_pairs,
    select,
)


# ---------------------------------------------------------------------------
# aggregate_by_taskset -- getting the pairing unit right
# ---------------------------------------------------------------------------


def test_aggregate_collapses_seeds_to_one_value_per_taskset():
    # Two task sets, three seeds each, laid out task-set-major.
    values = [1.0, 2.0, 3.0,   10.0, 20.0, 30.0]
    assert aggregate_by_taskset(values, 3) == [2.0, 20.0]


def test_aggregate_rejects_a_ragged_layout():
    """Silently mixing task sets together would corrupt the pairing."""
    with pytest.raises(ValueError, match="whole number of task sets"):
        aggregate_by_taskset([1.0, 2.0, 3.0, 4.0, 5.0], 2)


def test_aggregate_rejects_nonpositive_seed_counts():
    with pytest.raises(ValueError, match="must be positive"):
        aggregate_by_taskset([1.0, 2.0], 0)


def test_aggregating_before_pairing_beats_pairing_per_run():
    """The unit of pairing is the task set, and it matters.

    Two configurations consume the random stream differently, so their runs at
    a given seed decorrelate; task-set means stay correlated because the task
    set is shared. Averaging first therefore cancels more variance than
    differencing per run does.
    """
    rng = np.random.default_rng(23)
    n_sets, n_seeds = 20, 20
    per_run_base, per_run_cand = [], []
    for _ in range(n_sets):
        taskset_effect = rng.normal(100, 40)      # shared, dominant
        for _ in range(n_seeds):
            per_run_base.append(taskset_effect + rng.normal(0, 15))
            per_run_cand.append(taskset_effect + rng.normal(0, 15) - 5)

    per_run = paired_compare(per_run_base, per_run_cand)
    by_set = paired_compare(
        aggregate_by_taskset(per_run_base, n_seeds),
        aggregate_by_taskset(per_run_cand, n_seeds),
    )
    assert by_set.std_err < per_run.std_err


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_confirmation_seeds_are_disjoint_from_selection_seeds():
    """Selection and confirmation must use independent randomness.

    Overlapping blocks would re-use the draws that produced the winner to
    confirm it, which is the winner's curse the confirmation step exists to
    remove.
    """
    assert not set(SEED_BLOCK) & set(CONFIRM_SEED_BLOCK)


def test_severity_grid_excludes_the_pinned_level_and_reaches_one():
    grid = CANONICAL.severity_grid()
    assert 0.0 not in grid, "x=0 is pinned as level 1, not a free variable"
    assert grid[-1] == pytest.approx(1.0)
    assert grid == sorted(grid)
    assert len(grid) == 20


def test_k_levels_stop_where_the_budget_stops_resolving():
    """k=5 is excluded deliberately; the exclusion should be visible."""
    assert 5 not in CANONICAL.k_levels
    assert CANONICAL.k_levels == (2, 3, 4)


def test_grid_is_immutable():
    with pytest.raises(Exception):
        CANONICAL.duration = 1


# ---------------------------------------------------------------------------
# paired_compare
# ---------------------------------------------------------------------------


def test_unequal_samples_are_refused_not_silently_unpaired():
    """The one error that must never degrade gracefully."""
    with pytest.raises(ValueError, match="same task sets and the same seeds"):
        paired_compare([1.0, 2.0, 3.0], [1.0, 2.0])


def test_too_few_pairs_refused():
    with pytest.raises(ValueError, match="at least 2 pairs"):
        paired_compare([1.0], [2.0])


def test_identical_samples_show_no_difference():
    values = [3.0, 5.0, 11.0, 7.0, 2.0]
    r = paired_compare(values, values)
    assert r.mean_diff == 0.0
    assert r.std_err == 0.0
    assert not r.significant


def test_constant_shift_is_detected_exactly():
    """A shift with no added noise has zero variance, so it is always significant."""
    base = [10.0, 20.0, 30.0, 40.0]
    r = paired_compare(base, [b - 2.0 for b in base])
    assert r.mean_diff == pytest.approx(-2.0)
    assert r.std_err == pytest.approx(0.0)
    assert r.significant
    assert r.relative == pytest.approx(-2.0 / 25.0)


def test_pairing_beats_unpairing_on_correlated_samples():
    """The property the whole contract exists for.

    Task-set-to-task-set variance dwarfs the effect; pairing cancels it. This
    reproduces in miniature the 79x variance reduction measured on the engine.
    """
    rng = np.random.default_rng(0)
    n = 60
    taskset_effect = rng.normal(100, 40, n)      # large shared component
    base = taskset_effect + rng.normal(0, 3, n)
    cand = taskset_effect + rng.normal(0, 3, n) - 5.0   # true effect: -5

    paired = paired_compare(base, cand)

    # Unpaired standard error on the same numbers, for comparison.
    unpaired_se = math.sqrt(
        base.var(ddof=1) / n + cand.var(ddof=1) / n
    )

    assert paired.std_err < unpaired_se / 5, (
        f"pairing should shrink the standard error substantially: "
        f"paired={paired.std_err:.3f} unpaired={unpaired_se:.3f}"
    )
    assert paired.significant
    assert not paired.underpowered
    assert paired.mean_diff == pytest.approx(-5.0, abs=1.5)


def test_noise_alone_is_not_reported_as_a_difference():
    rng = np.random.default_rng(7)
    n = 80
    shared = rng.normal(50, 20, n)
    r = paired_compare(shared + rng.normal(0, 2, n), shared + rng.normal(0, 2, n))
    assert not r.significant


def test_underpowered_samples_are_flagged():
    """A comparison that cannot resolve the effect floor must say so.

    Without this the caller cannot distinguish "no effect" from "no power",
    which are opposite conclusions.
    """
    rng = np.random.default_rng(3)
    noisy = paired_compare(
        list(rng.normal(100, 50, 4)), list(rng.normal(100, 50, 4))
    )
    assert noisy.underpowered
    assert noisy.resolvable > EFFECT_FLOOR
    assert "UNDERPOWERED" in noisy.summary()


def test_small_but_precise_difference_is_not_practically_significant():
    """Statistical significance alone is not a result."""
    base = [100.0] * 40
    cand = [99.5] * 40          # a real 0.5% improvement, measured exactly
    r = paired_compare(base, cand)
    assert r.significant
    assert abs(r.relative) < EFFECT_FLOOR
    assert not r.practically_significant


def test_zero_baseline_does_not_divide_by_zero():
    r = paired_compare([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert math.isnan(r.relative)
    assert math.isinf(r.resolvable)
    assert r.underpowered


def test_ci_brackets_the_mean_difference():
    rng = np.random.default_rng(11)
    n = 50
    shared = rng.normal(30, 10, n)
    r = paired_compare(shared, shared - 3 + rng.normal(0, 1, n))
    lo, hi = r.ci
    assert lo < r.mean_diff < hi
    assert lo < -3 < hi


# ---------------------------------------------------------------------------
# required_pairs
# ---------------------------------------------------------------------------


def test_required_pairs_grows_as_the_target_effect_shrinks():
    rng = np.random.default_rng(5)
    n = 30
    shared = rng.normal(100, 30, n)
    base = shared + rng.normal(0, 8, n)
    cand = shared + rng.normal(0, 8, n) - 5

    at_5 = required_pairs(base, cand, effect=0.05)
    at_10 = required_pairs(base, cand, effect=0.10)
    at_20 = required_pairs(base, cand, effect=0.20)

    assert at_5 > at_10 > at_20
    # Quartering per doubling of the effect, since n scales with 1/effect^2.
    assert at_10 == pytest.approx(at_20 * 4, rel=0.2)


def test_required_pairs_grows_with_noise():
    rng = np.random.default_rng(9)
    n = 40
    shared = rng.normal(100, 5, n)
    quiet = required_pairs(shared, shared + rng.normal(0, 1, n))
    loud = required_pairs(shared, shared + rng.normal(0, 20, n))
    assert loud > quiet


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_reports_ties_rather_than_a_false_winner():
    scores = {"a": 10.0, "b": 10.2, "c": 15.0}
    errs = {"a": 0.5, "b": 0.5, "c": 0.5}
    s = select(scores, errs)
    assert s.best == "a"
    assert set(s.indifference_set) == {"a", "b"}
    assert not s.is_unique


def test_select_identifies_a_clear_winner_as_unique():
    scores = {"a": 10.0, "b": 20.0, "c": 30.0}
    errs = {"a": 0.1, "b": 0.1, "c": 0.1}
    s = select(scores, errs)
    assert s.best == "a"
    assert s.is_unique


def test_select_refuses_an_empty_field():
    with pytest.raises(ValueError, match="no configurations"):
        select({}, {})
