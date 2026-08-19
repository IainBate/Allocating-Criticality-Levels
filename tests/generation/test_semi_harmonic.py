"""Tests for semi-harmonic period generation (AMC-RH paper's stress regime).

The paper's own results (Table I) show semi-harmonic periods give
substantially worse JNE+LDM than log-uniform ("non-harmonic") periods at the
same utilisation -- these tests pin the generation mechanics; the empirical
JNE comparison lives in the settings-selection pilot, not here, since it
needs the simulator rather than just the generator.
"""

from __future__ import annotations

import pytest

from amc_tasksim.generation.taskset import (
    _SEMI_HARMONIC_PERIODS_TICKS,
    generate_taskset,
)


def test_semi_harmonic_periods_are_drawn_only_from_the_paper_families():
    ts = generate_taskset(n=200, period_mode="semi_harmonic", rng_seed=0)
    assert set(ts.T) <= set(_SEMI_HARMONIC_PERIODS_TICKS)


def test_semi_harmonic_uses_both_base_frequency_families():
    """Neither family alone should be enough to explain a large sample."""
    ts = generate_taskset(n=500, period_mode="semi_harmonic", rng_seed=1)
    family_a = {250, 500, 1000, 2500, 5000, 10000}  # 25,50,100,250,500,1000ms
    family_b = {200, 400, 800, 2000, 4000, 8000}  # 20,40,80,200,400,800ms
    observed = set(ts.T)
    assert observed & family_a, "family A never sampled"
    assert observed & family_b, "family B never sampled"


def test_semi_harmonic_values_match_the_paper_in_ticks():
    """0.1ms/tick, matching period_range=(100,10000) == 10ms-1s in log_uniform."""
    expected_ms = {25, 50, 100, 250, 500, 1000, 20, 40, 80, 200, 400, 800}
    assert set(_SEMI_HARMONIC_PERIODS_TICKS) == {ms * 10 for ms in expected_ms}


def test_deadlines_equal_periods_in_semi_harmonic_mode_too():
    ts = generate_taskset(n=30, period_mode="semi_harmonic", rng_seed=2)
    assert ts.D == ts.T


def test_log_uniform_is_still_the_default():
    ts = generate_taskset(n=30, rng_seed=3)
    assert ts.metadata["period_mode"] == "log_uniform"
    # Log-uniform periods are essentially never exactly one of the twelve
    # harmonic values; a large majority falling outside them confirms the
    # default path, not the new one, produced this task set.
    off_harmonic = sum(1 for t in ts.T if t not in _SEMI_HARMONIC_PERIODS_TICKS)
    assert off_harmonic >= len(ts.T) - 1


def test_unknown_period_mode_rejected():
    with pytest.raises(ValueError, match="period_mode"):
        generate_taskset(n=5, period_mode="weekly", rng_seed=0)  # type: ignore[arg-type]


def test_semi_harmonic_reproducible_with_same_seed():
    a = generate_taskset(n=40, period_mode="semi_harmonic", rng_seed=42)
    b = generate_taskset(n=40, period_mode="semi_harmonic", rng_seed=42)
    assert a.T == b.T
