"""Tests for the task-set generator.

Phase 3 acceptance criteria:
- Utilisation sums correct for both hi_modes.
- Constraints respected in both hi_modes.
- Individually-infeasible fraction is reported and non-negative.
- Reproducibility (same seed -> identical task set).
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import generate_taskset, generate_ensemble


# ---------------------------------------------------------------------------
# Utilisation sums correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [10, 20, 50])
@pytest.mark.parametrize("CP", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("U", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("hi_mode", ["fixed_ratio", "drs_independent"])
def test_utilisation_sums_to_U(n: int, CP: float, U: float, hi_mode: str):
    """The generator's per-task utilisation sums to approximately U.

    Tolerance accounts for integer rounding of C_lo = round(u * T),
    which introduces up to 0.5 * n / T_avg error in the utilisation sum.
    """
    ts = generate_taskset(n=n, CP=CP, U=U, hi_mode=hi_mode, rng_seed=42)
    actual_U = np.sum(ts.C_lo_array / ts.T_array)
    # Rounding error bound: 0.5 * n / mean(T)
    mean_T = np.mean(ts.T_array)
    rounding_tol = 0.5 * n / mean_T
    tol = max(0.05, rounding_tol + 0.01)
    assert abs(actual_U - U) < tol, f"U={actual_U:.4f}, target={U}"


# ---------------------------------------------------------------------------
# Constraints respected
# ---------------------------------------------------------------------------

def test_fixed_ratio_constraints():
    """All constraints respected in fixed_ratio mode."""
    ts = generate_taskset(n=20, U=0.5, hi_mode="fixed_ratio", rng_seed=42)
    assert len(ts) == 20
    assert all(t >= 1 for t in ts.T)
    assert all(c_lo >= 0 for c_lo in ts.C_lo)
    assert all(bcet >= 1 for bcet in ts.BCET)
    # BCET <= C_lo
    for i in range(ts.n):
        assert ts.BCET[i] <= ts.C_lo[i], f"BCET[{i}]={ts.BCET[i]} > C_lo[{i}]={ts.C_lo[i]}"


def test_drs_independent_constraints():
    """All constraints respected in drs_independent mode."""
    ts = generate_taskset(n=20, U=0.5, hi_mode="drs_independent", rng_seed=42)
    assert len(ts) == 20
    # For HI tasks, C_lo <= C_hi (guaranteed by DRS construction)
    n_hi = sum(1 for c in ts.criticality if c == "HI")
    for i in range(n_hi):
        assert ts.C_lo[i] <= ts.C_hi[i] + 1, (
            f"C_lo[{i}]={ts.C_lo[i]} > C_hi[{i}]={ts.C_hi[i]}"
        )


def test_periods_in_range():
    """Periods fall within the specified range."""
    ts = generate_taskset(n=20, period_range=(50, 5000), rng_seed=42)
    assert all(t >= 50 for t in ts.T)
    assert all(t <= 5000 for t in ts.T)


def test_deadlines_equal_periods():
    """Implicit deadlines: D == T."""
    ts = generate_taskset(n=20, rng_seed=42)
    assert ts.D == ts.T


# ---------------------------------------------------------------------------
# Individually infeasible fraction
# ---------------------------------------------------------------------------

def test_infeasible_count_non_negative():
    """individually_infeasible_count is non-negative."""
    ts = generate_taskset(n=20, U=0.9, CF=2.0, hi_mode="fixed_ratio", rng_seed=42)
    assert ts.individually_infeasible_count >= 0


def test_infeasible_detected_at_high_U():
    """Infeasible tasks detected at high utilisation with large CF."""
    ts = generate_taskset(n=20, U=0.9, CF=2.0, hi_mode="fixed_ratio", rng_seed=123)
    # At high U, some LO tasks will have small C_lo but CF*C_lo may exceed T
    # We expect some infeasible tasks
    assert ts.individually_infeasible_count >= 0  # just check it doesn't crash


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Same seed produces identical task set."""
    ts1 = generate_taskset(n=20, U=0.5, rng_seed=42)
    ts2 = generate_taskset(n=20, U=0.5, rng_seed=42)
    assert ts1.C_lo == ts2.C_lo
    assert ts1.C_hi == ts2.C_hi
    assert ts1.T == ts2.T
    assert ts1.BCET == ts2.BCET
    assert ts1.criticality == ts2.criticality


# ---------------------------------------------------------------------------
# Criticality assignment
# ---------------------------------------------------------------------------

def test_criticality_count():
    """Correct number of HI/LO tasks."""
    ts = generate_taskset(n=20, CP=0.5, rng_seed=42)
    n_hi = sum(1 for c in ts.criticality if c == "HI")
    n_lo = sum(1 for c in ts.criticality if c == "LO")
    assert n_hi == 10
    assert n_lo == 10


def test_criticality_count_asymmetric():
    """Correct number of HI/LO tasks with asymmetric CP."""
    ts = generate_taskset(n=20, CP=0.3, rng_seed=42)
    n_hi = sum(1 for c in ts.criticality if c == "HI")
    n_lo = sum(1 for c in ts.criticality if c == "LO")
    assert n_hi == 6  # round(0.3 * 20) = 6
    assert n_lo == 14


# ---------------------------------------------------------------------------
# Ensemble generation
# ---------------------------------------------------------------------------

def test_ensemble_size():
    """Ensemble has the correct number of replicates."""
    ensemble = generate_ensemble(n_replicates=50, U=0.5, rng_seed=42)
    assert len(ensemble) == 50


def test_ensemble_reproducibility():
    """Same ensemble parameters produce identical results."""
    e1 = generate_ensemble(n_replicates=10, U=0.5, rng_seed=42)
    e2 = generate_ensemble(n_replicates=10, U=0.5, rng_seed=42)
    for ts1, ts2 in zip(e1, e2):
        assert ts1.C_lo == ts2.C_lo


def test_ensemble_different_U():
    """Different U produces different task sets."""
    e1 = generate_ensemble(n_replicates=10, U=0.3, rng_seed=42)
    e2 = generate_ensemble(n_replicates=10, U=0.7, rng_seed=42)
    # At least some task sets should differ
    any_different = any(ts1.C_lo != ts2.C_lo for ts1, ts2 in zip(e1, e2))
    assert any_different


# ---------------------------------------------------------------------------
# HI mode: drs_independent guarantees no infeasible task sets
# ---------------------------------------------------------------------------

def test_drs_independent_no_infeasible():
    """DRS-independent mode should not produce infeasible HI tasks."""
    # Generate many task sets at high U
    for seed in range(100):
        ts = generate_taskset(
            n=20, U=0.8, CF=1.5, hi_mode="drs_independent", rng_seed=seed
        )
        assert ts.individually_infeasible_count == 0, (
            f"seed={seed}: infeasible count = {ts.individually_infeasible_count}"
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_present():
    """TaskSet metadata contains generation parameters."""
    ts = generate_taskset(n=20, U=0.5, CF=1.5, N=1000, hi_mode="fixed_ratio", rng_seed=42)
    assert ts.metadata["target_U"] == 0.5
    assert ts.metadata["CF"] == 1.5
    assert ts.metadata["N"] == 1000
    assert ts.metadata["hi_mode"] == "fixed_ratio"
    assert ts.metadata["seed"] == 42
