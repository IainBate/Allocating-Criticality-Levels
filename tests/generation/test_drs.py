"""Tests for the DRS (Dirichlet-Rescale) algorithm.

Phase 2 acceptance criteria:
1. Output always sums to U within epsilon.
2. Every box constraint is respected.
3. DRS == UUnifast for canonical case (umax=ones, umin=zeros), verified by KS test.
4. Performance sanity check for n=50.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

from amc_tasksim.generation.drs import drs, uunifast


# ---------------------------------------------------------------------------
# Test 1: Output sums to U within epsilon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 5, 10, 20, 50])
@pytest.mark.parametrize("U", [0.01, 0.1, 0.5, 1.0])
def test_drs_sums_to_U(n: int, U: float):
    """Output always sums to U within epsilon for unconstrained case."""
    result = drs(n, U)
    assert abs(np.sum(result) - U) < 1e-4, f"sum={np.sum(result)}, target={U}"


def test_drs_sums_to_U_large_with_wide_bounds():
    """U=5.0 with n=2 works when umax allows it."""
    result = drs(2, 5.0, umax=np.array([3.0, 3.0]))
    assert abs(np.sum(result) - 5.0) < 1e-4
    assert result[0] <= 3.0 + 1e-10
    assert result[1] <= 3.0 + 1e-10


@pytest.mark.parametrize(
    "umax,umin",
    [
        (np.ones(10) * 0.3, np.zeros(10)),  # symmetric upper
        (np.ones(10) * 0.1, np.ones(10) * 0.01),  # tight symmetric
        (np.concatenate([np.ones(5) * 0.3, np.ones(5) * 0.1]), np.zeros(10)),  # asymmetric
        (np.arange(1, 11) * 0.05, np.arange(0, 10) * 0.02),  # linearly varying
    ],
)
def test_drs_sums_to_U_constrained(umax: np.ndarray, umin: np.ndarray):
    """Output sums to U within epsilon for constrained cases."""
    U = np.sum(umin) + 0.5 * (np.sum(umax) - np.sum(umin))
    result = drs(len(umax), U, umax=umax, umin=umin)
    assert abs(np.sum(result) - U) < 1e-4


# ---------------------------------------------------------------------------
# Test 2: Box constraints respected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "umax,umin",
    [
        (np.ones(10), np.zeros(10)),
        (np.ones(10) * 0.3, np.zeros(10)),
        (np.ones(10) * 0.1, np.ones(10) * 0.01),
        (np.concatenate([np.ones(5) * 0.3, np.ones(5) * 0.1]), np.zeros(10)),
        (np.arange(1, 11) * 0.05, np.arange(0, 10) * 0.02),
    ],
)
def test_drs_constraints_respected(umax: np.ndarray, umin: np.ndarray):
    """Every output component satisfies umin_i <= x_i <= umax_i."""
    U = np.sum(umin) + 0.5 * (np.sum(umax) - np.sum(umin))
    for _ in range(100):
        result = drs(len(umax), U, umax=umax, umin=umin)
        assert np.all(result >= umin - 1e-10), f"Below umin: {result[result < umin]}"
        assert np.all(result <= umax + 1e-10), f"Above umax: {result[result > umax]}"


def test_drs_edge_min():
    """Output at lower bound when U = sum(umin)."""
    umin = np.array([0.1, 0.2, 0.3])
    umax = np.ones(3)
    result = drs(3, 0.6, umax=umax, umin=umin)
    np.testing.assert_array_almost_equal(result, umin, decimal=8)


def test_drs_edge_max():
    """Output at upper bound when U = sum(umax)."""
    umin = np.zeros(3)
    umax = np.array([0.3, 0.4, 0.5])
    result = drs(3, 1.2, umax=umax, umin=umin)
    np.testing.assert_array_almost_equal(result, umax, decimal=8)


# ---------------------------------------------------------------------------
# Test 3: DRS == UUnifast for canonical case (KS test)
# ---------------------------------------------------------------------------

def test_drs_equivalence_to_uunifast():
    """DRS with umax=ones, umin=zeros is statistically indistinguishable from UUnifast."""
    n = 10
    U = 1.0
    n_samples = 10000

    # Generate samples from both methods
    drs_samples = np.column_stack([drs(n, U) for _ in range(n_samples)])
    uunifast_samples = np.column_stack([uunifast(n, U) for _ in range(n_samples)])

    # Project onto a fixed direction (e.g., first component)
    # Both should give Beta(1, n-1) * U for the first component
    drs_proj = drs_samples[0]
    uunifast_proj = uunifast_samples[0]

    # Two-sample KS test
    stat, pvalue = ks_2samp(drs_proj, uunifast_proj)
    assert pvalue > 0.05, (
        f"DRS and UUnifast distributions differ: KS stat={stat:.4f}, p={pvalue:.4f}"
    )

    # Also check second component
    stat2, pvalue2 = ks_2samp(drs_samples[1], uunifast_samples[1])
    assert pvalue2 > 0.05


def test_uunifast_basic():
    """UUnifast produces valid outputs."""
    result = uunifast(5, 1.0)
    assert len(result) == 5
    assert abs(np.sum(result) - 1.0) < 1e-10
    assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# Test 4: Performance sanity check
# ---------------------------------------------------------------------------

def test_drs_performance_n50():
    """For n=50, U=0.5, umax=UUnifast(50,1), complete quickly."""
    import time

    n = 50
    U = 0.5
    umax = uunifast(n, 1.0)
    umin = np.zeros(n)

    start = time.perf_counter()
    for _ in range(100):
        drs(n, U, umax=umax, umin=umin)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0, f"DRS took {elapsed:.1f}s for 100 calls (n=50); expected < 30s"


# ---------------------------------------------------------------------------
# Validation: ValueError on infeasible constraints
# ---------------------------------------------------------------------------

def test_drs_invalid_umax_umin():
    """Raise ValueError when umax < umin elementwise."""
    with pytest.raises(ValueError):
        drs(3, 1.0, umax=np.array([0.1, 0.2, 0.3]), umin=np.array([0.4, 0.5, 0.6]))


def test_drs_invalid_U_too_small():
    """Raise ValueError when U < sum(umin)."""
    with pytest.raises(ValueError):
        drs(3, 0.1, umax=np.ones(3), umin=np.array([0.1, 0.2, 0.3]))


def test_drs_invalid_U_too_large():
    """Raise ValueError when U > sum(umax)."""
    with pytest.raises(ValueError):
        drs(3, 2.0, umax=np.array([0.3, 0.4, 0.5]), umin=np.zeros(3))
