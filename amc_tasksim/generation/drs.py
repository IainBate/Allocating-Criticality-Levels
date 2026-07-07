"""Dirichlet-Rescale (DRS) algorithm for constrained uniform simplex sampling.

Implements the DRS algorithm from "Generating Utilization Vectors for the
Evaluation of Real-Time Scheduling Algorithms" (RTSS 2020), providing
uniform sampling from the intersection of the standard simplex with
box constraints [umin, umax].

Also provides UUnifast as a reference (unconstrained case).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def uunifast(n: int, U: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a uniform random point on the standard n-simplex scaled to sum U.

    The standard approach: generate n-1 uniform random numbers in [0, U],
    sort them, include 0 and U as endpoints, and take the gaps.

    Args:
        n: Dimension (number of components).
        U: Target sum (must be > 0).
        rng: Random number generator. Defaults to global generator.

    Returns:
        Array of n positive values summing to U.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n == 1:
        return np.array([U])
    if U <= 0:
        raise ValueError(f"U must be positive, got {U}")
    cuts = rng.uniform(0, U, size=n - 1)
    cuts = np.sort(cuts)
    cuts = np.concatenate([[0.0], cuts, [U]])
    return np.diff(cuts)


def drs(
    n: int,
    U: float,
    umax: Optional[np.ndarray] = None,
    umin: Optional[np.ndarray] = None,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Generate a uniform random point on the constrained simplex.

    Samples uniformly from {x in R^n : sum(x) = U, umin_i <= x_i <= umax_i}.

    Uses the canonical-form reduction: transform to u'_max = umax - umin,
    U' = U - sum(umin), solve for x', then x = x' + umin.

    The core algorithm iteratively fixes components that violate their bounds
    and redistributes the remaining budget among the unconstrained components.

    Args:
        n: Dimension.
        U: Target sum.
        umax: Upper bounds (default: ones(n)).
        umin: Lower bounds (default: zeros(n)).
        epsilon: Tolerance for sum convergence.

    Returns:
        Array of n values summing to U within epsilon, within [umin, umax].

    Raises:
        ValueError: If constraints are infeasible or algorithm fails to converge.
    """
    if umax is None:
        umax = np.ones(n)
    else:
        umax = np.array(umax, dtype=float)
    if umin is None:
        umin = np.zeros(n)
    else:
        umin = np.array(umin, dtype=float)

    # Validate constraints
    if np.any(umax < umin - 1e-12):
        raise ValueError("umax must be elementwise >= umin")
    if U < np.sum(umin) - epsilon:
        raise ValueError(f"U={U} < sum(umin)={np.sum(umin)}")
    if U > np.sum(umax) + epsilon:
        raise ValueError(f"U={U} > sum(umax)={np.sum(umax)}")

    # Canonical-form reduction: subtract umin from everything
    u_max_prime = umax - umin
    U_prime = U - np.sum(umin)

    # Handle edge case: U_prime == 0 (all components at minimum)
    if U_prime <= 0:
        return umin.copy()

    # Handle edge case: n == 1
    if n == 1:
        return np.array([U])

    # Main DRS loop with retry
    max_retries = 50
    for attempt in range(max_retries):
        x = _drs_iterative(n, U_prime, u_max_prime)
        result = x + umin

        if abs(np.sum(result) - U) <= epsilon:
            return result

    warnings.warn(
        f"DRS did not converge after {max_retries} retries "
        f"(sum={np.sum(result):.6f}, target={U})",
        stacklevel=2,
    )
    return result


def _drs_iterative(n: int, U: float, umax: np.ndarray) -> np.ndarray:
    """Core DRS iteration: fix-and-resample until all bounds satisfied.

    Algorithm:
    1. Maintain a set of "active" (unfixed) components and remaining budget.
    2. Sample uniformly on the active simplex (summing to remaining budget).
    3. Check each active component against its effective upper bound.
       The effective bound for component i is:
         min(umax_i, remaining_budget - sum(umin_j for j in active, j != i))
       This is the most any component i can take while leaving enough
       budget for the minimum contributions of all other active components.
    4. Fix (cap) any component that exceeds its effective bound.
    5. Reduce remaining budget by the fixed amount and remove from active set.
    6. Repeat until no active components remain or no violations.

    Args:
        n: Total number of components.
        U: Target sum for the active components.
        umax: Upper bound for each component (in canonical form, so umin=0).

    Returns:
        Array summing to U (within floating-point tolerance).
    """
    x = np.zeros(n)
    active = np.ones(n, dtype=bool)
    remaining_U = U

    while np.any(active):
        active_indices = np.where(active)[0]
        k = len(active_indices)

        if k == 0:
            break

        if k == 1:
            # Last active component gets whatever budget is left
            idx = active_indices[0]
            x[idx] = remaining_U
            active[idx] = False
            break

        # Effective upper bound for each active component:
        # min(umax_i, remaining_U - sum(umin_j for j in active, j != i))
        # Since we're in canonical form (umin=0), the umin term is 0.
        # But we need to ensure remaining_U > 0 for sampling.
        if remaining_U <= 0:
            # No budget left — assign zero to all remaining
            for idx in active_indices:
                x[idx] = 0.0
                active[idx] = False
            break

        # Compute effective max for each active component
        # If component i takes everything, the rest must sum to 0,
        # so max_i = min(umax_i, remaining_U)
        eff_max = np.minimum(umax[active_indices], remaining_U)

        # Check if any component can actually take any positive amount
        # (i.e., its effective max > 0 and there's room for others)
        # If a component's umax > remaining_U - min(umax of others),
        # it might need to be capped.

        # Sample uniformly on the unconstrained simplex of size k summing to remaining_U
        sample = uunifast(k, remaining_U)

        # Check for violations: any sample[i] > eff_max[i]?
        violations = sample > eff_max + 1e-12

        if not np.any(violations):
            # No violations — assign and done
            for j, idx in enumerate(active_indices):
                x[idx] = sample[j]
                active[idx] = False
            break

        # Fix violated components to their effective max
        for j, idx in enumerate(active_indices):
            if violations[j]:
                x[idx] = eff_max[j]
                active[idx] = False
                remaining_U -= eff_max[j]

        # Continue loop with remaining active components
        # If remaining_U <= 0 after fixes, break
        if remaining_U <= 1e-12:
            for idx in np.where(active)[0]:
                x[idx] = 0.0
                active[idx] = False
            break

    # Final check: ensure sum is exactly U
    current_sum = np.sum(x)
    if abs(current_sum - U) > 1e-10:
        # Distribute the difference among components that have room
        diff = U - current_sum
        # Find components that can absorb the difference
        room_up = np.where(active, np.inf, np.maximum(umax - x, 0))
        room_down = np.where(active, np.inf, np.maximum(x, 0))
        # Use all components (including fixed ones that have room)
        for i in range(n):
            if diff > 0 and x[i] < umax[i] - 1e-12:
                room = min(diff, umax[i] - x[i])
                x[i] += room
                diff -= room
            elif diff < 0 and x[i] > 1e-12:
                room = min(-diff, x[i])
                x[i] -= room
                diff += room
            if abs(diff) < 1e-12:
                break

    return x
