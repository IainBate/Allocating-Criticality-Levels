"""Dirichlet-Rescale (DRS) algorithm for constrained uniform simplex sampling.

Implements the DRS algorithm of "Generating Utilization Vectors for the
Evaluation of Real-Time Scheduling Algorithms" (Griffin, Bate and Davis,
RTSS 2020), Section III-B. DRS draws points uniformly from

    {x in R^n : sum(x) = U, umin_i <= x_i <= umax_i}

by generating a point in the standard simplex and then applying a sequence of
affine transformations that fold it into the valid region. Because the
transformations are affine, uniformity is preserved -- unlike clamping a
violating component to its bound, which puts a point mass on the boundary.

The sub-functions below follow the paper's decomposition:

===========  ==============================================================
RMSS(S)      the transformation mapping a regular simplex S onto the
             standard simplex
CtS(r)       the constraints simplex for a constraint vector r
Rescale(r,P) folds P into the valid region using the broken constraints
SSR(r,P)     transposes the problem when the constraints simplex is the
             smaller of the two, then calls Rescale
===========  ==============================================================

UUnifast and UUnifast-Discard are also provided: the former as the
unconstrained reference of Section II-C, the latter as a slow but provably
uniform ground truth for the constrained case, used in the tests.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# Points are compared against constraints with this slack. The transformations
# are exact in principle but accumulate rounding error over successive folds.
_TOL = 1e-12


def uunifast(n: int, U: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a uniform random point on the simplex {x >= 0, sum(x) = U}.

    Section II-C of the DRS paper: draw n-1 uniform values in [0, U], sort them,
    add 0 and U as endpoints, and take the gaps.

    Args:
        n: Dimension (number of components).
        U: Target sum (must be > 0).
        rng: Random number generator. Defaults to a fresh generator.

    Returns:
        Array of n non-negative values summing to U.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n == 1:
        return np.array([float(U)])
    if U <= 0:
        raise ValueError(f"U must be positive, got {U}")
    cuts = np.sort(rng.uniform(0, U, size=n - 1))
    return np.diff(np.concatenate([[0.0], cuts, [float(U)]]))


def uunifast_discard(
    n: int,
    U: float,
    umax: Optional[np.ndarray] = None,
    umin: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 1_000_000,
) -> np.ndarray:
    """Rejection-sample the constrained simplex (UUnifast-Discard).

    Provably uniform but exponentially wasteful as the constraints tighten. Used
    as ground truth when testing :func:`drs`, never in the experiments.

    Raises:
        RuntimeError: If no acceptable point is found within `max_attempts`.
    """
    if rng is None:
        rng = np.random.default_rng()
    umax = np.ones(n) if umax is None else np.asarray(umax, dtype=float)
    umin = np.zeros(n) if umin is None else np.asarray(umin, dtype=float)

    for _ in range(max_attempts):
        x = uunifast(n, U, rng=rng)
        if np.all(x <= umax + _TOL) and np.all(x >= umin - _TOL):
            return x
    raise RuntimeError(f"UUnifast-Discard found no valid point in {max_attempts} attempts")


# ---------------------------------------------------------------------------
# DRS sub-functions
# ---------------------------------------------------------------------------


def _cts(r: np.ndarray) -> np.ndarray:
    """CtS(r): the constraints simplex for constraint vector r.

    Vertex i is r with its i-th component replaced by 1 - sum(r) + r_i, so the
    simplex bounds {x : x_i <= r_i} on the hyperplane sum(x) = 1. Components may
    legitimately be negative.

    Returns:
        An (n, n) array whose rows are the vertices.
    """
    n = len(r)
    s = np.tile(r, (n, 1))
    np.fill_diagonal(s, 1.0 - r.sum() + r)
    return s


def _rmss_apply(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply RMSS(CtS(r)) to a point p.

    RMSS returns the affine map sending each vertex of a regular simplex to the
    corresponding standard basis vector, and an affine map is fixed by those
    images. For S = CtS(r) the map has a closed form: vertex i of S is r with
    its i-th component replaced by 1 - sum(r) + r_i, so

        (v_i - r) / (1 - sum(r)) = e_i

    and therefore the map is p -> (p - r) / (1 - sum(r)). Using it directly is
    exact, O(n) rather than O(n^3), and avoids inverting a matrix that becomes
    ill-conditioned as the simplex shrinks.
    """
    return (p - r) / (1.0 - r.sum())


def _rmss_invert(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_rmss_apply`: map the standard simplex back onto CtS(r)."""
    return p * (1.0 - r.sum()) + r


def _simplex_scale(r: np.ndarray) -> float:
    """Edge-length ratio of CtS(r) to the standard simplex.

    Vertices i and j of CtS(r) differ by (1 - sum(r)) * (e_i - e_j), so the
    constraints simplex is the standard simplex scaled by |1 - sum(r)|, and the
    volume ratio in n-1 dimensions is that value to the power n-1.
    """
    return abs(1.0 - float(r.sum()))


def _rescale(r: np.ndarray, p: np.ndarray, max_iterations: int) -> Optional[np.ndarray]:
    """Rescale(r, P): fold P into the region where every constraint holds.

    Each fold builds the simplex spanned by the broken constraints -- which
    contains P by construction -- and maps it onto the standard simplex.

    Returns None if the fold did not converge within `max_iterations`, which the
    caller handles by drawing a fresh point.
    """
    for i in range(max_iterations):
        broken = p > r + _TOL
        if not broken.any():
            return p
        b = np.where(broken, r, 0.0)
        p = _rmss_apply(b, p)
    return None


def _ssr(r: np.ndarray, p: np.ndarray, max_iterations: int) -> Optional[np.ndarray]:
    """SmallestSimplexRescale(r, P).

    Points generated in the smaller of the two simplices break at most n-1 of
    the n constraints, which is what keeps the fold converging quickly. When the
    constraints simplex is the smaller one the problem is transposed, solved,
    and transposed back.
    """
    total = float(r.sum())
    if abs(1.0 - total) >= 1.0:
        return _rescale(r, p, max_iterations)

    # Transposed constraints: the image of e_j under RMSS(CtS(r)) is
    # (e_j - r) / (1 - sum(r)), whose off-diagonal entry in column k is
    # -r_k / (1 - sum(r)).
    q = r / (total - 1.0)
    t = _rescale(q, p, max_iterations)
    if t is None:
        return None
    return _rmss_invert(r, t)


# ---------------------------------------------------------------------------
# DRS
# ---------------------------------------------------------------------------


def drs(
    n: int,
    U: float,
    umax: Optional[np.ndarray] = None,
    umin: Optional[np.ndarray] = None,
    epsilon: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
    max_retries: int = 50,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Draw a point uniformly from the constrained simplex.

    Samples uniformly from {x : sum(x) = U, umin_i <= x_i <= umax_i}.

    Args:
        n: Dimension.
        U: Target sum.
        umax: Upper bounds (default: ones(n)).
        umin: Lower bounds (default: zeros(n)).
        epsilon: Tolerance on the sum of the returned vector.
        rng: Random number generator, for reproducibility.
        max_retries: Fresh initial points to try before giving up.
        max_iterations: Fold iterations allowed per initial point.

    Returns:
        Array of n values summing to U within epsilon, within [umin, umax].

    Raises:
        ValueError: If the constraints are infeasible.
    """
    if rng is None:
        rng = np.random.default_rng()

    umax = np.ones(n, dtype=float) if umax is None else np.array(umax, dtype=float)
    umin = np.zeros(n, dtype=float) if umin is None else np.array(umin, dtype=float)

    if umax.shape != (n,) or umin.shape != (n,):
        raise ValueError(f"umax and umin must both have length {n}")
    if np.any(umax < umin - 1e-12):
        raise ValueError("umax must be elementwise >= umin")
    if U < np.sum(umin) - epsilon:
        raise ValueError(f"U={U} < sum(umin)={np.sum(umin)}")
    if U > np.sum(umax) + epsilon:
        raise ValueError(f"U={U} > sum(umax)={np.sum(umax)}")

    # Step 2: canonical form, with minimum constraints of zero.
    span = umax - umin
    u_prime = U - float(np.sum(umin))

    if n == 1:
        return np.array([float(U)])
    if u_prime <= epsilon:
        return umin.copy()
    total_span = float(np.sum(span))
    if u_prime >= total_span - epsilon:
        # Every component is pinned to its maximum.
        return umin + span

    # Components with zero span cannot take any share of the budget.
    free = span > 0
    if not free.all():
        sub = drs(
            int(free.sum()),
            u_prime,
            umax=span[free],
            umin=np.zeros(int(free.sum())),
            epsilon=epsilon,
            rng=rng,
            max_retries=max_retries,
            max_iterations=max_iterations,
        )
        out = np.zeros(n)
        out[free] = sub
        return out + umin

    r = span / u_prime
    for _ in range(max_retries):
        # Step 3: an initial point drawn from the flat Dirichlet distribution,
        # i.e. uniformly over the standard simplex.
        p = rng.dirichlet(np.ones(n))
        # Step 4: fold it into the valid region and undo the normalisation.
        solved = _ssr(r, p, max_iterations)
        if solved is None:
            continue
        x = u_prime * solved
        if abs(x.sum() - u_prime) > epsilon:
            continue
        if np.any(x < -epsilon) or np.any(x > span + epsilon):
            continue
        return np.clip(x, 0.0, span) + umin

    warnings.warn(
        f"DRS did not converge after {max_retries} retries for n={n}, U={U}; "
        f"falling back to rejection sampling",
        stacklevel=2,
    )
    return uunifast_discard(n, u_prime, umax=span, rng=rng) + umin
