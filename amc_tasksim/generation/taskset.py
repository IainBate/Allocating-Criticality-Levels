"""Task-set generation for AMC scheduling experiments.

The default mode reproduces the procedure of "Analysis-Runtime Co-design for
Adaptive Mixed-Criticality Scheduling" (RTAS 2022), Section V-C:

1. The number of HI-criticality tasks is ``n * CP``.
2. Periods are drawn log-uniformly over a range with a factor of 100 between
   the smallest and largest, representing 10 ms to 1 s at a 0.1 ms tick.
   Deadlines are implicit.
3. HI-criticality utilisations ``U_i(HI)`` are drawn by DRS for the
   HI-criticality tasks, summing to ``U(HI) = CP * CF * U``.
4. LO-criticality utilisations ``U_i(LO)`` are drawn by DRS for *all* tasks,
   summing to ``U``, with each HI-criticality task constrained to
   ``[0, U_i(HI)]`` -- which is what guarantees ``C_i(LO) <= C_i(HI)`` -- and
   each LO-criticality task to ``[0, 1]``.
5. Execution times follow as ``C_i(x) = U_i(x) * T_i``, and BCET is drawn
   uniformly between 80% and 100% of ``C_i(LO)``.

Note that CF is a ratio of *aggregate* utilisations, not a per-task multiplier:
individual tasks have a spread of C(HI)/C(LO) ratios. The legacy
``fixed_ratio`` mode, which sets ``C_i(HI) = CF * C_i(LO)`` for every task, is
retained for comparison but is not what the papers do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .drs import drs

HiMode = Literal["drs_independent", "fixed_ratio"]
PeriodMode = Literal["log_uniform", "semi_harmonic"]

# The AMC-RH paper's two base-frequency families (Bate & Burns 2003), in ms,
# converted to ticks at 0.1ms/tick (the tick unit generate_taskset's own
# defaults assume: period_range=(100, 10000) ticks == 10ms-1s, matching the
# paper's "non-harmonic" range exactly). "Semi-harmonic periods were chosen
# at random from a set of harmonics of two base frequencies" -- read here as
# the union of both families, each task's period drawn from it uniformly.
_SEMI_HARMONIC_PERIODS_TICKS = tuple(
    sorted(
        int(round(ms * 10))
        for ms in (25, 50, 100, 250, 500, 1000, 20, 40, 80, 200, 400, 800)
    )
)


@dataclass
class TaskSet:
    """A single synthetic mixed-criticality task set.

    Attributes:
        n: Number of tasks.
        criticality: "HI" or "LO" per task.
        T: Periods in integer ticks.
        D: Deadlines in integer ticks (== T for implicit deadlines).
        C_lo: Execution time budgets in normal mode.
        C_hi: Execution time budgets in degraded mode.
        BCET: Best-case execution times.
        priority: Assigned priority order (lower number = higher priority).
        metadata: Generation parameters for reproducibility.
    """

    n: int
    criticality: list[str]
    T: list[int]
    D: list[int]
    C_lo: list[int]
    C_hi: list[int]
    BCET: list[int]
    priority: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Derived diagnostics
    individually_infeasible_count: int = 0
    individually_infeasible_indices: list[int] = field(default_factory=list)
    aggregate_hi_utilisation: float = 0.0
    zero_budget_count: int = 0

    def __len__(self) -> int:
        return self.n

    @property
    def C_hi_array(self) -> np.ndarray:
        return np.array(self.C_hi, dtype=float)

    @property
    def C_lo_array(self) -> np.ndarray:
        return np.array(self.C_lo, dtype=float)

    @property
    def T_array(self) -> np.ndarray:
        return np.array(self.T, dtype=float)

    @property
    def U_lo(self) -> np.ndarray:
        """Per-task utilisation Ci(LO) / Ti."""
        return self.C_lo_array / self.T_array

    @property
    def U_hi(self) -> np.ndarray:
        """Per-task HI-criticality utilisation Ci(HI) / Ti (zero for LO tasks)."""
        u = np.zeros(self.n)
        for i in range(self.n):
            if self.criticality[i] == "HI":
                u[i] = self.C_hi[i] / self.T[i]
        return u


def _min_utilisation(T: np.ndarray, budget: float) -> np.ndarray:
    """Lower bounds that make every C_i round to at least one tick.

    An execution time of zero is not a task, and it also breaks BCET <= C(LO).
    Requesting u_i >= 1/T_i fixes that, but only if the total target leaves room;
    at very low utilisations it does not, and the caller records how many tasks
    ended up with a zero budget instead.
    """
    floor = 1.0 / T
    if floor.sum() > 0.5 * budget:
        return np.zeros_like(T)
    return floor


def generate_taskset(
    n: int = 20,
    CP: float = 0.5,
    U: float = 0.8,
    CF: float = 2.0,
    N: int = 10000,
    hi_mode: HiMode = "drs_independent",
    period_range: tuple[int, int] = (100, 10000),
    bcet_fraction_range: tuple[float, float] = (0.8, 1.0),
    rng_seed: int | None = None,
) -> TaskSet:
    """Generate a single mixed-criticality task set.

    Args:
        n: Number of tasks.
        CP: Criticality proportion -- the fraction of tasks that are HI.
        U: Target total LO-criticality utilisation.
        CF: Criticality factor. In the default mode this is the ratio of total
            HI-criticality utilisation to total LO-criticality utilisation of
            the HI-criticality tasks; in `fixed_ratio` it is a per-task
            multiplier.
        N: Inverse failure probability, recorded in metadata (FP = 1/N).
        hi_mode: "drs_independent" for the papers' procedure, "fixed_ratio" for
            the legacy per-task multiplier.
        period_range: (min, max) for log-uniform period generation, in ticks.
        bcet_fraction_range: (min, max) fraction of C_i(LO) for BCET.
        rng_seed: Random seed for reproducibility.

    Returns:
        A TaskSet with all fields populated.
    """
    rng = np.random.default_rng(rng_seed)

    n_hi = round(CP * n)
    n_lo = n - n_hi
    criticality = ["HI"] * n_hi + ["LO"] * n_lo

    # Periods first: the utilisation bounds that keep every budget at one tick
    # or more depend on them.
    log_periods = rng.uniform(math.log(period_range[0]), math.log(period_range[1]), size=n)
    T = np.maximum(np.round(np.exp(log_periods)).astype(int), 1)
    D = T.copy()

    if hi_mode == "drs_independent":
        u_lo, u_hi = _utilisations_paper(n, n_hi, U, CP, CF, T, rng)
    elif hi_mode == "fixed_ratio":
        u_lo, u_hi = _utilisations_fixed_ratio(n, n_hi, U, CF, T, rng)
    else:
        raise ValueError(f"Unknown hi_mode: {hi_mode}")

    C_lo = np.round(u_lo * T).astype(int)
    C_hi = np.round(u_hi * T).astype(int)
    C_hi = np.maximum(C_hi, C_lo)  # rounding must never invert C_lo <= C_hi

    bcet_fracs = rng.uniform(bcet_fraction_range[0], bcet_fraction_range[1], size=n)
    BCET = np.minimum(np.round(C_lo * bcet_fracs).astype(int), C_lo)

    infeasible = [i for i in range(n_hi) if C_hi[i] > T[i]]
    C_hi = np.minimum(C_hi, T)

    ts = TaskSet(
        n=n,
        criticality=criticality,
        T=T.tolist(),
        D=D.tolist(),
        C_lo=C_lo.tolist(),
        C_hi=C_hi.tolist(),
        BCET=BCET.tolist(),
        metadata={
            "seed": rng_seed,
            "target_U": U,
            "CP": CP,
            "CF": CF,
            "N": N,
            "hi_mode": hi_mode,
            "period_range": period_range,
            "bcet_fraction_range": bcet_fraction_range,
        },
        individually_infeasible_count=len(infeasible),
        individually_infeasible_indices=infeasible,
        aggregate_hi_utilisation=float(sum(C_hi[i] / T[i] for i in range(n_hi))),
        zero_budget_count=int((C_lo == 0).sum()),
    )
    return ts


def _utilisations_paper(
    n: int,
    n_hi: int,
    U: float,
    CP: float,
    CF: float,
    T: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """RTAS 2022 Section V-C: HI utilisations first, then LO utilisations
    constrained by them."""
    u_hi = np.zeros(n)

    if n_hi > 0:
        target_hi = CP * CF * U
        floor_hi = _min_utilisation(T[:n_hi], target_hi)
        u_hi[:n_hi] = drs(
            n_hi, target_hi, umax=np.ones(n_hi), umin=floor_hi, rng=rng
        )

    # LO-criticality utilisations for every task, with HI-criticality tasks
    # capped at their own HI-criticality utilisation so C_i(LO) <= C_i(HI).
    umax = np.ones(n)
    umax[:n_hi] = u_hi[:n_hi]
    floor_lo = np.minimum(_min_utilisation(T, U), umax)
    u_lo = drs(n, U, umax=umax, umin=floor_lo, rng=rng)

    return u_lo, u_hi


def _utilisations_fixed_ratio(
    n: int,
    n_hi: int,
    U: float,
    CF: float,
    T: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy mode: C_i(HI) = CF * C_i(LO) for every HI-criticality task."""
    floor = _min_utilisation(T, U)
    u_lo = drs(n, U, umax=np.ones(n), umin=floor, rng=rng)
    u_hi = np.zeros(n)
    u_hi[:n_hi] = CF * u_lo[:n_hi]
    return u_lo, u_hi


def generate_ensemble(
    n_replicates: int,
    U: float,
    rng_seed: int = 42,
    **kwargs,
) -> list[TaskSet]:
    """Generate an ensemble of task sets at one utilisation level.

    Seeds are derived from both U and the replicate index, so different
    utilisation levels get independent periods rather than reusing one set of
    draws across the whole sweep.

    Args:
        n_replicates: Number of task sets to generate.
        U: Target utilisation, shared across the ensemble.
        rng_seed: Base seed.
        **kwargs: Passed through to generate_taskset.

    Returns:
        List of TaskSet objects.
    """
    base = rng_seed + int(round(U * 1000)) * 1_000_003
    return [generate_taskset(U=U, rng_seed=base + i, **kwargs) for i in range(n_replicates)]
