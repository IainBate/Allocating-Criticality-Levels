"""Task-set generation for AMC scheduling experiments.

Builds on the DRS (Dirichlet-Rescale) algorithm to produce synthetic
mixed-criticality task sets with controlled utilisation and period distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .drs import drs


@dataclass
class TaskSet:
    """A single synthetic mixed-criticality task set.

    Attributes:
        n: Number of tasks.
        criticality: Array of "HI" or "LO" per task.
        T: Periods in integer ticks.
        D: Deadlines in integer ticks (== T for implicit deadlines).
        C_lo: Execution time budgets in normal (LO) mode.
        C_hi: Execution time budgets in degraded (HI) mode.
        BCET: Best-case execution times.
        priority: Assigned priority order (filled in Phase 4).
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
        """Per-task HI-criticality utilisation Ci(HI) / Ti (for HI tasks)."""
        U = np.zeros(self.n)
        for i in range(self.n):
            if self.criticality[i] == "HI":
                U[i] = self.C_hi[i] / self.T[i]
        return U


def generate_taskset(
    n: int = 20,
    CP: float = 0.5,
    U: float = 0.5,
    CF: float = 1.5,
    N: int = 10000,
    hi_mode: Literal["fixed_ratio", "drs_independent"] = "fixed_ratio",
    period_range: tuple[int, int] = (100, 10000),
    bcet_fraction_range: tuple[float, float] = (0.8, 1.0),
    rng_seed: int | None = None,
) -> TaskSet:
    """Generate a single mixed-criticality task set.

    Args:
        n: Number of tasks.
        CP: Proportion of tasks that are HI-criticality.
        U: Target total utilisation.
        CF: HI/LO criticality factor (Ci(HI) = CF * Ci(LO) in fixed_ratio mode).
        N: Inverse failure probability (FP = 1/N).
        hi_mode: How to generate HI-criticality utilisation.
        period_range: (min, max) for log-uniform period generation.
        bcet_fraction_range: (min, max) fraction of Ci(LO) for BCET.
        rng_seed: Random seed for reproducibility.

    Returns:
        A TaskSet with all fields populated.
    """
    rng = np.random.default_rng(rng_seed)

    # Step 1: assign criticality levels
    n_hi = round(CP * n)
    n_lo = n - n_hi
    criticality = ["HI"] * n_hi + ["LO"] * n_lo

    # Step 2: per-task utilisation via DRS
    umax = np.ones(n)
    umin = np.zeros(n)
    u = drs(n, U, umax=umax, umin=umin)

    # Step 3: periods — log-uniform
    log_min = math.log(period_range[0])
    log_max = math.log(period_range[1])
    log_periods = rng.uniform(log_min, log_max, size=n)
    T = np.maximum(np.round(np.exp(log_periods)).astype(int), 1)
    D = T.copy()

    # Step 4: C_lo and BCET
    C_lo = np.round(u * T).astype(int)
    bcet_fracs = rng.uniform(bcet_fraction_range[0], bcet_fraction_range[1], size=n)
    BCET = np.round(C_lo * bcet_fracs).astype(int)
    BCET = np.maximum(BCET, 1)  # BCET must be at least 1

    # Step 5: C_hi for HI-criticality tasks
    C_hi = C_lo.copy()
    infeasible_indices: list[int] = []

    if hi_mode == "fixed_ratio":
        for i in range(n_hi):
            c_hi = round(CF * C_lo[i])
            c_hi = min(c_hi, T[i])  # cap at period
            C_hi[i] = c_hi
            if c_hi > T[i]:
                infeasible_indices.append(i)
            # Note: c_hi was capped above, so check before capping
            # Re-check: flag if original CF*C_lo[i] / T[i] > 1
            if (CF * C_lo[i]) / T[i] > 1.0:
                if i not in infeasible_indices:
                    infeasible_indices.append(i)

    elif hi_mode == "drs_independent":
        # Two-level DRS for HI-criticality tasks
        target_u_hi_hi = CF * CP * U
        u_hi_max = drs(n_hi, target_u_hi_hi, umax=np.ones(n_hi), umin=np.zeros(n_hi))
        # Use u_hi_max as constraint for HI tasks' LO utilisation
        u_hi = drs(n_hi, CP * U, umax=u_hi_max, umin=np.zeros(n_hi))
        u[:n_hi] = u_hi
        # Recalculate C_lo from corrected utilisation
        C_lo = np.round(u_hi * T[:n_hi]).astype(int)
        BCET[:n_hi] = np.maximum(np.round(C_lo * bcet_fracs[:n_hi]).astype(int), 1)
        # C_hi = CF * C_lo, capped at period
        for i in range(n_hi):
            c_hi = min(round(CF * C_lo[i]), T[i])
            C_hi[i] = c_hi
    else:
        raise ValueError(f"Unknown hi_mode: {hi_mode}")

    # Recompute infeasible after any corrections
    infeasible_indices = [
        i for i in range(n_hi) if (CF * C_lo[i]) / T[i] > 1.0
    ]

    # Step 6: aggregate HI utilisation
    aggregate_hi_util = sum(C_hi[i] / T[i] for i in range(n_hi))

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
        individually_infeasible_count=len(infeasible_indices),
        individually_infeasible_indices=infeasible_indices,
        aggregate_hi_utilisation=aggregate_hi_util,
    )
    return ts


def generate_ensemble(
    n_replicates: int,
    U: float,
    **kwargs,
) -> list[TaskSet]:
    """Generate an ensemble of task sets.

    Uses deterministic seeding derived from (U, replicate_index) so the
    same (U, n_replicates) always produces the same ensemble.

    Args:
        n_replicates: Number of task sets to generate.
        U: Target utilisation (shared across the ensemble).
        **kwargs: Passed through to generate_taskset.

    Returns:
        List of TaskSet objects.
    """
    base_seed = int(hash((U, n_replicates)) % (2**31))
    task_sets: list[TaskSet] = []
    for i in range(n_replicates):
        seed = base_seed + i
        ts = generate_taskset(U=U, rng_seed=seed, **kwargs)
        task_sets.append(ts)
    return task_sets
