"""Experiment sweep orchestration for AMC task-set generation and simulation.

Runs the utilisation x failure-probability sweep and stores results as a tidy
DataFrame, one row per (U, N, protocol, replicate).

Two things follow the papers rather than convenience:

- Task sets are filtered to the population the papers evaluate: unschedulable
  under single-criticality FPPS, but schedulable under AMC-rtb with Audsley's
  optimal priority assignment. Generation continues until the requested number
  of *qualifying* task sets is reached.
- Every protocol runs on the same task sets with the same seeds, so the
  comparison is like-for-like and Table I of the AMC-RH paper can be computed
  directly from one sweep.

The simulation duration is expressed in jobs of the longest-period task, as the
paper does ("sufficient for 10^6 jobs of the task with the longest period"),
so it scales with the periods actually drawn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import amc_rtb, fpps_schedulable
from amc_tasksim.scheduling.priority import assign_audsley_opa
from amc_tasksim.simulation.engine import (
    AMC_RA,
    AMC_RH,
    OriginalAMC,
    SimulationResult,
    simulate,
)

PROTOCOLS = ("original_amc", "amc_ra", "amc_rh")


@dataclass(frozen=True)
class Scale:
    """A sizing preset for the sweep.

    Attributes:
        n_replicates: Qualifying task sets per utilisation level.
        duration_jobs: Simulation length, in jobs of the longest-period task.
        U_values: Utilisation levels.
        N_values: Failure-probability levels (FP = 1/N).
    """

    n_replicates: int
    duration_jobs: int
    U_values: tuple[float, ...]
    N_values: tuple[int, ...]


SCALES = {
    # Small enough to iterate on in minutes, large enough that the metrics are
    # within reach of the paper's. Utilisations are restricted to the band where
    # the non-trivial population actually exists.
    "debug": Scale(
        n_replicates=20,
        duration_jobs=200,
        U_values=(0.6, 0.7, 0.8, 0.9),
        N_values=(100, 1000, 10000),
    ),
    # The configuration of RTAS 2022 Section V-D.
    "paper": Scale(
        n_replicates=500,
        duration_jobs=10**6,
        U_values=(0.6, 0.7, 0.8, 0.9),
        N_values=(100, 1000, 10000, 100000),
    ),
}


def _default_output() -> str:
    return str(Path("results") / "sweep.parquet")


def build_population(
    n_replicates: int,
    U: float,
    seed: int,
    max_attempts_factor: int = 400,
    **kwargs,
) -> tuple[list[TaskSet], int]:
    """Generate task sets until `n_replicates` of them qualify.

    A task set qualifies if it is unschedulable under single-criticality FPPS
    (so it genuinely needs AMC) and schedulable under AMC-rtb with Audsley's
    optimal priority assignment. Qualifying task sets are returned with their
    OPA priorities already assigned.

    Returns:
        (task sets, number of candidates generated)
    """
    base = seed + int(round(U * 1000)) * 1_000_003
    accepted: list[TaskSet] = []
    attempts = 0
    limit = n_replicates * max_attempts_factor

    while len(accepted) < n_replicates and attempts < limit:
        ts = generate_taskset(U=U, rng_seed=base + attempts, **kwargs)
        attempts += 1
        if fpps_schedulable(ts):
            continue  # no mixed-criticality scheme needed
        if not assign_audsley_opa(ts):
            continue  # not schedulable under AMC-rtb by any priority ordering
        accepted.append(ts)

    return accepted, attempts


def run_sweep(
    scale: str = "debug",
    U_values: Optional[Sequence[float]] = None,
    N_values: Optional[Sequence[int]] = None,
    n_replicates: Optional[int] = None,
    duration_jobs: Optional[int] = None,
    protocols: Sequence[str] = PROTOCOLS,
    hi_mode: str = "drs_independent",
    n: int = 20,
    CP: float = 0.5,
    CF: float = 2.0,
    seed: int = 42,
    output: Optional[str] = None,
    power_threshold: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the AMC experiment sweep.

    Args:
        scale: Sizing preset, "debug" or "paper". Any of the sizing arguments
            below override it individually.
        U_values: Utilisation levels.
        N_values: Failure-probability levels (FP = 1/N).
        n_replicates: Qualifying task sets per utilisation level.
        duration_jobs: Simulation length in jobs of the longest-period task.
        protocols: Which mode-change protocols to simulate.
        hi_mode: Task-set generation mode.
        n: Tasks per task set.
        CP: Criticality proportion.
        CF: Criticality factor.
        seed: Base random seed.
        output: Output parquet path. None writes nothing.
        power_threshold: Warn when a cell sees fewer HI-behaviour jobs than this.
        verbose: Print progress.

    Returns:
        DataFrame with one row per (U, N, protocol, replicate).
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
    preset = SCALES[scale]

    U_values = tuple(preset.U_values if U_values is None else U_values)
    N_values = tuple(preset.N_values if N_values is None else N_values)
    n_replicates = preset.n_replicates if n_replicates is None else n_replicates
    duration_jobs = preset.duration_jobs if duration_jobs is None else duration_jobs

    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocols {sorted(unknown)}; expected {list(PROTOCOLS)}")

    rows: list[dict] = []
    total_cells = len(U_values) * len(N_values)
    cell = 0

    for U in U_values:
        population, attempts = build_population(
            n_replicates=n_replicates,
            U=U,
            seed=seed,
            n=n,
            CP=CP,
            CF=CF,
            hi_mode=hi_mode,
        )
        if verbose:
            yield_pct = 100.0 * len(population) / attempts if attempts else 0.0
            print(
                f"\nU={U:.2f}: {len(population)} qualifying task sets "
                f"from {attempts} candidates ({yield_pct:.1f}% yield)"
            )
        if not population:
            if verbose:
                print(f"  WARNING: no non-trivial schedulable task sets at U={U:.2f}; skipping")
            continue
        if len(population) < n_replicates and verbose:
            print(
                f"  WARNING: only {len(population)} of {n_replicates} task sets found at "
                f"U={U:.2f}; the non-trivial population is thin here"
            )

        # Cached per task set: response times for the response-time protocols,
        # and the duration implied by the longest period.
        analyses = [(amc_rtb(ts).r_lo, duration_jobs * max(ts.T)) for ts in population]

        for N in N_values:
            cell += 1
            fp = 1.0 / N
            if verbose:
                print(f"  [{cell}/{total_cells}] U={U:.2f}, N={N}, FP={fp:.0e}")

            triggers = 0
            for i, (ts, (r_lo, duration)) in enumerate(zip(population, analyses)):
                for name in protocols:
                    if name == "amc_rh":
                        protocol = AMC_RH(r_lo)
                    elif name == "amc_ra":
                        protocol = AMC_RA(r_lo)
                    else:
                        protocol = OriginalAMC()

                    r = simulate(
                        ts,
                        duration=duration,
                        seed=seed + i,
                        mode_protocol=protocol,
                        fp=fp,
                    )
                    if name == protocols[0]:
                        triggers += r.hi_trigger_events

                    rows.append(
                        {
                            "U": U,
                            "N": N,
                            "FP": fp,
                            "hi_mode": hi_mode,
                            "protocol": name,
                            "replicate_index": i,
                            "duration": duration,
                            "nid": r.nid,
                            "tid": r.tid,
                            "jne": r.jne,
                            "ldm": r.ldm,
                            "hdm": r.hdm,
                            "nid_pct": r.nid_pct,
                            "tid_pct": r.tid_pct,
                            "jne_ldm_pct": r.jne_ldm_pct,
                            "hi_trigger_events": r.hi_trigger_events,
                            "total_hi_releases": r.total_hi_releases,
                            "total_lo_releases": r.total_lo_releases,
                            "budget_overruns": r.budget_overruns,
                            "zero_budget_count": ts.zero_budget_count,
                            "aggregate_hi_utilisation": ts.aggregate_hi_utilisation,
                        }
                    )

            if triggers < power_threshold and verbose:
                print(
                    f"      WARNING: only {triggers} HI-behaviour jobs in this cell; "
                    f"NiD/TiD/JNE estimates will be noisy"
                )

    df = pd.DataFrame(rows)
    if output and not df.empty:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        df.to_parquet(output, index=False)
        if verbose:
            print(f"\nResults saved to {output} ({len(df)} rows)")
    return df
