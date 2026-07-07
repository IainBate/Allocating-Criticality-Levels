"""Experiment sweep orchestration for AMC task-set generation and simulation.

Runs the full utilisation x failure-probability sweep, parallelises across
replicates, and stores results as a tidy pandas DataFrame / parquet file.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from amc_tasksim.generation.taskset import generate_taskset, generate_ensemble
from amc_tasksim.scheduling.amc_rtb import amc_rtb, is_nontrivial_amc_taskset
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import simulate, SimulationResult, ModeChangeProtocol, OriginalAMC
from amc_tasksim.simulation.protocols import AMC_RH, AMC_RA


def _default_output() -> str:
    return str(Path("results") / "sweep.parquet")


def run_sweep(
    U_range: tuple[float, float, float] = (0.05, 0.95, 0.05),
    N_values: list[int] = None,
    n_replicates: int = 1000,
    hi_mode: str = "fixed_ratio",
    duration: int = 10**6,
    seed: int = 42,
    mode_protocol: Optional[str] = None,
    output: Optional[str] = None,
    quick: bool = False,
    power_threshold: int = 100,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Run the full AMC experiment sweep.

    Args:
        U_range: (start, stop, step) for utilisation sweep.
        N_values: List of N values for failure-probability sweep (FP = 1/N).
        n_replicates: Number of task-set replicates per (U, N).
        hi_mode: HI-criticality utilisation mode.
        duration: Simulation duration per run.
        seed: Base random seed.
        mode_protocol: Protocol name ("original_amc", "amc_rh", "amc_ra")
            or a ModeChangeProtocol instance (default: "original_amc").
        output: Output parquet file path.
        quick: If True, use 20 replicates instead of 1000.
        power_threshold: Minimum HI-trigger events for statistical power.
        n_workers: Number of parallel workers (1 = sequential).

    Returns:
        DataFrame with sweep results.
    """
    if N_values is None:
        N_values = [10, 100, 1000, 10000, 100000]

    if U_range is None:
        U_range = (0.05, 0.95, 0.05)

    if quick:
        n_replicates = 20

    if output is None:
        output = _default_output()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    U_values = []
    u = U_range[0]
    while u <= U_range[1] + 1e-9:
        U_values.append(round(u, 4))
        u += U_range[2]

    all_rows = []
    total_combos = len(U_values) * len(N_values)
    combo_count = 0

    for U in U_values:
        # Generate task sets for this U (once, reused across all N)
        print(f"\nGenerating {n_replicates} task sets for U={U:.2f} ...")
        ensemble = generate_ensemble(
            n_replicates=n_replicates,
            U=U,
            hi_mode=hi_mode,
            rng_seed=seed,
        )

        for N in N_values:
            combo_count += 1
            fp = 1.0 / N
            print(f"  [{combo_count}/{total_combos}] U={U:.2f}, N={N}, FP={fp:.0e} ...")

            # Simulate each replicate with this N
            results: list[SimulationResult] = []
            for i, ts in enumerate(ensemble):
                assign_deadline_monotonic(ts)

                # Determine protocol
                protocol = mode_protocol
                if isinstance(protocol, str):
                    if protocol == "amc_rh":
                        rt_result = amc_rtb(ts)
                        protocol = AMC_RH(rt_result.r_lo)
                    elif protocol == "amc_ra":
                        rt_result = amc_rtb(ts)
                        protocol = AMC_RA(rt_result.r_lo)
                    else:
                        protocol = OriginalAMC()

                r = simulate(
                    ts,
                    duration=duration,
                    seed=seed + i,
                    mode_protocol=protocol,
                    fp=fp,
                )
                results.append(r)

            # Aggregate metrics
            total_hi_triggers = sum(r.hi_trigger_events for r in results)
            total_hi_releases = sum(sum(r.hi_releases_per_task) for r in results)

            # Statistical power warning
            if total_hi_triggers < power_threshold:
                print(
                    f"  WARNING: Only {total_hi_triggers} HI-trigger events "
                    f"for (U={U:.2f}, N={N}). Estimates may be unreliable."
                )

            for i, (ts, r) in enumerate(zip(ensemble, results)):
                row = {
                    "U": U,
                    "N": N,
                    "FP": fp,
                    "hi_mode": hi_mode,
                    "protocol": mode_protocol if isinstance(mode_protocol, str) else "original_amc",
                    "replicate_index": i,
                    "nid": r.nid,
                    "tid": r.tid,
                    "jne": r.jne,
                    "ldm": r.ldm,
                    "hdm": r.hdm,
                    "hi_trigger_events": r.hi_trigger_events,
                    "total_hi_releases": sum(r.hi_releases_per_task),
                    "individually_infeasible": ts.individually_infeasible_count,
                    "aggregate_hi_utilisation": ts.aggregate_hi_utilisation,
                    "schedulable_amc_rtb": amc_rtb(ts).overall_schedulable,
                    "nontrivial_amc": is_nontrivial_amc_taskset(ts),
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_parquet(output, index=False)
    print(f"\nResults saved to {output} ({len(df)} rows)")
    return df


def main():
    """CLI entry point for the sweep."""
    import argparse

    parser = argparse.ArgumentParser(description="AMC Experiment Sweep")
    parser.add_argument("--quick", action="store_true", help="Run with 20 replicates")
    parser.add_argument("--hi-mode", choices=["fixed_ratio", "drs_independent"], default="fixed_ratio")
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--duration", type=int, default=10**6)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    n_replicates = args.n_replicates if args.n_replicates else (20 if args.quick else 1000)
    output = args.output if args.output else _default_output()

    df = run_sweep(
        n_replicates=n_replicates,
        hi_mode=args.hi_mode,
        duration=args.duration,
        output=output,
        quick=args.quick,
        n_workers=args.n_workers,
    )

    # Print summary
    print("\n--- Summary ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nHI-trigger events by (U, N):")
    power = df.groupby(["U", "N"])["hi_trigger_events"].sum()
    print(power)


if __name__ == "__main__":
    main()
