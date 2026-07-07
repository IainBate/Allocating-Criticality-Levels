"""CLI entry point for amc-tasksim."""

import argparse
import sys

from amc_tasksim.experiments.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="AMC Task-Set Generation & Simulation")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick smoke-test sweep (20 replicates instead of 1000)",
    )
    parser.add_argument(
        "--hi-mode",
        choices=["fixed_ratio", "drs_independent"],
        default="fixed_ratio",
        help="HI-criticality utilisation mode (default: fixed_ratio)",
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=None,
        help="Override replicate count (overrides --quick default)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10**6,
        help="Simulation duration in ticks (default: 1000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: results/sweep.parquet)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="N values for FP sweep (default: 10 100 1000 10000 100000)",
    )
    parser.add_argument(
        "--U-range",
        type=float,
        nargs=3,
        default=None,
        metavar=("START", "STOP", "STEP"),
        help="U sweep range (default: 0.05 0.95 0.05)",
    )
    args = parser.parse_args()

    n_replicates = args.n_replicates if args.n_replicates else (20 if args.quick else 1000)
    output = args.output if args.output else "results/sweep.parquet"

    df = run_sweep(
        n_replicates=n_replicates,
        hi_mode=args.hi_mode,
        duration=args.duration,
        output=output,
        quick=args.quick,
        n_workers=args.n_workers,
        N_values=args.n_values,
        U_range=args.U_range,
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
