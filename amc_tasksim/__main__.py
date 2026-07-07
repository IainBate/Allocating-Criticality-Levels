"""CLI entry point for amc-tasksim."""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from amc_tasksim.experiments.sweep import run_sweep


def _clean(dry_run: bool = False, include_venv: bool = False) -> list[str]:
    """Remove sweep results, figures, cache, and build artifacts.

    Returns a list of paths that would be (or were) removed.
    """
    removed: list[str] = []

    patterns = [
        "results/sweep*.parquet",
        "results/sweep*.csv",
        "results/figures/",
        "results/SUMMARY.md",
        ".pytest_cache/",
        "build/",
        "dist/",
        "*.egg-info/",
        "*.egg",
    ]

    for pat in patterns:
        for entry in glob.glob(pat):
            full = os.path.abspath(entry)
            if os.path.isdir(full):
                removed.append(full)
                if not dry_run:
                    shutil.rmtree(full)
            else:
                removed.append(full)
                if not dry_run:
                    os.remove(full)

    # Also clean __pycache__ directories recursively (skip .venv)
    for root, dirs, _ in os.walk("."):
        # Skip .venv and its subdirs
        if ".venv" in root.split(os.sep):
            continue
        if "__pycache__" in dirs:
            pcache = os.path.join(root, "__pycache__")
            removed.append(pcache)
            if not dry_run:
                shutil.rmtree(pcache)

    # Clean .venv only if --clean-all is set
    if include_venv and os.path.isdir(".venv"):
        removed.append(os.path.abspath(".venv"))
        if not dry_run:
            shutil.rmtree(".venv")

    return removed


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
    parser.add_argument(
        "--protocol",
        choices=["original_amc", "amc_rh", "amc_ra"],
        default="original_amc",
        help="Mode-change protocol (default: original_amc)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean sweep results, figures, and cache before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --clean: list files that would be deleted without deleting them",
    )
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="With --clean: also remove .venv (virtual environment)",
    )
    args = parser.parse_args()

    # Clean if requested
    if args.clean:
        removed = _clean(dry_run=args.dry_run, include_venv=args.clean_all)
        if removed:
            action = "would remove" if args.dry_run else "removed"
            print(f"Cleaning ({action} {len(removed)} paths):")
            for p in removed:
                print(f"  {p}")
        else:
            print("Nothing to clean.")
        if args.dry_run:
            sys.exit(0)

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
        mode_protocol=args.protocol,
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
