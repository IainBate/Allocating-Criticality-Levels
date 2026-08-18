"""CLI entry point for amc-tasksim."""

import argparse
import glob
import os
import shutil
import sys

from amc_tasksim.experiments.sweep import PROTOCOLS, SCALES, run_sweep


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

    for root, dirs, _ in os.walk("."):
        if ".venv" in root.split(os.sep):
            continue
        if "__pycache__" in dirs:
            pcache = os.path.join(root, "__pycache__")
            removed.append(pcache)
            if not dry_run:
                shutil.rmtree(pcache)

    if include_venv and os.path.isdir(".venv"):
        removed.append(os.path.abspath(".venv"))
        if not dry_run:
            shutil.rmtree(".venv")

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMC task-set generation and simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES),
        default="debug",
        help="sizing preset: 'debug' runs in minutes, 'paper' matches RTAS 2022 Section V-D",
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=None,
        help="qualifying task sets per utilisation level (overrides --scale)",
    )
    parser.add_argument(
        "--duration-jobs",
        type=int,
        default=None,
        help="simulation length in jobs of the longest-period task (overrides --scale)",
    )
    parser.add_argument(
        "--U-values",
        type=float,
        nargs="+",
        default=None,
        help="utilisation levels (overrides --scale)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="failure-probability levels, FP = 1/N (overrides --scale)",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
        help="mode-change protocols to simulate on each task set",
    )
    parser.add_argument(
        "--hi-mode",
        choices=["drs_independent", "fixed_ratio"],
        default="drs_independent",
        help="'drs_independent' follows the papers; 'fixed_ratio' is the legacy per-task multiplier",
    )
    parser.add_argument("--tasks", type=int, default=20, help="tasks per task set")
    parser.add_argument("--cp", type=float, default=0.5, help="criticality proportion")
    parser.add_argument("--cf", type=float, default=2.0, help="criticality factor")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument(
        "--output", type=str, default="results/sweep.parquet", help="output parquet path"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="generate figures and the validation report after the sweep",
    )
    parser.add_argument("--clean", action="store_true", help="remove results and caches first")
    parser.add_argument(
        "--dry-run", action="store_true", help="with --clean: list what would be removed"
    )
    parser.add_argument(
        "--clean-all", action="store_true", help="with --clean: also remove .venv"
    )
    args = parser.parse_args()

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

    df = run_sweep(
        scale=args.scale,
        U_values=args.U_values,
        N_values=args.n_values,
        n_replicates=args.n_replicates,
        duration_jobs=args.duration_jobs,
        protocols=args.protocols,
        hi_mode=args.hi_mode,
        n=args.tasks,
        CP=args.cp,
        CF=args.cf,
        seed=args.seed,
        output=args.output,
    )

    if df.empty:
        print("\nNo results produced.")
        return

    print("\n--- Summary ---")
    print(f"Rows: {len(df)}")
    summary = (
        df.groupby(["protocol"])[["nid_pct", "tid_pct", "jne_ldm_pct", "hdm"]]
        .agg({"nid_pct": "mean", "tid_pct": "mean", "jne_ldm_pct": "mean", "hdm": "sum"})
        .round(6)
    )
    print(summary.to_string())

    if args.plots:
        from amc_tasksim.analysis.plots import generate_plots

        for name, path in generate_plots(args.output).items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
