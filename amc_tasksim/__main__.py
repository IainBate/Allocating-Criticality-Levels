"""CLI entry point for amc-tasksim."""

import argparse
import sys


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
    args = parser.parse_args()

    # Placeholder — will be wired up in Phase 7
    print(f"Mode: {args.hi_mode}, quick={args.quick}, replicates={args.n_replicates}, duration={args.duration}")
    print("Phase 1 scaffold complete — algorithm logic to be added in subsequent phases.")
    sys.exit(0)


if __name__ == "__main__":
    main()
