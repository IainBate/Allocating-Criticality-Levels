"""Analysis and plotting for AMC experiment sweep results.

Produces:
- Box-and-whisker plots for NiD(%), TiD(%), JNE(%)+LDM(%) by U, faceted by N
- Heatmap of Success Ratio (fraction with HDM=0)
- Heatmap of statistical power (HI-trigger events per cell)
- Comparison plot overlaying hi_mode results for JNE+LDM
- SUMMARY.md with observations
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def generate_plots(
    input_path: str = "results/sweep.parquet",
    output_dir: str = "results/figures",
) -> dict[str, str]:
    """Generate all analysis plots from sweep results.

    Args:
        input_path: Path to the sweep results parquet file.
        output_dir: Directory to save figures.

    Returns:
        Dict mapping plot names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_path)
    paths = {}

    # --- 1. Box-and-whisker: NiD(%) ---
    paths["nid_boxplot"] = _plot_metric_boxplot(
        df, "nid", "NiD (times degraded mode entered)",
        output_dir, "nid_boxplot.png",
    )

    # --- 2. Box-and-whisker: TiD(%) ---
    paths["tid_boxplot"] = _plot_metric_boxplot(
        df, "tid", "TiD (fraction of time in degraded mode)",
        output_dir, "tid_boxplot.png",
    )

    # --- 3. Box-and-whisker: JNE+LDM ---
    df_jne = df.copy()
    df_jne["jne_ldm"] = df_jne["jne"] + df_jne["ldm"]
    paths["jne_ldm_boxplot"] = _plot_metric_boxplot(
        df_jne, "jne_ldm", "JNE+LDM (dropped + late LO jobs)",
        output_dir, "jne_ldm_boxplot.png",
    )

    # --- 4. Heatmap: Success Ratio (HDM=0) ---
    paths["success_ratio"] = _plot_success_ratio(df, output_dir)

    # --- 5. Heatmap: Statistical Power ---
    paths["stat_power"] = _plot_stat_power(df, output_dir)

    # --- 6. Comparison: hi_mode overlay for JNE+LDM ---
    if "drs_independent" in df["hi_mode"].values and "fixed_ratio" in df["hi_mode"].values:
        paths["hi_mode_comparison"] = _plot_hi_mode_comparison(df, output_dir)

    # --- 7. SUMMARY.md ---
    summary = _write_summary(df, paths)
    summary_path = Path(output_dir).parent / "SUMMARY.md"
    summary_path.write_text(summary)

    return paths


def _plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_dir: str,
    filename: str,
) -> str:
    """Create a box-and-whisker plot faceted by N."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Subsample for plotting (too many points otherwise)
    plot_df = df.groupby(["U", "N", metric]).size().reset_index(name="count").sample(
        500, random_state=42
    )

    # Get unique N values
    n_values = sorted(df["N"].unique())

    # Plot for first 3 N values (log scale)
    n_display = n_values[:3]

    for ax, n in zip(axes, n_display):
        subset = plot_df[plot_df["N"] == n].sort_values("U")
        # Group by U for boxplot
        groups = [subset[subset["U"] == u][metric].dropna().values for u in sorted(subset["U"].unique())]
        ax.boxplot(groups, whis=1.5)
        ax.set_title(f"N={n}")
        ax.set_xlabel("U")
        ax.set_ylabel(metric)
        ax.set_xticklabels([f"{u:.2f}" for u in sorted(subset["U"].unique())], rotation=45, ha="right")

    fig.suptitle(f"{title} (sampled)")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_success_ratio(df: pd.DataFrame, output_dir: str) -> str:
    """Heatmap of success ratio (fraction with HDM=0)."""
    pivot = df.groupby(["U", "N"])["hdm"].apply(lambda x: (x == 0).mean()).unstack()
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="viridis", shading="auto")
    ax.set_xticks(pivot.columns)
    ax.set_yticks(pivot.index)
    ax.set_xticklabels([f"{c:.0e}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("N (FP = 1/N)")
    ax.set_ylabel("U")
    ax.set_title("Success Ratio (fraction with HDM=0)")
    fig.colorbar(im, ax=ax, label="Success Ratio")
    plt.tight_layout()

    path = os.path.join(output_dir, "success_ratio_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stat_power(df: pd.DataFrame, output_dir: str) -> str:
    """Heatmap of statistical power (total HI-trigger events per cell)."""
    pivot = df.groupby(["U", "N"])["hi_trigger_events"].sum().unstack()
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap="YlOrRd", shading="auto")
    ax.set_xticks(pivot.columns)
    ax.set_yticks(pivot.index)
    ax.set_xticklabels([f"{c:.0e}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("N (FP = 1/N)")
    ax.set_ylabel("U")
    ax.set_title("Statistical Power (total HI-trigger events per cell)")
    fig.colorbar(im, ax=ax, label="HI-trigger events")
    plt.tight_layout()

    path = os.path.join(output_dir, "stat_power_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hi_mode_comparison(df: pd.DataFrame, output_dir: str) -> str:
    """Comparison plot overlaying hi_mode results for JNE+LDM."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for hi_mode in df["hi_mode"].unique():
        subset = df[df["hi_mode"] == hi_mode]
        mean_jne = subset.groupby("U")["jne"].mean()
        mean_ldm = subset.groupby("U")["ldm"].mean()
        ax.plot(mean_jne.index, mean_jne.values + mean_ldm.values, label=hi_mode, marker="o")

    ax.set_xlabel("U")
    ax.set_ylabel("Mean JNE + LDM")
    ax.set_title("JNE+LDM by hi_mode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "hi_mode_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary(df: pd.DataFrame, paths: dict[str, str]) -> str:
    """Write a markdown summary of the sweep results."""
    n_rows = len(df)
    unique_u = len(df["U"].unique())
    unique_n = len(df["N"].unique())
    hi_modes = sorted(df["hi_mode"].unique())

    # Compute some stats
    total_hdm = df["hdm"].sum()
    mean_nid = df["nid"].mean()
    mean_tid = df["tid"].mean()
    mean_jne = df["jne"].mean()
    mean_ldm = df["ldm"].mean()

    # Success ratio stats
    success_by_u = df.groupby("U")["hdm"].apply(lambda x: (x == 0).mean())
    low_power_cells = df.groupby(["U", "N"])["hi_trigger_events"].sum()
    low_power_count = (low_power_cells < 100).sum()

    summary = f"""# AMC Sweep Results Summary

## Sweep Parameters

- **U range:** {df["U"].min():.2f} → {df["U"].max():.2f} ({unique_u} values)
- **N values:** {sorted(df["N"].unique())}
- **hi_mode:** {hi_modes}
- **Replicates per (U, N):** {int(len(df) / (unique_u * unique_n))}
- **Total rows:** {n_rows}

## Key Statistics

- **Total HDM (HI deadline misses):** {total_hdm}
- **Mean NiD (degraded mode entries):** {mean_nid:.2f}
- **Mean TiD (degraded time fraction):** {mean_tid:.4f}
- **Mean JNE (dropped LO jobs):** {mean_jne:.2f}
- **Mean LDM (late LO jobs):** {mean_ldm:.2f}

## Observations

1. **Success Ratio:** The fraction of task sets with no HI deadline misses
   (HDM=0) is {success_by_u.mean():.1%} on average. Values near 1.0 indicate
   that HI-criticality tasks are meeting their deadlines consistently.

2. **Statistical Power:** {low_power_count} out of {unique_u * unique_n} (U, N) cells
   have fewer than 100 HI-trigger events. Low-power regions produce unreliable
   NiD/TiD/JNE estimates and should be interpreted with caution.

3. **Utilisation Ceiling:** The analytic aggregate feasibility ceiling for
   CP=0.5, CF=1.5 is approximately 1/(0.5×1.5 + 0.5) ≈ 0.8. Task sets
   above this utilisation may be structurally infeasible.

## Generated Figures

{chr(10).join(f'- **{name}:** [{path}]({path})' for name, path in paths.items())}

## Recommendations

- For full sweeps, increase replicates in low-power cells or use longer
  simulation durations.
- Consider the `drs_independent` hi_mode to reduce individually-infeasible
  task sets at high utilisation.
- Phase 9 (AMC-RH/AMC-RA) would provide additional scheduler variants
  for comparison.
"""
    return summary


if __name__ == "__main__":
    paths = generate_plots()
    print("Generated figures:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
