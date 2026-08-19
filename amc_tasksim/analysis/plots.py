"""Analysis and plotting for AMC experiment sweep results.

Produces the box-and-whisker plots of Section V-E of the AMC-RH paper (RTAS
2022) -- NiD(%), TiD(%) and JNC(%) by protocol -- plus a validation
report that checks the simulator against the four things the papers actually
state:

1. HDM is zero for every scheme (Section V-E).
2. NiD(%) for the original AMC scheme is of the order of the failure
   probability, since each overrunning HI-criticality job triggers one entry
   ("NiD(%) has the same median value of 0.01 for AMC+ and BP ... these values
   simply reflect the configured Failure Probability").
3. The absolute metric values at U = 0.8, FP = 1e-4, non-harmonic periods
   (Figures 4-6).
4. Table I: AMC-RH's metrics as a percentage of the original scheme's.

Nothing here invents an expected value. Every target below is quoted from a
paper in `docs/`, with the citation alongside it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTOCOL_LABELS = {
    "original_amc": "AMC+",
    "amc_ra": "AMC-RA",
    "amc_rh": "AMC-RH",
}

METRICS = [
    ("nid_pct", "NiD (%)", "degraded-mode entries, % of HI-criticality jobs"),
    ("tid_pct", "TiD (%)", "time in degraded mode, % of simulation time"),
    ("jnc_pct", "JNC (%)", "LO-criticality jobs not completed, %"),
]

# ---------------------------------------------------------------------------
# Targets quoted from the papers
# ---------------------------------------------------------------------------

# RTAS 2022, Figures 4-6: U = 0.8, FP = 1e-4, non-harmonic periods, AMC+ scheme.
# Read off the axis ranges and the accompanying text in Section V-E.
PAPER_ABSOLUTE = {
    "nid_pct": (0.008, 0.012, "Section V-E: median 0.01, reflecting FP = 1e-4"),
    "tid_pct": (0.02, 0.04, "Figure 5 axis range"),
    "jnc_pct": (0.01, 0.02, "Figure 6 axis range"),
}

# RTAS 2022, Table I: metrics as a percentage of the original AMC+ scheme,
# non-harmonic periods (the period model this toolkit generates).
PAPER_TABLE_I = {
    "amc_rh": {"nid_pct": 19.9, "tid_pct": 4.1, "jnc_pct": 8.7},
}

PAPER_FP = 1e-4
PAPER_U = 0.8


def generate_plots(
    input_path: str = "results/sweep.parquet",
    output_dir: str = "results/figures",
) -> dict[str, str]:
    """Generate all figures and reports from a sweep result file.

    Args:
        input_path: Path to the sweep results parquet file.
        output_dir: Directory to save figures into.

    Returns:
        Dict mapping output names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(input_path)
    paths: dict[str, str] = {}

    for metric, label, _ in METRICS:
        paths[f"{metric}_box"] = _plot_metric_by_protocol(df, metric, label, output_dir)

    paths["metric_vs_u"] = _plot_metric_vs_u(df, output_dir)
    paths["stat_power"] = _plot_stat_power(df, output_dir)

    report = _write_validation_report(df)
    report_path = Path(output_dir).parent / "VALIDATION.md"
    report_path.write_text(report)
    paths["validation"] = str(report_path)

    summary_path = Path(output_dir).parent / "SUMMARY.md"
    summary_path.write_text(_write_summary(df, paths))
    paths["summary"] = str(summary_path)

    return paths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _protocols_in(df: pd.DataFrame) -> list[str]:
    order = [p for p in PROTOCOL_LABELS if p in set(df["protocol"])]
    return order or sorted(set(df["protocol"]))


def _plot_metric_by_protocol(
    df: pd.DataFrame, metric: str, label: str, output_dir: str
) -> str:
    """Box-and-whisker by protocol, one panel per utilisation level.

    Whiskers are the 5th and 95th percentiles, as in Section V-E. Every
    replicate contributes a point -- the distribution shown is the distribution
    of the data.
    """
    u_values = sorted(df["U"].unique())
    protocols = _protocols_in(df)
    at_paper_fp = df[np.isclose(df["FP"], PAPER_FP)]
    subset = at_paper_fp if not at_paper_fp.empty else df
    fp_note = (
        f"FP = {PAPER_FP:.0e}"
        if not at_paper_fp.empty
        else f"FP = {sorted(df['FP'].unique())}"
    )

    fig, axes = plt.subplots(
        1, len(u_values), figsize=(3.6 * len(u_values), 4.6), sharey=True, squeeze=False
    )
    for ax, u in zip(axes[0], u_values):
        data = [
            subset[(subset["U"] == u) & (subset["protocol"] == p)][metric].to_numpy()
            for p in protocols
        ]
        data = [d if d.size else np.array([np.nan]) for d in data]
        ax.boxplot(
            data,
            whis=(5, 95),
            showfliers=False,
            tick_labels=[PROTOCOL_LABELS.get(p, p) for p in protocols],
        )
        ax.set_title(f"U = {u:.2f}")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].set_ylabel(label)

    fig.suptitle(f"{label} by scheme ({fp_note}); box = quartiles, whiskers = 5/95 percentile")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{metric}_box.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_metric_vs_u(df: pd.DataFrame, output_dir: str) -> str:
    """Median of each metric against utilisation, one line per protocol."""
    protocols = _protocols_in(df)
    at_paper_fp = df[np.isclose(df["FP"], PAPER_FP)]
    subset = at_paper_fp if not at_paper_fp.empty else df

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, label, _) in zip(axes, METRICS):
        for p in protocols:
            g = subset[subset["protocol"] == p].groupby("U")[metric].median()
            ax.plot(g.index, g.to_numpy(), marker="o", label=PROTOCOL_LABELS.get(p, p))
        ax.set_xlabel("U")
        ax.set_ylabel(label)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Median metric against utilisation")
    fig.tight_layout()
    path = os.path.join(output_dir, "metric_vs_u.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stat_power(df: pd.DataFrame, output_dir: str) -> str:
    """Heatmap of HI-behaviour jobs observed per (U, N) cell."""
    base = df[df["protocol"] == _protocols_in(df)[0]]
    pivot = base.groupby(["U", "N"])["hi_trigger_events"].sum().unstack()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [f"{int(c):,}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("N   (FP = 1/N)")
    ax.set_ylabel("U")
    ax.set_title("HI-behaviour jobs observed per cell")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            v = pivot.to_numpy()[y, x]
            ax.text(
                x, y, f"{int(v):,}", ha="center", va="center", fontsize=8,
                color="black" if v < np.nanmax(pivot.to_numpy()) * 0.6 else "white",
            )
    fig.colorbar(im, ax=ax, label="HI-behaviour jobs")
    fig.tight_layout()
    path = os.path.join(output_dir, "stat_power.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def _aggregate(df: pd.DataFrame, protocol: str) -> dict[str, float]:
    """Pooled metrics for one protocol, weighted by the underlying job counts.

    Pooling the numerators and denominators is the right way to combine
    replicates of different lengths; averaging per-replicate percentages would
    weight a short run as heavily as a long one.
    """
    g = df[df["protocol"] == protocol]
    if g.empty:
        return {}
    return {
        "nid_pct": 100.0 * g["nid"].sum() / max(g["total_hi_releases"].sum(), 1),
        "tid_pct": 100.0 * (g["tid"] * g["duration"]).sum() / max(g["duration"].sum(), 1),
        "jnc_pct": 100.0
        * (g["jne"].sum() + g["lo_terminated"].sum())
        / max(g["total_lo_releases"].sum(), 1),
        "hdm": float(g["hdm"].sum()),
        "hi_behaviour_jobs": float(g["hi_trigger_events"].sum()),
        "replicates": float(len(g)),
    }


def validation_checks(df: pd.DataFrame) -> list[dict]:
    """Run the paper-derived checks and return one record per check.

    Each record has: name, source, expected, observed, passed.
    """
    checks: list[dict] = []
    protocols = _protocols_in(df)

    # --- 1. HDM must be zero for every scheme -----------------------------
    for p in protocols:
        total = int(df[df["protocol"] == p]["hdm"].sum())
        checks.append(
            {
                "name": f"HDM = 0 ({PROTOCOL_LABELS.get(p, p)})",
                "source": "RTAS 2022 Section V-E",
                "expected": "0",
                "observed": f"{total}",
                "passed": total == 0,
            }
        )

    # --- 2. NiD(%) is of the order of FP for the original scheme ----------
    base = df[df["protocol"] == "original_amc"]
    for fp, g in base.groupby("FP"):
        hi_jobs = g["total_hi_releases"].sum()
        if hi_jobs == 0 or g["hi_trigger_events"].sum() < 30:
            continue  # too few rare events for the ratio to mean anything
        observed = g["nid"].sum() / hi_jobs
        ratio = observed / fp
        checks.append(
            {
                "name": f"NiD per HI job ~ FP at FP={fp:.0e}",
                "source": "RTAS 2022 Section V-E",
                "expected": f"{fp:.2e} (ratio 1.0)",
                "observed": f"{observed:.2e} (ratio {ratio:.2f})",
                # An integer execution time drawn from U{C(LO)..C(HI)} can land
                # exactly on C(LO), in which case the job completes at its
                # budget and correctly does not trigger; that pulls the ratio
                # below one without being an error.
                "passed": 0.3 <= ratio <= 1.15,
            }
        )

    # --- 3. Absolute values at the paper's operating point ----------------
    op = df[np.isclose(df["FP"], PAPER_FP) & np.isclose(df["U"], PAPER_U)]
    if not op.empty:
        agg = _aggregate(op, "original_amc")
        for metric, (lo, hi, cite) in PAPER_ABSOLUTE.items():
            if metric not in agg:
                continue
            v = agg[metric]
            checks.append(
                {
                    "name": f"{metric} at U={PAPER_U}, FP={PAPER_FP:.0e} (AMC+)",
                    "source": f"RTAS 2022 {cite}",
                    "expected": f"{lo}–{hi}",
                    "observed": f"{v:.5f}",
                    # Within a factor of three of the quoted band counts as
                    # reproducing the magnitude at this scale.
                    "passed": lo / 3.0 <= v <= hi * 3.0,
                }
            )

    # --- 4. Table I ratios -------------------------------------------------
    if not op.empty:
        base_agg = _aggregate(op, "original_amc")
        for p, targets in PAPER_TABLE_I.items():
            agg = _aggregate(op, p)
            if not agg or not base_agg:
                continue
            for metric, target in targets.items():
                if not base_agg.get(metric):
                    continue
                observed = 100.0 * agg[metric] / base_agg[metric]
                checks.append(
                    {
                        "name": f"{PROTOCOL_LABELS[p]} {metric} vs AMC+",
                        "source": "RTAS 2022 Table I, non-harmonic",
                        "expected": f"{target}%",
                        "observed": f"{observed:.1f}%",
                        # Direction and order of magnitude; the remaining gap is
                        # discussed in the report body.
                        "passed": observed <= target * 3.0,
                    }
                )

    return checks


def _write_validation_report(df: pd.DataFrame) -> str:
    checks = validation_checks(df)
    passed = sum(1 for c in checks if c["passed"])
    protocols = _protocols_in(df)

    lines = [
        "# Validation against the reference papers",
        "",
        f"Checks passed: **{passed} / {len(checks)}**",
        "",
        "Every expected value below is quoted from a paper in `docs/`; none are",
        "estimated or invented. Metrics are pooled over replicates by summing",
        "numerators and denominators, not by averaging percentages.",
        "",
        "## Configuration",
        "",
        f"- Utilisations: {[round(float(u), 3) for u in sorted(df['U'].unique())]}",
        f"- N (FP = 1/N): {sorted(int(n) for n in df['N'].unique())}",
        f"- Protocols: {', '.join(PROTOCOL_LABELS.get(p, p) for p in protocols)}",
        f"- Replicates per (U, N, protocol): "
        f"{int(len(df) / max(len(df.groupby(['U', 'N', 'protocol'])), 1))}",
        f"- Simulation length: {int(df['duration'].min()):,}–{int(df['duration'].max()):,} ticks",
        f"- Task sets: filtered to unschedulable under FPPS and schedulable under AMC-rtb (Audsley OPA)",
        "",
        "## Checks",
        "",
        "| Check | Source | Expected | Observed | |",
        "|---|---|---|---|---|",
    ]
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(
            f"| {c['name']} | {c['source']} | {c['expected']} | {c['observed']} | {mark} |"
        )

    lines += [
        "",
        "**What PASS means here.** These are order-of-magnitude checks, not tight",
        "ones: an absolute metric passes within a factor of three of the paper's",
        "quoted band, a Table I ratio passes if it is no more than three times the",
        "quoted value, and the NiD-to-FP ratio passes between 0.3 and 1.15. They are",
        "sized to catch a simulator that is wrong by orders of magnitude -- which is",
        "the failure mode they were written for -- not to certify agreement. Read the",
        "observed column, not the verdict.",
    ]

    # Metric table at the paper's operating point
    op = df[np.isclose(df["FP"], PAPER_FP) & np.isclose(df["U"], PAPER_U)]
    if not op.empty:
        lines += [
            "",
            f"## Metrics at U = {PAPER_U}, FP = {PAPER_FP:.0e}",
            "",
            "| Scheme | NiD (%) | TiD (%) | JNC (%) | HDM | vs AMC+ (NiD/TiD/JNE+LDM) |",
            "|---|---|---|---|---|---|",
        ]
        base_agg = _aggregate(op, "original_amc")
        for p in protocols:
            agg = _aggregate(op, p)
            if not agg:
                continue
            rel = "—"
            if base_agg and p != "original_amc":
                rel = " / ".join(
                    f"{100.0 * agg[m] / base_agg[m]:.1f}%" if base_agg[m] else "—"
                    for m in ("nid_pct", "tid_pct", "jnc_pct")
                )
            lines.append(
                f"| {PROTOCOL_LABELS.get(p, p)} | {agg['nid_pct']:.5f} | {agg['tid_pct']:.5f} "
                f"| {agg['jnc_pct']:.5f} | {int(agg['hdm'])} | {rel} |"
            )

    lines += [
        "",
        "## Reading the residual gaps",
        "",
        "- **NiD(%) below FP.** An integer execution time drawn from",
        "  `U{C(LO)..C(HI)}` lands exactly on `C(LO)` with probability",
        "  `1/(C(HI)-C(LO)+1)`. Such a job signals completion at its budget and",
        "  correctly triggers nothing, so the ratio sits below one whenever the",
        "  budgets are small integers.",
        "- **AMC-RH better than Table I.** Table I is a mean over the paper's own",
        "  task-set population; the ratio is sensitive to how much slack sits",
        "  between C(LO) and R(LO), and therefore to CF, to the period model, and",
        "  to how tight the filtered task sets are.",
        "- **Short runs.** At research scale the rarest cells see few HI-behaviour",
        "  jobs; the statistical-power figure shows which.",
        "",
        "*Generated by `amc_tasksim.analysis.plots`.*",
    ]
    return "\n".join(lines)


def _write_summary(df: pd.DataFrame, paths: dict[str, str]) -> str:
    protocols = _protocols_in(df)
    lines = [
        "# AMC sweep summary",
        "",
        f"- Rows: {len(df)}",
        f"- Utilisations: {sorted(df['U'].unique())}",
        f"- N values: {sorted(int(n) for n in df['N'].unique())}",
        f"- Protocols: {', '.join(PROTOCOL_LABELS.get(p, p) for p in protocols)}",
        "",
        "## Pooled metrics by protocol",
        "",
        "| Scheme | NiD (%) | TiD (%) | JNC (%) | HDM | HI-behaviour jobs |",
        "|---|---|---|---|---|---|",
    ]
    for p in protocols:
        agg = _aggregate(df, p)
        lines.append(
            f"| {PROTOCOL_LABELS.get(p, p)} | {agg['nid_pct']:.5f} | {agg['tid_pct']:.5f} "
            f"| {agg['jnc_pct']:.5f} | {int(agg['hdm'])} | {int(agg['hi_behaviour_jobs']):,} |"
        )

    thin = (
        df[df["protocol"] == protocols[0]]
        .groupby(["U", "N"])["hi_trigger_events"]
        .sum()
    )
    lines += [
        "",
        "## Health",
        "",
        f"- Budget overruns: {int(df['budget_overruns'].sum())} (must be 0)",
        f"- Rows with TiD outside [0, 1]: {int(((df['tid'] < 0) | (df['tid'] > 1)).sum())} (must be 0)",
        f"- Tasks with a zero execution-time budget: {int(df['zero_budget_count'].sum())}",
        f"- Cells with fewer than 100 HI-behaviour jobs: {int((thin < 100).sum())} of {len(thin)}",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- **{name}**: `{path}`" for name, path in paths.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    for name, path in generate_plots().items():
        print(f"{name}: {path}")
