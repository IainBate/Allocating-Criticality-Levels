# amc-tasksim User Guide

## What This Is

`amc-tasksim` is a research toolkit for simulating **Adaptive Mixed-Criticality (AMC)** fixed-priority scheduling on a single core. It generates large-scale synthetic task sets, simulates them under different mode-change protocols, and produces statistics on degraded-mode behaviour (NiD, TiD, JNE, LDM, HDM).

## Prerequisites

- **Python 3.11+**
- **uv** — the project uses uv for dependency management. Install it if you don't have it:

  ```bash
  brew install uv   # macOS
  # or follow instructions at https://docs.astral.sh/uv/
  ```

## Quick Start

```bash
# 1. Clone or copy the repo
git clone https://github.com/IainBate/Allocating-Criticality-Levels.git
cd Allocating-Criticality-Levels

# 2. Install dependencies
uv sync --all-extras

# 3. Run tests
uv run pytest

# 4. Run a quick smoke-test sweep (20 replicates, ~seconds)
uv run amc_tasksim --quick

# 5. Run a full sweep (1000 replicates per cell, ~minutes to hours)
uv run amc_tasksim
```

## CLI Reference

```
uv run amc_tasksim [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--quick` | off | Use 20 replicates instead of 1000 (smoke test) |
| `--hi-mode` | `fixed_ratio` | HI-criticality utilisation mode: `fixed_ratio` or `drs_independent` |
| `--n-replicates N` | 1000 (20 if `--quick`) | Override replicate count |
| `--duration N` | 1000000 | Simulation duration in ticks |
| `--output PATH` | `results/sweep.parquet` | Output parquet file path |
| `--protocol P` | `original_amc` | Mode-change protocol: `original_amc`, `amc_rh`, or `amc_ra` |
| `--n-workers N` | 1 | Number of parallel workers (sequential for now) |
| `--n-values N1 N2 ...` | `10 100 1000 10000 100000` | Failure-probability sweep values (FP = 1/N) |
| `--U-range START STOP STEP` | `0.05 0.95 0.05` | Utilisation sweep range |

### Example: Full sweep with all three protocols

```bash
# Original AMC
uv run amc_tasksim --output results/sweep_amc.parquet

# AMC-RH
uv run amc_tasksim --protocol amc_rh --output results/sweep_amc_rh.parquet

# AMC-RA
uv run amc_tasksim --protocol amc_ra --output results/sweep_amc_ra.parquet
```

## Running the Analysis

After a sweep completes, generate plots and a summary:

```python
uv run python -c "
from amc_tasksim.analysis.plots import generate_plots
paths = generate_plots('results/sweep.parquet')
for name, path in paths.items():
    print(f'{name}: {path}')
"
```

This produces:

| Plot | File | What it shows |
|---|---|---|
| NiD boxplot | `results/figures/nid_boxplot.png` | Times degraded mode entered (by U, faceted by N) |
| TiD boxplot | `results/figures/tid_boxplot.png` | Fraction of time in degraded mode |
| JNE+LDM boxplot | `results/figures/jne_ldm_boxplot.png` | Dropped + late LO jobs |
| Success ratio heatmap | `results/figures/success_ratio_heatmap.png` | Fraction of task sets with zero HI deadline misses |
| Statistical power heatmap | `results/figures/stat_power_heatmap.png` | Total HI-trigger events per (U, N) cell |
| SUMMARY.md | `results/SUMMARY.md` | Written observations and recommendations |

## Understanding the Output

The output parquet file has one row per `(U, N, replicate)` with these columns:

| Column | Meaning |
|---|---|
| `U` | Target utilisation |
| `N` | Inverse failure probability (FP = 1/N) |
| `FP` | Failure probability |
| `hi_mode` | Generation mode (`fixed_ratio` or `drs_independent`) |
| `protocol` | Mode-change protocol used |
| `replicate_index` | Replicate number within this (U, N) cell |
| `nid` | Number of degraded-mode entries |
| `tid` | Fraction of time in degraded mode |
| `jne` | LO jobs dropped in degraded mode |
| `ldm` | LO jobs that missed deadline (late) |
| `hdm` | HI deadline misses (should be 0 for schedulable sets) |
| `hi_trigger_events` | HI-criticality overrun events observed |
| `total_hi_releases` | Total HI-criticality job releases |
| `individually_infeasible` | Count of HI tasks with C_hi > T |
| `aggregate_hi_utilisation` | Sum of C_hi/T for HI tasks |
| `schedulable_amc_rtb` | Whether AMC-rtb says the task set is schedulable |
| `nontrivial_amc` | Whether the task set benefits from AMC over plain FP |

## Protocol Comparison

Three mode-change protocols are available:

| Protocol | Entry Trigger | Exit Trigger |
|---|---|---|
| **OriginalAMC** | HI job executes C_i(LO) without completing | Idle instant |
| **AMC-RH** | HI job reaches R_i(LO) from busy-period start | No active HI job past R_i(LO) |
| **AMC-RA** | HI job reaches R_i(LO) from busy-period start | Idle instant |

**AMC-RH** exits degraded mode earlier than the other two (when the active HI-criticality job no longer needs protection), which typically reduces JNE and TiD. **AMC-RA** uses the same exit as OriginalAMC but enters later (at R_i(LO) rather than C_i(LO)), also reducing degradation impact.

## Generating Task Sets Programmatically

```python
from amc_tasksim.generation.taskset import generate_taskset, generate_ensemble
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import simulate

# Single task set
ts = generate_taskset(
    n=20,
    CP=0.5,       # 50% HI-criticality
    U=0.5,        # target utilisation
    CF=1.5,       # C_hi = 1.5 * C_lo
    N=10000,      # FP = 1/10000
    hi_mode="fixed_ratio",   # or "drs_independent"
    period_range=(100, 10000),
    rng_seed=42,
)

# Assign priorities (Deadline Monotonic)
assign_deadline_monotonic(ts)

# Simulate
result = simulate(ts, duration=10**6, seed=42, fp=1e-4)

print(f"NiD={result.nid}, TiD={result.tid:.4f}, "
      f"JNE={result.jne}, LDM={result.ldm}, HDM={result.hdm}")
```

## Key Design Decisions

1. **Event-driven simulation** — the engine jumps between events (releases, completions) rather than iterating tick by tick. This makes 10^6-tick simulations tractable.

2. **Per-job execution time** — each job draws its execution time once at release from `uniform(BCET, C_lo)` (LO mode) or `uniform(C_lo, C_hi)` (HI mode). The time is not resampled each tick.

3. **Deterministic seeding** — the same `(U, n_replicates, rng_seed)` always produces the same ensemble. Seeds are derived from `(U, replicate_index)`.

4. **DRS-independent mode** — generates HI-criticality utilisation independently of LO, guaranteeing no individually-infeasible HI tasks (C_hi ≤ T). Use this mode when you want to avoid the structural infeasibility artefact that the fixed-ratio mode produces at high utilisation.

5. **Statistical power threshold** — the sweep warns when total HI-trigger events fall below 100 per cell. Low-power cells produce unreliable NiD/TiD/JNE estimates.

6. **Default FP** — the simulator defaults to `fp=1e-4` (FP = 1/10000) when not specified. This matches the spec's convention of sweeping N over log-spaced values.

## Directory Layout

```
amc-tasksim/
  generation/       DRS algorithm + task-set generator
  scheduling/       Priority assignment + AMC-rtb analysis
  simulation/       Event-driven engine + mode-change protocols
  experiments/      Sweep orchestration
  analysis/         Plotting + summary generation
tests/              pytest suite mirroring package structure
docs/               Reference PDFs (DRS/RTSS 2020; AMC-RH/RTAS 2022; AMC/RTNS 2022)
results/            Experiment output (gitignored)
```

## Troubleshooting

**"DRS did not converge" warnings** — rare, only with tight asymmetric constraints. The algorithm retries up to 50 times with fresh random samples. If it persists, check that `U` is within `[sum(umin), sum(umax)]`.

**"Only 0 HI-trigger events" warnings** — expected for low-U task sets where C_lo values are small (often 0 or 1 after rounding), leaving no room for HI overrun. This doesn't indicate a bug; it means that (U, N) combination is not interesting for degraded-mode analysis.

**ImportError on parquet** — install pyarrow: `uv pip install pyarrow`.

**Tests fail after updating code** — run `uv run pytest` to verify. The full suite is 157 tests covering DRS correctness, task-set generation, AMC-rtb (with the paper's Appendix A worked example), simulator validation, and protocol behaviour.
