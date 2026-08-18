# amc-tasksim User Guide

## What This Is

`amc-tasksim` generates synthetic **Adaptive Mixed-Criticality (AMC)** task sets, simulates them under three mode-change protocols on a single core, and reports the five metrics of Section V-A of the AMC-RH paper (RTAS 2022): HDM, JNE, LDM, TiD and NiD.

The reference papers are in `docs/`:

1. **DRS** — Griffin, Bate & Davis, *Generating Utilization Vectors for the Evaluation of Real-Time Scheduling Algorithms*, RTSS 2020
2. **AMC-RH** — Bate, Burns & Davis, *Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling*, RTAS 2022
3. **C-AMC** — Davis, Burns & Bate, *Compensating Adaptive Mixed Criticality Scheduling*, RTNS 2022

## Prerequisites

- **Python 3.11+**
- **uv** for dependency management: `brew install uv` on macOS, or see https://docs.astral.sh/uv/

## Quick Start

```bash
uv sync --all-extras          # install
uv run pytest                 # 186 tests

# A sweep you can iterate on: minutes, not hours
uv run python -m amc_tasksim --scale debug --plots

# The paper's configuration: run this on a big machine
uv run python -m amc_tasksim --scale paper --output results/sweep_paper.parquet --plots
```

## Scale presets

The sweep is sized by `--scale`. Everything it sets can be overridden individually.

| | `debug` | `paper` |
|---|---|---|
| Qualifying task sets per U | 20 | 500 |
| Simulation length | 200 jobs of the longest-period task | 10⁶ jobs |
| Utilisations | 0.6, 0.7, 0.8, 0.9 | 0.6, 0.7, 0.8, 0.9 |
| N (FP = 1/N) | 100, 1000, 10000 | 100, 1000, 10000, 100000 |
| Wall clock | ~5 minutes | hours to days |

`paper` matches RTAS 2022 Section V-D: 500 task sets and a run long enough for 10⁶ jobs of the longest-period task. `debug` shortens both so the framework can be exercised end to end; the metrics land in the right range but the rarest cells are noisy, which the statistical-power figure makes visible.

## CLI Reference

```
uv run python -m amc_tasksim [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--scale` | `debug` | Sizing preset: `debug` or `paper` |
| `--n-replicates N` | from `--scale` | Qualifying task sets per utilisation level |
| `--duration-jobs N` | from `--scale` | Run length in jobs of the longest-period task |
| `--U-values ...` | from `--scale` | Utilisation levels |
| `--n-values ...` | from `--scale` | Failure-probability levels (FP = 1/N) |
| `--protocols ...` | all three | `original_amc`, `amc_ra`, `amc_rh` |
| `--hi-mode` | `drs_independent` | `drs_independent` follows the papers; `fixed_ratio` is a legacy per-task multiplier |
| `--tasks N` | 20 | Tasks per task set |
| `--cp F` | 0.5 | Criticality proportion |
| `--cf F` | 2.0 | Criticality factor |
| `--seed N` | 42 | Base random seed |
| `--output PATH` | `results/sweep.parquet` | Output parquet file |
| `--plots` | off | Generate figures and the validation report afterwards |
| `--clean` | off | Remove results, figures and caches first |

All protocols run on the **same task sets with the same seeds**, so a single sweep supports the like-for-like comparison the papers make. Execution times are drawn for every release, including one about to be abandoned in degraded mode, so the release sequence is identical under every protocol.

## The task-set population

The papers do not evaluate arbitrary task sets. Section V-C of the AMC-RH paper requires each task set to be

- **unschedulable** under exact fixed-priority analysis ignoring criticality (every task at `max(C(LO), C(HI))`) — otherwise it does not need a mixed-criticality scheme at all, and
- **schedulable** under AMC-rtb.

The sweep generates candidates until it has the requested number of qualifying task sets, and reports the yield. Around U = 0.8 most candidates qualify; by U = 0.9 roughly a quarter do; below about U = 0.55 almost none do, because plain FPPS already copes.

Priorities are assigned by **Audsley's Optimal Priority Assignment**, as both papers do. Deadline monotonic is not optimal for AMC-rtb — at U = 0.85 it rejects task sets that OPA schedules — and using it would silently change which task sets the experiment ever sees.

## Task-set generation

Default (`--hi-mode drs_independent`), following RTAS 2022 Section V-C:

1. Periods log-uniform over a factor of 100 (10 ms – 1 s at a 0.1 ms tick, so 100–10 000 ticks); deadlines implicit.
2. `U_i(HI)` drawn by DRS for the HI-criticality tasks, summing to `CP · CF · U`.
3. `U_i(LO)` drawn by DRS for **all** tasks, summing to `U`, with each HI-criticality task capped at its own `U_i(HI)` — which is what guarantees `C_i(LO) ≤ C_i(HI)`.
4. `C_i(x) = U_i(x) · T_i`; BCET uniform in 80–100% of `C_i(LO)`.

**CF is a ratio of aggregate utilisations, not a per-task multiplier.** Individual tasks end up with a spread of `C(HI)/C(LO)` ratios (typically 1× to 20×), which is the source of most of the variance in the box plots. The legacy `fixed_ratio` mode sets `C_i(HI) = CF · C_i(LO)` for every task and is kept only for comparison.

Utilisation lower bounds are set to `1/T_i` where the budget allows, so every execution time rounds to at least one tick. Below about U = 0.2 there is not enough budget for that and some tasks get a zero budget; `zero_budget_count` records how many.

## Output columns

One row per `(U, N, protocol, replicate)`:

| Column | Meaning |
|---|---|
| `U`, `N`, `FP` | Target utilisation; failure-probability level; FP = 1/N |
| `protocol` | `original_amc`, `amc_ra` or `amc_rh` |
| `replicate_index` | Task set index within this utilisation level |
| `duration` | Simulation length in ticks for this task set |
| `nid` | Times degraded mode was entered |
| `tid` | Fraction of the simulation spent in degraded mode |
| `jne` | LO-criticality jobs abandoned on release in degraded mode |
| `ldm` | LO-criticality jobs that executed but missed their deadline |
| `hdm` | HI-criticality deadline misses (zero for these task sets) |
| `nid_pct` | **NiD(%)** — `nid` as a % of HI-criticality jobs |
| `tid_pct` | **TiD(%)** — `tid` as a % of simulation time |
| `jne_ldm_pct` | **JNE(%)+LDM(%)** — as a % of LO-criticality jobs; the paper's headline metric |
| `hi_trigger_events` | Jobs selected to exhibit HI-criticality behaviour |
| `total_hi_releases`, `total_lo_releases` | Metric denominators |
| `budget_overruns` | Defensive check; must be 0 |
| `zero_budget_count` | Tasks whose C(LO) rounded to zero |
| `aggregate_hi_utilisation` | Σ C(HI)/T over HI-criticality tasks |

When aggregating across replicates, **pool numerators and denominators** rather than averaging the percentage columns — runs have different lengths, so averaging percentages weights a short run as heavily as a long one. `amc_tasksim.analysis.plots._aggregate` does this.

## Protocols

| Protocol | Enter degraded mode | Exit degraded mode |
|---|---|---|
| **OriginalAMC** (AMC+) | A HI-criticality job has executed for `C_i(LO)` without completing | Idle instant |
| **AMC-RA** | An active HI-criticality job reaches `R_i(LO)` after the start of its priority level-i busy period | Idle instant |
| **AMC-RH** | As AMC-RA | A HI-criticality job completes and no active HI-criticality job is past its own trigger |

AMC-RA and AMC-RH enter degraded mode no earlier than AMC+, because `R_i(LO) ≥ C_i(LO)`; AMC-RH leaves no later than AMC-RA. AMC-RH typically enters *slightly more often* than AMC-RA, because AMC-RA's wait for an idle instant merges what would otherwise be two separate intervals.

Busy-period start times are tracked with the O(1) rule of Appendix B: a task released at the head of the run-queue starts its own busy period, otherwise it inherits from the job immediately ahead, with simultaneous releases processed in priority order.

## Analysis

```bash
uv run python -c "
from amc_tasksim.analysis.plots import generate_plots
for name, path in generate_plots('results/sweep.parquet').items():
    print(f'{name}: {path}')
"
```

| Output | What it shows |
|---|---|
| `results/figures/nid_pct_box.png` | NiD(%) by scheme, one panel per U — the paper's Figures 1/4/7/10 |
| `results/figures/tid_pct_box.png` | TiD(%) by scheme — Figures 2/5/8/11 |
| `results/figures/jne_ldm_pct_box.png` | JNE(%)+LDM(%) by scheme — Figures 3/6/9/12 |
| `results/figures/metric_vs_u.png` | Median of each metric against utilisation |
| `results/figures/stat_power.png` | HI-behaviour jobs observed per (U, N) cell |
| `results/VALIDATION.md` | Checks against values quoted from the papers |
| `results/SUMMARY.md` | Pooled metrics and health checks |

`VALIDATION.md` checks only things the papers actually state: HDM = 0; NiD(%) of the order of FP for AMC+; the absolute metrics at U = 0.8, FP = 10⁻⁴; and the Table I ratios. The pass criteria are deliberately wide (a factor of three) — they are there to catch a simulator that is wrong by orders of magnitude, not to certify agreement. Read the observed values.

## Programmatic use

```python
from amc_tasksim.generation.taskset import generate_taskset
from amc_tasksim.scheduling.amc_rtb import amc_rtb, is_nontrivial_amc_taskset
from amc_tasksim.scheduling.priority import assign_audsley_opa
from amc_tasksim.simulation.engine import simulate, OriginalAMC, AMC_RA, AMC_RH

ts = generate_taskset(n=20, CP=0.5, U=0.8, CF=2.0, rng_seed=1)
if is_nontrivial_amc_taskset(ts):           # assigns OPA priorities
    r_lo = amc_rtb(ts).r_lo
    for protocol in (OriginalAMC(), AMC_RA(r_lo), AMC_RH(r_lo)):
        result = simulate(ts, duration=200 * max(ts.T), seed=1,
                          mode_protocol=protocol, fp=1e-4)
        print(f"{protocol.name:<13} NiD={result.nid_pct:.5f}%  "
              f"TiD={result.tid_pct:.5f}%  JNE+LDM={result.jne_ldm_pct:.5f}%")
```

```
original_amc  NiD=0.02341%  TiD=0.11941%  JNE+LDM=0.10965%
amc_ra        NiD=0.00000%  TiD=0.00000%  JNE+LDM=0.00000%
amc_rh        NiD=0.00000%  TiD=0.00000%  JNE+LDM=0.00000%
```

Not every task set qualifies — `is_nontrivial_amc_taskset` returns False for a task set that plain FPPS already schedules, or that AMC-rtb cannot. And on a single short run the response-time protocols often never degrade at all, which is the effect the papers are measuring; the sweep averages over many task sets to quantify it.

### Reproducing a scripted scenario

```python
trace = []
simulate(ts, duration=14, seed=0, mode_protocol=OriginalAMC(), fp=1.0,
         release_offsets=[0, 6, 0],   # per-task phase of the first release
         exec_time_mode="wcet",       # every job executes exactly its budget
         trace=trace)                 # (time, event, task_id) tuples
```

This is how the Appendix A / Figure 13 schedule is checked in the test suite.

## Troubleshooting

**"DRS did not converge"** — DRS raises rather than returning a biased sample. The fold is hardest when `sum(umax)/U` is near 2, where the constraints simplex and the standard simplex have equal volume and the duality optimisation cannot help; there the run is long enough to exhaust 64-bit precision (DRS paper, Section III-D). The generator's own constraint shapes are far from that regime and take about 0.1 ms per task set.

**"only N HI-behaviour jobs in this cell"** — the run is too short for the failure probability. Raise `--duration-jobs` or `--n-replicates`; a cell with fewer than ~100 HI-behaviour jobs cannot say much about NiD, TiD or JNE.

**No qualifying task sets at low U** — expected. Below about U = 0.55, plain FPPS already schedules nearly everything, so nothing meets the paper's non-trivial criterion.

**ImportError on parquet** — `uv pip install pyarrow`.
