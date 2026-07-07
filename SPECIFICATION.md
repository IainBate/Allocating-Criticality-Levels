# AMC Task-Set Generation & Simulation

A research toolkit for generating large-scale synthetic AMC task sets,
simulating them under different mode-change protocols, and measuring
degraded-mode behaviour (NiD, TiD, JNE, LDM, HDM).

Built on three papers:

1. **DRS** — "Generating Utilization Vectors for the Evaluation of
   Real-Time Scheduling Algorithms", Baruah et al., RTSS 2020
2. **AMC-RH** — "Analysis-Runtime Co-design for Adaptive Mixed-Criticality
   Scheduling", Bate et al., RTAS 2022
3. **AMC** — "Compensating Adaptive Mixed-Criticality Scheduling",
   Bate et al., RTNS 2022

## Shared Parameters

| Parameter | Default | Notes |
|---|---|---|
| HI/LO ratio (CF) | `1.5` | C_hi = CF × C_lo in `fixed_ratio` mode |
| Failure probability (FP) | `1/N`, N ∈ {10, 100, 1000, 10000, 100000} | 1 in N jobs exhibits HI behaviour |
| Task-set cardinality (n) | `20` | Paper convention |
| Criticality proportion (CP) | `0.5` | Fraction of tasks that are HI |
| Utilisation sweep U(LO) | `0.05 → 0.95`, step `0.05` | Analytic aggregate-feasibility ceiling ≈ `1 / (CP × CF + (1 − CP))` ≈ `0.8` for CP=0.5, CF=1.5 |
| Replicates per (U, N) | `1000` | `--quick` uses `20` |
| Simulation duration | `10⁶` ticks | Engine is event-driven internally |
| Periods | Log-uniform, ratio `100` | Integer ticks; semi-harmonic left as a later option |
| Deadlines | `D_i = T_i` (implicit) | Simplest starting point |
| BCET | Uniform `80–100%` of C_lo | Can be disabled (BCET = C_lo) |
| Priority assignment | Deadline Monotonic | Optimal for implicit deadlines under single-criticality FPPS; Audsley OPA under AMC-rtb is a later extension |
| Mode-change protocol | Original AMC | Pluggable; AMC-RH and AMC-RA available |
| Seeding | Deterministic from `(U, replicate_index)` | Full reproducibility |

## How to Use

Place the three reference PDFs in `docs/` before starting. The build
process reads them for exact equations.

Run the sweep from the command line:

```bash
uv run amc_tasksim --quick                    # smoke test
uv run amc_tasksim --protocol amc_rh          # full sweep, AMC-RH
uv run amc_tasksim --protocol amc_ra          # full sweep, AMC-RA
```

Or use the library programmatically (see USER_GUIDE.md for examples).

---

## Phase 1 — Project Scaffold

Set up a Python research project called `amc-tasksim`.

- Python 3.11+, dependency management via **uv** (`pyproject.toml`, hatchling build)
- Core dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `joblib`
- Dev dependencies: `pytest`, `pytest-cov`
- Directory layout:

  ```
  amc_tasksim/generation/    DRS algorithm + task-set generator
  amc_tasksim/scheduling/    Priority assignment + analytic schedulability tests
  amc_tasksim/simulation/    Event-driven simulator core
  amc_tasksim/experiments/   Sweep orchestration + result storage
  amc_tasksim/analysis/      Plotting + summary statistics
  tests/                     pytest suite mirroring package structure
  docs/                      Reference PDFs (DRS/RTSS 2020; AMC-RH/RTAS 2022; AMC/RTNS 2022)
  results/                   Experiment output (gitignored)
  ```

- README.md summarising the project goal
- Initialise git

**Do not implement any algorithm logic yet — this is project setup only.**

**Definition of done:** repo exists, `uv run pytest` runs cleanly (zero tests collected), tree matches the layout.

---

## Phase 2 — DRS Core Algorithm

Implement the Dirichlet-Rescale (DRS) algorithm in `amc_tasksim/generation/drs.py`.

Read `docs/DRSRTSS2020.pdf`, Section III ("Dirichlet-Rescale Algorithm"), for the exact
sub-functions: RMSS (rescale-matrix-to-standard-simplex), CtS (constraints-to-simplex),
Rescale, and SmallestSimplexRescale (SSR).

### Signature

```python
drs(n: int, U: float, umax: np.ndarray | None = None,
    umin: np.ndarray | None = None, epsilon: float = 1e-4,
    rng: np.random.Generator | None = None) -> np.ndarray
```

- Default `umax = ones(n)`, `umin = zeros(n)`
- Validate: `sum(umin) ≤ U ≤ sum(umax)`, and elementwise `umax ≥ umin`. Raise `ValueError` otherwise.
- Handle the `umin ≠ 0` canonical-form reduction: transform to `u'max = umax − umin`,
  `U' = U − sum(umin)`, solve for `x'`, then `x = x' + umin`.
- If floating-point divergence between `sum(output)` and `U` exceeds `epsilon`, retry
  with a fresh random sample. Cap retries at a sane limit (50) and warn if exceeded.
- Pass `rng` through the entire call chain for reproducibility.

### Reference: UUnifast

Also implement `uunifast(n, U, rng)` as a reference for the unconstrained case (eq. 6,
Section II-C of the DRS paper): generate `n−1` uniform random numbers in `[0, U]`, sort
them, include `0` and `U` as endpoints, take the gaps.

### Tests (`tests/generation/test_drs.py`)

1. **Sum accuracy** — output always sums to U within epsilon, for a range of `n`, `U`,
   and both symmetric and asymmetric `umax`/`umin`.
2. **Constraint respect** — every `umin_i ≤ output_i ≤ umax_i` for 100 samples per
   configuration.
3. **DRS ≡ UUnifast equivalence** — draw 100,000 samples from both `drs(n, U)` and
   `uunifast(n, U)`, confirm via two-sample Kolmogorov–Smirnov test on component
   marginals that the distributions are not statistically distinguishable (p > 0.05).
4. **Performance** — for `n=50, U=0.5, umax=UUnifast(50,1)`, mean call time stays
   under a few hundred milliseconds over 100 calls.
5. **Edge cases** — `umin` at boundary (U = sum(umin) returns umin), `umax` at boundary
   (U = sum(umax) returns umax), `n=1` returns `[U]`.
6. **Invalid inputs** — `ValueError` raised for `umax < umin`, `U < sum(umin)`,
   `U > sum(umax)`.

**Definition of done:** all DRS tests pass, including the DRS-versus-UUnifast equivalence check.

---

## Phase 3 — Task-Set Generator

Implement `amc_tasksim/generation/taskset.py`, building on `drs()` from Phase 2.

### TaskSet Dataclass

```
n: int
criticality: list[str]  # "HI" or "LO" per task
T: list[int]            # periods
D: list[int]            # deadlines (= T for implicit)
C_lo: list[int]         # execution time budgets in normal mode
C_hi: list[int]         # execution time budgets in degraded mode
BCET: list[int]         # best-case execution times
priority: list[int]     # filled in Phase 4
metadata: dict          # generation params, seed, target U
```

Derived diagnostics:
- `individually_infeasible_count` — HI tasks where CF × C_lo > T
- `individually_infeasible_indices` — which tasks
- `aggregate_hi_utilisation` — sum of C_hi/T for HI tasks

### generate_taskset()

```python
generate_taskset(n=20, CP=0.5, U=0.5, CF=1.5, N=10000,
    hi_mode="fixed_ratio", period_range=(100, 10000),
    bcet_fraction_range=(0.8, 1.0), rng_seed=None) -> TaskSet
```

**Steps:**

1. `n_hi = round(CP × n)`, `n_lo = n − n_hi`. First `n_hi` tasks are HI-criticality.
2. `u = drs(n, U, umax=ones(n), umin=zeros(n))` — per-task utilisation.
3. **Periods** — draw log-uniform in `period_range`, round to nearest integer tick,
   enforce `T_i ≥ 1`. `D_i = T_i`.
4. `C_lo = round(u × T)`. `BCET = round(C_lo × uniform(bcet_fraction_range))`,
   clamped to ≥ 1.
5. **C_hi** depends on `hi_mode`:

   | Mode | Method |
   |---|---|
   | `fixed_ratio` (default) | `C_hi = min(round(CF × C_lo), T)` for HI tasks. Flag + count any task where `CF × C_lo / T > 1` as individually infeasible. Report the fraction of task sets with at least one infeasible task — do not silently discard. |
   | `drs_independent` | Two-level DRS: draw `u(HI)` summing to `CF × CP × U` via DRS, then use `u(HI)` as the per-task max constraint for HI-criticality LO utilisation via a second DRS call. This guarantees `C_lo ≤ C_hi` by construction. Remaining budget distributed among LO-criticality tasks. |

6. Report aggregate HI utilisation: `sum(C_hi/T for HI tasks)`.

### generate_ensemble()

```python
generate_ensemble(n_replicates: int, U: float,
    rng_seed: int = 42, **kwargs) -> list[TaskSet]
```

Uses deterministic seeding: base seed derived from `(U, n_replicates)`, each replicate
uses `rng_seed + i`.

### Tests (`tests/generation/test_taskset.py`)

- Utilisation sums correct for both `hi_mode` values across a spread of U
- Constraints respected in both modes (C_lo ≤ C_hi for HI tasks in `drs_independent`)
- Individually-infeasible fraction is reported and non-negative
- Reproducibility: same seed → identical task set
- Criticality count: `n_hi = round(CP × n)`, `n_lo = n − n_hi`
- Ensemble size and reproducibility
- `drs_independent` produces zero infeasible tasks across 100 seeds at high U

**Definition of done:** generator runs for both `hi_mode` values across a spread of U values,
tests pass, and the individually-infeasible diagnostic is visibly non-trivial at high U
under `fixed_ratio` (demonstrating why the flag matters).

---

## Phase 4 — Priority Assignment & Analytic Schedulability Filter

### Priority Assignment (`amc_tasksim/scheduling/priority.py`)

```python
assign_deadline_monotonic(taskset: TaskSet) -> TaskSet
```

Lower deadline = higher priority (smaller priority number). Ties broken by task index.

### AMC-rtb (`amc_tasksim/scheduling/amc_rtb.py`)

Implement the AMC-rtb schedulability test from `docs/AMCRRTAS2022.pdf` Section IV-A
(equations 1–2), or equivalently `docs/CAMCRTNS2022.pdf` Section 4.1.

**Ri(LO)** — standard fixed-priority response-time analysis (fixed-point iteration):

```
R_i(LO) = C_i(LO) + Σ_{j∈hp(i)} ⌈R_i(LO) / T_j⌉ × C_j(LO)
```

where `hp(i)` = higher-priority tasks.

**Ri(HI)** (HI-criticality tasks only):

```
R_i(HI) = C_i(HI) + Σ_{j∈hp_HI(i)} ⌈R_i(HI) / T_j⌉ × C_j(HI)
        + Σ_{j∈hp_LO(i)} min(⌈R_i(HI) / T_j⌉, ⌈R_i(LO) / T_j⌉ + 1) × C_j(LO)
```

where `hp_HI(i)` = higher-priority HI-criticality tasks,
`hp_LO(i)` = higher-priority LO-criticality tasks.

Return per-task response times and an overall `schedulable: bool`.

### Non-Trivial AMC Filter (`amc_tasksim/scheduling/filters.py`)

```python
is_nontrivial_amc_taskset(taskset: TaskSet) -> bool
```

A task set is "non-trivial" if:
- (a) Schedulable under AMC-rtb, AND
- (b) At least one HI-criticality task is NOT schedulable under plain fixed-priority
    analysis assuming every task executes at `max(C_i(LO), C_i(HI))`

This follows the methodology in `docs/AMCRRTAS2022.pdf` Section V-C.

### Tests (`tests/scheduling/test_amc_rtb.py`)

Reproduce the worked 3-task example from `docs/AMCRRTAS2022.pdf` Appendix A by hand:

| Task | C_lo | C_hi | T | D | Criticality |
|---|---|---|---|---|---|
| τ₁ | 1 | 1 | 2 | 2 | LO |
| τ₂ | 1 | 5 | 10 | 10 | HI |
| τ₃ | 4 | 4 | 100 | 18 | HI |

Confirm: **R₃(LO) = 10** (interference from 5 jobs of τ₁ at 1 tick each, and 1 job of τ₂
at 1 tick), **R₃(HI) = 19** (same τ₁ interference plus 2 jobs of τ₂ at 5 ticks each).
τ₃ is unschedulable (R₃(HI) = 19 > D₃ = 18).

Also test: a schedulable task set passes AMC-rtb, the non-trivial filter works on both
schedulable and non-schedulable sets, and DMPA priority ordering is correct.

**Definition of done:** the Appendix-A worked example reproduces R₃(LO) = 10, R₃(HI) = 19
exactly.

---

## Phase 5 — Event-Driven Simulator Core

Implement `amc_tasksim/simulation/engine.py`: a discrete-event (not tick-by-tick) simulator
for fixed-priority preemptive scheduling of a single TaskSet on one core.

### Job Model

Each job draws its execution time **once at release**:

| Task type | Execution time distribution |
|---|---|
| LO-criticality | `uniform(BCET_i, C_lo_i)` |
| HI-criticality (normal) | `uniform(BCET_i, C_lo_i)` |
| HI-criticality (HI behaviour) | `uniform(C_lo_i, C_hi_i)` |

HI behaviour is triggered with probability **FP = 1/N** per job.

### Mode-Change Protocol Interface

```python
class ModeChangeProtocol(ABC):
    def check_enter_degraded(self, active_jobs: list[Job], current_time: int) -> bool: ...
    def check_exit_degraded(self, active_jobs: list[Job], current_time: int) -> bool: ...
```

### OriginalAMC (baseline)

| Event | Behaviour |
|---|---|
| Enter degraded | Any active HI-criticality job has executed for C_i(LO) without completing |
| Exit degraded | Idle instant (no active jobs of any task) |
| LO-criticality releases in degraded | Dropped (not queued, not executed) — counted as JNE |

### Metrics

| Metric | Meaning |
|---|---|
| **NiD** | Number of times degraded mode was entered |
| **TiD** | Total time in degraded mode, as fraction of duration |
| **JNE** | Count of LO-criticality jobs dropped in degraded mode |
| **LDM** | Count of LO-criticality jobs that executed but missed deadline |
| **HDM** | Count of HI-criticality deadline misses (should be 0 for schedulable sets) |
| **HI releases per task** | Total HI-criticality job releases per task |
| **HI trigger events** | Total HI-behaviour "trigger" events observed |

### Defensive Check

If any job's realised execution time exceeds its enforced budget
(C_hi for HI tasks, C_lo for LO tasks in normal mode), abort the job and log a warning.

### Testing Hook

Support optional `release_times: list[int | None]` for explicit release times (needed for
Phase 6 validation).

**Function signature:**

```python
simulate(taskset, duration=10**6, seed=None,
         mode_protocol=OriginalAMC(), fp=1e-4,
         release_times=None) -> SimulationResult
```

**Definition of done:** engine runs on a small hand-built task set without errors, metrics
dataclass populates correctly, defensive budget-overrun check exists and is tested.

---

## Phase 6 — Validation Suite

### Appendix A / Figure 13 Scenario

Reproduce the exact scenario from `docs/AMCRRTAS2022.pdf` Appendix A / Figure 13:

- τ₁: C_lo=1, T=2, D=2, LO
- τ₂: C_lo=1, C_hi=5, T=10, D=10, HI
- τ₃: C_lo=C_hi=4, T=100, D=18, HI
- Worst-case: τ₂ released at t=6, forced to exhibit full HI behaviour (execute for C_hi=5)
- Expected: degraded mode entered at t=8 (when τ₂ has executed 1 tick = C₂(LO) without completing)
- τ₃ completes its final time unit for a worst-case HI-criticality response time of 13

### Sanity Tests

1. **FP = 0** (N → ∞) — never enters degraded mode (NiD = 0, TiD = 0, JNE = 0)
   *Requires BCET < C_lo for HI tasks so execution times never reach C_lo.*
2. **CF = 1.0** (C_hi = C_lo for all tasks) — behaves like single-criticality FPPS;
   degraded mode may enter/exit per the trigger definition but JNE/LDM driven purely by
   execution time overruns, not by criticality distinction.
3. **Utilisation conservation** — for a large ensemble, empirical fraction of ticks each
   task spends executing (in the schedulable subset) is consistent with its assigned U_i(LO).

**Definition of done:** the Appendix-A scripted scenario reproduces the paper's trace; all
sanity tests pass.

---

## Phase 7 — Experiment Orchestration (the Sweep)

### Sweep Structure

- **U** ∈ [0.05, 0.10, …, 0.95] (configurable)
- **N** ∈ [10, 100, 1000, 10000, 100000] (configurable, log-spaced)
- **1000 replicates** per (U, N) combination (20 with `--quick`)

**Key optimisation:** task-set generation depends only on U (not N). Generate each task set
once per U, then simulate it once per N value. This avoids regenerating task sets redundantly.

### CLI

```bash
uv run amc_tasksim \
    --quick                          # 20 replicates
    --hi-mode fixed_ratio            # or drs_independent
    --protocol original_amc          # or amc_rh, amc_ra
    --n-replicates 1000
    --duration 1000000
    --output results/sweep.parquet
    --n-values 10 100 1000 10000 100000
    --U-range 0.05 0.95 0.05
```

### Output

Single tidy pandas DataFrame / parquet file, one row per `(U, N, hi_mode, protocol, replicate_index)`:

| Column | Type |
|---|---|
| U | float |
| N | int |
| FP | float |
| hi_mode | str |
| protocol | str |
| replicate_index | int |
| nid, tid, jne, ldm, hdm | int/float |
| hi_trigger_events | int |
| total_hi_releases | int |
| individually_infeasible | int |
| aggregate_hi_utilisation | float |
| schedulable_amc_rtb | bool |
| nontrivial_amc | bool |

### Statistical Power Warning

After each (U, N) combination completes, compute the total number of HI-trigger events
observed across all replicates. If this falls below **100 events** (configurable), print
a clear warning that NiD/TiD/JNE estimates for this cell may be unreliable due to too few
rare-event occurrences given the simulation duration.

**Definition of done:** `--quick` sweep completes, produces a results file, and the
statistical-power warning logic demonstrably triggers for at least the largest N value
(expected and fine — it tells you where you need longer duration or more replicates).

---

## Phase 8 — Analysis & Plotting

Implement `amc_tasksim/analysis/plots.py`, reading the results file from Phase 7.

### Plots

1. **Box-and-whisker: NiD(%)** — min, 5th percentile, 25th/median/75th percentile box,
   95th percentile, max. Faceted/coloured by N.
2. **Box-and-whisker: TiD(%)** — same format.
3. **Box-and-whisker: JNE(%) + LDM(%)** — same format.
4. **Heatmap: Success Ratio** — fraction of task sets in each (U, N) cell with HDM = 0.
   Should be ~100% everywhere if generation/analysis are correct; surfaces bugs quickly.
5. **Heatmap: Statistical Power** — total HI-trigger events per cell. Low-power regions
   are visually obvious.
6. **Comparison plot: hi_mode overlay** — overlay `fixed_ratio` vs `drs_independent`
   results on the JNE + LDM metric for the same (U, N) grid.

### Summary

Write `results/SUMMARY.md` describing what the plots show and flagging:
- Non-zero HDM (should be 0)
- Low statistical power regions
- Unexpectedly high individually-infeasible fractions under `fixed_ratio`

**Definition of done:** figures generated from the `--quick` run's output, `SUMMARY.md` written
and sensible.

---

## Phase 9 — Pluggable Scheduler Variants: AMC-RH / AMC-RA

Extend the `ModeChangeProtocol` interface with two additional variants from
`docs/AMCRRTAS2022.pdf` Section IV.

### AMC-RH

| Event | Behaviour |
|---|---|
| Enter degraded | Active HI-criticality job τ_i reaches R_i(LO) from the start of the priority level-i busy period in which it was released |
| Exit degraded | No active HI-criticality job has reached R_i(LO) from its busy period start |

Requires tracking, per active job, the start time of its priority level-i busy period.
Uses the O(1)-per-release bookkeeping from Appendix B of the AMC-RH paper: inherit
busy-period start time from the next-higher-priority active task in the run-queue.

### AMC-RA

| Event | Behaviour |
|---|---|
| Enter degraded | Same as AMC-RH (R_i(LO) from busy period start) |
| Exit degraded | Idle instant (as in the original AMC scheme) |

### Sweep

Re-run the Phase 7 sweep (at least `--quick` scale) with all three protocols
(original AMC, AMC-RH, AMC-RA) and confirm qualitatively that AMC-RH/AMC-RA reduce
NiD, TiD, and JNE + LDM relative to original AMC, consistent with Table I of
`docs/AMCRRTAS2022.pdf` (AMC-RH reducing JNE + LDM to roughly 2.5% of the original
AMC value for semi-harmonic periods). Exact reproduction isn't the goal since the task
model and duration differ, but the direction and rough magnitude should match.

**Definition of done:** all three protocols runnable through the same sweep infrastructure;
AMC-RH shows a clear, large reduction in JNE + LDM relative to original AMC in the data.

---

## Build Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                                                       → Phase 7 → Phase 8
Phase 9 (optional, can run in parallel with 7–8)
```

Each phase ends with a "Definition of done" — tests green, example output shown.
Don't move to the next phase until that's satisfied.
