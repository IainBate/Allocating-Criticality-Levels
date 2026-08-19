# Optimal Mode Design: Implementation Plan

## Phase 3 Overview

This phase determines:
1. The optimal number of degradation levels `k`
2. The optimal trigger point spacing `{R_1, R_2, ..., R_{k-1}}`
3. The best configuration strategy for practical implementation

---

## Task 3.1: Fixed Trigger Point Analysis (All R_level = R_trigger)

### Objective
When intermediate triggers all equal the full degradation trigger `R_trigger`, the system only differs in **which tasks are dropped** at each level transition, not **when** transitions occur.

### Approach

#### 3.1.1: Drop Strategy Characterization
For a fixed k-level scheme with all `R_level = R_trigger`:

| Level | Entry Condition | Drop Decision |
|-------|-----------------|---------------|
| L_0 → L_1 | Any HI task reaches R_trigger | Drop subset S_1 of LO tasks |
| L_1 → L_2 | Any HI task reaches R_trigger (still) | Drop additional tasks in S_2 \ S_1 |
| ... | ... | ... |
| L_{k-2} → L_{k-1} | Any HI task reaches R_trigger | Drop remaining LO tasks |

**Key Insight**: Since all levels trigger at the same point, the only design variable is **which subset of LO tasks to drop at each transition**.

#### 3.1.2: Candidate Drop Strategies

| Strategy | Description | Complexity | Parameter Space |
|----------|-------------|------------|-----------------|
| **Priority-based** | Drop lowest-priority LO tasks first | O(1) per decision | None (inherent in FPPS) |
| **Utilization-based** | Drop highest-utilization LO tasks first | O(n log n) per level | None |
| **Deadline-based** | Drop tightest-deadline LO tasks first | O(n log n) per level | None |
| **Proportional f** | Drop fraction `f` of each task's active jobs | O(n) with tracking | Single scalar f ∈ [0,1] |
| **Hybrid** | Priority + utilization weighting | O(n log n) | Weight α ∈ [0,1] |

### Implementation Steps

```python
# Pseudocode for k-level simulation
def simulate_k_level(taskset, k, drop_strategy, seed):
    r_lo = amc_rtb(taskset).r_lo
    
    # All intermediate triggers equal full trigger
    trigger_points = [int(math.ceil(r)) for r in r_lo]  # R_1 = R_2 = ... = R_trigger
    
    state = SchedulerState()
    result = SimulationResult()
    
    # Drop sets: S_1 ⊆ S_2 ⊆ ... ⊆ S_{k-1} = all LO tasks
    drop_sets = compute_drop_sets(taskset, k, drop_strategy)
    
    while state.time < duration:
        # Check for mode entry (same as 2-level AMC-RH)
        trigger_time = find_earliest_trigger(state, trigger_points)
        
        if trigger_time and state.mode == "normal":
            next_level = get_current_level() + 1
            jobs_to_drop = drop_sets[next_level] - drop_sets[next_level-1]
            abandon_jobs(jobs_to_drop)
            state.mode = f"degraded_{next_level}"
        
        # ... rest of simulation engine ...
    
    return result
```

### Expected Findings

**Hypothesis**: For fixed trigger points, the number of levels matters less than **how** tasks are selected for dropping. A well-chosen k=3 scheme with smart dropping may outperform k=2 with naive dropping.

---

## Task 3.2: Variable Trigger Point Optimization

### Objective
Find optimal `{R_1, R_2, ..., R_{k-1}}` where `R_1 ≤ R_2 ≤ ... ≤ R_{k-1} = R_trigger`.

### Parameter Space

For k levels:
- **Trigger points**: `R_1 ≤ R_2 ≤ ... ≤ R_{k-1}`, so k-1 in total.
- **Fixed endpoint**: `R_{k-1} = R_trigger`, as stated in the Objective above. This is
  not a normalisation convenience: `safety_proof.md` pins it in three places (§Problem
  Statement, §Degradation Level Transitions, Theorem 1) and Theorem 1's proof depends
  on it. **The optimiser must never move it** — doing so forfeits the "no worse than
  AMC-RH" guarantee that motivates the whole scheme.
- **Free variables**: therefore **k-2**, not k-1.

| k | free dimensions | what is actually being optimised |
|---|-----------------|----------------------------------|
| 2 | 0 | nothing — this *is* AMC-RH |
| 3 | 1 | a one-dimensional line search over R_1 |
| 4 | 2 | (R_1, R_2) |
| 5 | 3 | (R_1, R_2, R_3) |

- **Normalization**: scale by R_trigger, giving `x_i = R_i / R_trigger` in (0, 1], with
  `x_{k-1} = 1` pinned and excluded from the search.

```python
# Free variables: x_1 <= x_2 <= ... <= x_{k-2}, each in (0, 1].
# x_{k-1} = 1 is pinned to R_trigger and is NOT searched.
R_i = x_i * R_trigger   for i = 1, ..., k-2
R_{k-1} = R_trigger
```

Enumerate the ordering constraint directly rather than sampling a box and repairing it
by sorting: the feasible set is exactly the non-decreasing (k-2)-tuples drawn from the
grid values, which `itertools.combinations_with_replacement` generates without any
rejection or projection step.

### Optimization Method

Exactly one method is used: **exhaustive enumeration of the ordered trigger lattice,
evaluated under common random numbers**. The rationale is below; it is recorded here
rather than in a comparison table because the choice follows from the structure of this
problem and should not be re-litigated per experiment.

```python
from itertools import combinations_with_replacement

def enumerate_trigger_lattice(k, grid):
    """Every feasible normalised trigger vector, in non-decreasing order.

    `grid` is the ladder of candidate values in (0, 1], e.g. [0.05, 0.10, ..., 1.00].
    The pinned endpoint x_{k-1} = 1 is appended, never searched.
    """
    return [list(c) + [1.0] for c in combinations_with_replacement(grid, k - 2)]


def optimise(tasksets, k, grid, seed_block):
    """Evaluate every lattice point on the SAME task sets and seeds."""
    surface = []
    for x in enumerate_trigger_lattice(k, grid):
        per_seed = evaluate_objective(tasksets, k, x, seed_block)   # paired sample
        surface.append((x, per_seed))
    return surface   # the whole surface, not an argmin -- see "Selecting a winner"
```

#### Why enumeration, and not a metaheuristic

**1. The search space is small.** With k-2 free dimensions and the ordering constraint,
the feasible set is the non-decreasing (k-2)-tuples, counted by `C(g+d-1, d)` for grid
size g and dimension d — not the `g^(k-1)` hypercube:

| k | free dims | hypercube count (g=20) | actual ordered lattice |
|---|-----------|------------------------|------------------------|
| 3 | 1 | 400 | **20** |
| 4 | 2 | 8,000 | **210** |
| 5 | 3 | 160,000 | **1,540** |
| 6 | 4 | 3,200,000 | **8,855** |

At k=3 — the headline experiment of §3.3.2 — this is a 20-point line search. Nothing in
the planned range k ∈ {2,3,4,5} is prohibitive, so there is no search-space pressure for
a metaheuristic to relieve.

**2. The binding constraint is Monte-Carlo noise, not search cost.** Every objective
value is an estimate. Measured per-seed coefficients of variation on this simulator are
0.157 (NiD), 0.314 (JNE) and 0.199 (degraded ticks). An argmin over M noisy evaluations
is biased low by roughly `sigma * sqrt(2 ln M)` — the winner's curse — and that bias
grows with the number of evaluations the method performs. A genetic algorithm at
population 50 x 100 generations makes 5,000 evaluations, inflating the bias to roughly
12%: more than double the 5% practical-significance threshold used to declare a result
meaningful. **A GA run against a perfectly flat objective would return a
publishable-looking optimum.** A 20-point lattice carries a smaller bias, and — because
its points lie on a regular grid — a *correctable* one: neighbouring points can be
pooled and an indifference set reported instead of a single winner.

**3. The deliverable is a surface, not an argmin.** §3.3.2 predicts a plateau ("optimal
R_1 is around 0.5-0.7"). That is a claim about the shape of the objective over the whole
domain. Uniform enumeration produces it natively; an adaptive sampler deliberately
concentrates evaluations and cannot support it.

**4. Reproducibility.** Enumeration has one hyperparameter — grid resolution — and it is
falsifiable: a referee refines the grid and checks the argmin does not move. A GA has at
least eight unpinned knobs (population, generations, selection fraction, crossover
operator, mutation rate and distribution, initialisation, elitism), Bayesian optimisation
a comparable number (kernel, acquisition, restarts, noise prior). `sklearn` is also not a
project dependency.

#### When this choice should be revisited

Enumeration stops being the right answer if either holds, and both are worth stating so
the decision is defensible rather than habitual:

- **k > 6 becomes interesting.** At k=7 the lattice passes 30,000 points.
- **The trigger vector gains dimensions.** If triggers become per-task rather than
  per-level, dimension scales with n instead of k and enumeration is immediately
  infeasible. At that point a method that exploits smoothness (Bayesian optimisation with
  an explicit noise term) earns its place — but it should be adopted *with* the paired
  evaluation protocol below, which is what actually makes any comparison resolvable.

> **Precondition — `skip_quiet` is unsound here.** The fast-forward in
> `amc_tasksim/simulation/engine.py` assumes an interval with no HI-criticality behaviour
> contains no mode change. That holds at the AMC-RH trigger under requirement R1, but is
> **false at any R_x < R_trigger**: measured with fp = 0 (no HI behaviour anywhere), a
> trigger at 0.9 * R_i(LO) still produced 14,118 mode changes and 356 abandoned jobs
> across 12 R1-schedulable task sets. Phase 3 must therefore budget the exact simulation
> cost, and `_skip_warm_up` needs a guard rejecting multi-level configurations rather
> than silently fast-forwarding over real transitions.
### Objective Function for Optimization

The objective is **stochastic**, so the evaluation budget is part of its definition, not
an implementation detail. Every configuration is evaluated on the *same* task sets and
the *same* seeds — common random numbers — so that configurations are compared as paired
samples and between-task-set variance cancels out of the comparison.

```python
def evaluate_objective(tasksets, k, x, seed_block):
    """Per-(taskset, seed) objective values for one configuration.

    Returns the raw vector rather than a scalar mean: the comparison between two
    configurations is PAIRED over this vector, and collapsing to a mean here would
    throw away the pairing that makes the comparison resolvable at all.

    `seed_block` is fixed across every configuration in the study (common random
    numbers). Do not derive it from the configuration.
    """
    return [
        # utilisation-first shedding matched the exhaustive minimum in every
        # case tested (see drop_sets.py); priority-first sheds 73-78% more
        # than necessary and is not a competitive default
        compute_objective(simulate_k_level(ts, k, x, drop_strategy="utilisation", seed=s))
        for ts in tasksets
        for s in seed_block
    ]

def compute_objective(result):
    """Compute Φ = α·JNE + β·TiD + γ·WastedCPU + δ·LevelTrans.

    See docs/metrics_objective.md "Proposed Objective: Hybrid Approach" for why
    LevelTrans rather than NiD: NiD only counts exits from L_0 and is blind to
    oscillation entirely within the degraded levels, which a badly-spaced
    severity ladder can produce without ever showing up in NiD or TiD.
    """
    alpha = get_alpha_by_utilisation(result.utilisation)   # weights adapt to U, not TiD
    beta = get_beta_by_utilisation(result.utilisation)
    gamma = 0.1   # fixed small weight for waste
    delta = 0.1   # fixed small weight for churn

    return (
        alpha * result.jne
        + beta * result.tid
        + gamma * result.wasted_cpu
        + delta * result.level_trans
    )
```

#### Why the pairing is load-bearing

Measured on this simulator, comparing two configurations on 20 task sets × 20 seeds:

| design | std err | smallest detectable difference |
|--------|---------|-------------------------------|
| unpaired (independent draws) | 7.719 | **72.5%** |
| paired (common random numbers) | 0.866 | **8.1%** |

A 79× variance reduction. Between-task-set sd is 24.54 and within-task-set sd 14.41, but
the sd of the *paired difference* is only 3.87 — pairing removes the dominant term. This
decides whether the project's own success criterion is reachable: `proof_requirements.md`
sets a **5% practical significance threshold**, which needs

| target effect | power | task sets, paired | task sets, unpaired |
|---------------|-------|-------------------|---------------------|
| 5% | 80% | 26 | 2,057 |
| 5% | 95% | **43** | 3,400 |
| 10% | 95% | 11 | 850 |

so 5% is comfortable when paired and effectively unreachable when not.

#### Selecting a winner

Never report a bare argmin over the surface. Instead:

1. Compute the paired mean and standard error at every lattice point.
2. Report the **indifference set** — every configuration within one standard error of the
   best — alongside the surface itself.
3. Re-evaluate only that set on a **held-out seed block** at higher replication, and
   declare a winner only if it survives at the 5% threshold.

Step 3 is what removes the winner's-curse bias: the selection and the confirmation use
independent randomness.

---

## Task 3.3: Research Report Deliverable

### Expected Content

#### Section 3.3.1: Optimal k Without Trigger Optimization

**Experiment**: For k ∈ {2,3,4} with all R_level = R_trigger:

| k | JNE (mean) | TiD (mean) | WastedCPU | Φ |
|---|------------|------------|-----------|----|
| 2 | baseline | baseline | baseline | baseline |
| 3 | ? ± se | ? ± se | ? ± se | ? |
| 4 | ? ± se | ? ± se | ? ± se | ? |

Report every cell as a **paired difference against the k=2 baseline, with its standard
error** — not as a bare percentage. The illustrative figures this table previously
carried (↓10%, ↓12%, ↓13% on JNE) implied a k=4-vs-k=5 separation of one percentage
point, which is below what any contemplated budget can resolve; stating differences
without their standard errors is what made that look reportable.

**Analysis**: Does the marginal benefit of additional levels justify complexity — and is
the k-to-k+1 difference larger than its own standard error?

#### Section 3.3.2: Optimal Trigger Spacing for k=3

**Experiment**: enumerate R_1/R_trigger over the full lattice (20 points at resolution
0.05), not a hand-picked four.

```
R_1/R_trigger → 0   enters L_1 almost immediately; degrades under ordinary load
R_1/R_trigger = 1   collapses to two-level AMC-RH
```

**Expected Finding**: the previous draft predicted an optimum around 0.5–0.7. Treat that
as a hypothesis to be falsified rather than confirmed: with fp = 0 (no HI-criticality
behaviour anywhere), a trigger at 0.9·R_i(LO) already produced 14,118 mode changes and
356 abandoned jobs across 12 R1-schedulable task sets. Intermediate triggers expressed as
a fraction of R_i(LO) fire on ordinary busy intervals, because R_i(LO) is a worst-case
bound that is rarely attained but whose fractions are attained routinely.

**The prior question this raises**: is a fraction of R_i(LO) the right trigger *quantity*
at all? Two alternatives worth evaluating before spending the budget on spacing — a
trigger on observed overrun (a job passing C_i(LO)) with a graded response, or one
derived from slack remaining to the deadline rather than elapsed busy-period time. If the
answer is no, the optimal spacing of a poorly-chosen trigger quantity is not a useful
result.

#### Section 3.3.3: Resolution and Cost

There is no method comparison to report — one method is used, for the reasons given in
§3.2. What this section reports instead is what the study could and could not resolve,
which is the question a referee will actually ask.

| k | lattice points | sims per point (30 sets × 30 seeds) | wall clock, 8 cores |
|---|----------------|-------------------------------------|---------------------|
| 3 | 20 | 900 | ~1.5 min |
| 4 | 210 | 900 | ~15 min |
| 5 | 1,540 | 900 | ~110 min |

Costed at the **exact** simulation rate (~43 ms per 10⁶ ticks), since `skip_quiet` is
unsound for R_x < R_trigger — see the precondition note in §3.2.

**Report alongside every result**: the seed block used, the paired standard error, and
the indifference set. A configuration reported without its standard error cannot be
distinguished from noise at these effect sizes.

---

## Implementation Checklist

- [ ] **3.0**: Guard `_skip_warm_up` against multi-level configurations (see §3.2
      precondition) — it currently fast-forwards over transitions that really occur
- [ ] **3.1a**: Implement k-level simulator with configurable drop strategies
      *(owned by Phase 5; Phase 3 consumes it — see the ownership note below)*
- [ ] **3.1b**: Add priority-based, utilization-based, deadline-based dropping
      *(owned by Phase 4)*
- [ ] **3.2a**: Implement lattice enumeration (`combinations_with_replacement`) and the
      paired evaluation harness with a fixed seed block
- [ ] **3.2b**: Implement indifference-set reporting and held-out-seed confirmation
- [ ] **3.3a**: Run experiments for k ∈ {2,3,4} at U ∈ {0.6, 0.7, 0.8, 0.9}
- [ ] **3.3b**: Analyze results and identify optimal configurations

> **Scope note on k.** §3.3.1's illustrative table separates k=4 from k=5 by one
> percentage point of JNE. At the measured variance that is below the resolution of any
> budget contemplated here, so the k range is stated as {2, 3, 4}. Adding k=5 requires
> stating the seed budget that makes it resolvable, not just running it.

> **Ownership.** Phase 3 owns **when** transitions occur (trigger spacing). Phase 4 owns
> **which** tasks are dropped. Phase 5 owns the k-level engine both depend on. Task 3.1
> as originally written spanned all three; keeping the split explicit is what lets the
> phases proceed independently.

---

## References

1. **AMC-RH Section V-D**: Experimental methodology for sweep studies
2. **DRS paper**: Task set generation distribution
3. **Common random numbers / variance reduction**: any standard simulation text, e.g.
   Law & Kelton, *Simulation Modeling and Analysis*, ch. 11 — the justification for the
   paired evaluation protocol in §3.2
4. **Selection bias in simulation optimisation**: the winner's-curse correction motivating
   the held-out-seed confirmation step
