# Metrics and Objective Function for Multi-Level Scheduling

## Background

This document extends the metrics from AMC-RH to support multi-level degradation scheduling.

## Existing Metrics (AMC-RH)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **NiD** | $\sum_{t} \Ind[\text{mode}(t-1) = L_0 \wedge \text{mode}(t) \neq L_0]$ | Number of degraded mode entries |
| **TiD** | $\frac{1}{T} \int_0^T \Ind[\text{mode}(t) \neq L_0] \, dt$ | Fraction of time in degraded mode |
| **JNE** | $\sum_{i \in \tau^{\mathrm{LO}}} \sum_{j=1}^{\infty} \Ind[\text{job } j \text{ of } \tau_i \text{ abandoned on release}]$ | LO jobs never executed |
| **HDM** | $\sum_{i \in \tau^{\mathrm{HI}}} \sum_{j=1}^{\infty} \Ind[\text{job } j \text{ misses deadline}]$ | HI jobs missing deadlines — a certification failure |

> **LDM is gone.** A LO-criticality job that has not completed at its deadline is
> *terminated* there, never allowed to complete late (`task_model.tex`
> §"Deadline Termination"), so a LO-criticality deadline miss is impossible by
> construction. What used to be counted as LDM is a deliberate policy action and
> is counted below as `Terminated`, feeding JNC — not as a failure.

## Primary Objective: Jobs Not Completed (JNC)

JNE alone is the wrong objective once retained LO tasks are permitted to fail:
it counts only jobs that never ran, and silently omits every job that ran and
was then thrown away. Measure the shortfall against a fixed reference instead.

Let $\mathrm{Exp}$ be the LO-criticality jobs a *perfect* scheduler would have
completed over a run of length $H$ — a function of $H$ and the periods alone,
so independent of anything the scheme did — and $\mathrm{Comp}$ those that
actually completed within their deadlines:

$$
\mathrm{Exp} = \sum_{i \in \tau^{\mathrm{LO}}} \left| \{\, \phi_i + m T_i < H \,\} \right|,
\qquad
\mathrm{JNC} = \mathrm{Exp} - \mathrm{Comp}
= \underbrace{\mathrm{JNE}}_{\text{never ran}} + \underbrace{\mathrm{Terminated}}_{\text{ran, discarded}} + \varepsilon
$$

where $\varepsilon$ is at most one in-flight job per task at the horizon.
Defining JNC by *difference* rather than by summing failure modes is deliberate:
it cannot silently omit one.

## Normalisation

**Every metric is reported normalised.** A raw count scales with both the run
length and the tasks' periods, so raw JNE is not comparable between task sets or
between experiments of different duration:

$$
\mathrm{JNC\%} = 100 \cdot \frac{\mathrm{JNC}}{\mathrm{Exp}}, \qquad
\mathrm{JNE\%} = 100 \cdot \frac{\mathrm{JNE}}{\mathrm{Exp}}, \qquad
\mathrm{WastedCPU\%} = 100 \cdot \frac{\mathrm{WastedCPU}}{H}
$$

## New Metrics for Multi-Level

### Wasted CPU
Processor time spent on work that was thrown away:

$$
\mathrm{WastedCPU} = \sum_{i \in \tau^{\mathrm{LO}}} \sum_{j=1}^{\infty}
\left( \text{executed}_{ij} \cdot \Ind[\text{job } j \text{ terminated at its deadline}] \right)
$$

Abandonment *on release* never contributes here — such a job executes nothing.
The metric is live only because of deadline termination, which abandons a job
*mid-execution*. It was zero by construction before termination was introduced.

### Service Ratio
Fraction of LO-criticality jobs that delivered a result:

$$
\mathrm{ServiceRatio} = \frac{\mathrm{Comp}}{\mathrm{Exp}} = 1 - \frac{\mathrm{JNC}}{\mathrm{Exp}}
$$

Note this is **not** $1 - \mathrm{JNE}/\mathrm{Exp}$, which was the earlier
definition: that counts a job as served if it was allowed to start, even if it
was later discarded having produced nothing.

### Level Transition Count
Number of times the system changes degradation levels:

$$
\mathrm{LevelTrans} = \sum_{t} \Ind[\text{level}(t-1) \neq \text{level}(t)]
$$

This penalizes excessive mode oscillation.

## Objective Function Candidates

### Candidate 1: Weighted Cost Minimization
$$
\Phi_1 = \alpha \cdot \E[\mathrm{JNC}] + \beta \cdot \E[\mathrm{TiD}] + \gamma \cdot \E[\mathrm{WastedCPU}]
$$

**Properties**:
- Lower is better (minimization)
- $\alpha, \beta, \gamma \geq 0$ are weighting coefficients
- Captures all three cost dimensions: lost work, degradation time, wasted cycles

### Candidate 2: Service Maximization
$$
\Phi_2 = \E[\mathrm{ServiceRatio}] - \alpha \cdot \E[\mathrm{JNC}] - \beta \cdot \E[\mathrm{TiD}]
$$

**Properties**:
- Higher is better (maximization)
- Directly optimizes service preservation
- Penalty terms prevent excessive degradation time

### Candidate 3: Multi-Objective Pareto Front
Represent the solution as a set of non-dominated points across:
- $\E[\mathrm{JNC}]$
- $\E[\mathrm{TiD}]$
- $\E[\mathrm{WastedCPU}]$

**Properties**:
- No weighting needed
- Reveals trade-offs between metrics
- Requires user selection from Pareto front

## Proposed Objective: Hybrid Approach

We recommend **$\Phi_1$ with adaptive weights, plus a churn term**:

$$
\Phi = \alpha(U) \cdot \E[\mathrm{JNC}] + \beta(U) \cdot \E[\mathrm{TiD}] + \gamma \cdot \E[\mathrm{WastedCPU}] + \delta \cdot \E[\mathrm{LevelTrans}]
$$

Where weights adapt to total utilisation $U$:
- Low $U$: $\alpha$ high (focus on preserving LO work)
- High $U$: $\beta$ high (minimize degradation time)
- $\gamma$ fixed small value (waste is always undesirable)
- $\delta$ fixed small value (churn is always undesirable)

### Why a churn term, and why LevelTrans rather than NiD

TiD measures total *duration* in a degraded state; it says nothing about
*frequency*. A severity ladder with poorly spaced thresholds can oscillate
across a boundary -- entering and leaving a level repeatedly with short dwell
times each visit -- while keeping both JNC and TiD low, because each excursion
is brief. Nothing else in $\Phi$ penalises that, so nothing steers the
optimiser away from it.

**LevelTrans** (`Number of degradation levels`, defined above) rather than the
legacy **NiD** (`Number of degraded mode entries`), because NiD only counts
transitions *out of* $L_0$ and is blind to oscillation entirely within the
degraded levels ($L_2 \leftrightarrow L_3$, say) -- exactly the failure mode a
badly-spaced $k$-level ladder can produce, and exactly the case NiD cannot see
by construction. LevelTrans is a strict generalisation: at $k=2$ it is
proportional to NiD (each excursion contributes one entry and one exit rather
than one entry alone), so nothing about the two-level baseline comparison is
lost.

One caveat worth stating rather than discovering later: LevelTrans grows
mechanically with $k$ even for equally good behaviour, because reaching the
same eventual severity through more, finer-grained levels costs more
transition *events* than reaching it through one big step. This is not a flaw
to correct — it is the real price of grading the response, and $\Phi$ should
charge for it rather than assume finer granularity is free. It does mean a
larger $k$ needs a strictly better JNC/TiD/WastedCPU trade to win on $\Phi$
overall, which is the correct comparison to make.

NiD remains a **reported** metric (Table above) for direct comparison against
the AMC-RH baseline, which only ever defines it. It is not double-counted in
$\Phi$ once LevelTrans is included.

### Weight Selection Strategy
| Utilisation Range | $\alpha$ | $\beta$ | $\gamma$ | $\delta$ |
|-------------------|----------|---------|----------|----------|
| $U < 0.5$ | 1.0 | 0.5 | 0.1 | 0.1 |
| $0.5 \leq U < 0.7$ | 0.8 | 0.8 | 0.2 | 0.1 |
| $0.7 \leq U < 0.85$ | 0.5 | 1.0 | 0.3 | 0.15 |
| $U \geq 0.85$ | 0.3 | 1.0 | 0.5 | 0.15 |

## "Meaningful Improvement" Criteria

### Criterion 1: Service Preservation (Monotonic JNC)

For any task set $\tau$ and configuration with $k \geq 2$ levels:
$$
\E[\mathrm{JNC}_k(\tau)] \leq \E[\mathrm{JNC}_{k-1}(\tau)]
$$

**Interpretation**: Adding a degradation level should never increase lost LO work for any task set.

**Validation Method**:
```python
# For a specific task set, run simulations with k and k-1 levels
def validate_service_preservation(taskset):
    jne_k = simulate_multi_level(taskset, k_levels)
    jne_km1 = simulate_multi_level(taskset, k-1_levels)
    return jne_k <= jne_km1 + epsilon  # Allow numerical tolerance
```

### Criterion 2: Waste Reduction (Total Bad Work Monotonicity)

For any task set $\tau$ and configuration with $k \geq 2$ levels:
$$
\E[\mathrm{WastedCPU}_k + \mathrm{JNC}_k] \leq
\E[\mathrm{WastedCPU}_{k-1} + \mathrm{JNC}_{k-1}]
$$

**Interpretation**: Total "bad" work (wasted CPU cycles + dropped jobs) should not increase when adding levels.

**Equivalently**:
$$
\E[\mathrm{WastedCPU}_k] - \E[\mathrm{WastedCPU}_{k-1}] \leq
\E[\mathrm{JNC}_{k-1}] - \E[\mathrm{JNC}_k]
$$

The reduction in dropped jobs should at least offset any increase in wasted cycles.

**Validation Method**:
```python
def validate_waste_reduction(taskset):
    wasted_k = simulate_multi_level(taskset, k_levels).wasted_cpu
    jne_k = simulate_multi_level(taskset, k_levels).jne
    wasted_km1 = simulate_multi_level(taskset, k-1_levels).wasted_cpu
    jne_km1 = simulate_multi_level(taskset, k-1_levels).jne

    total_bad_k = wasted_k + jne_k
    total_bad_km1 = wasted_km1 + jne_km1
    return total_bad_k <= total_bad_km1 + epsilon
```

### Criterion 3: Statistical Improvement (Expected Objective)

For task sets drawn from generation distribution $G$:
$$
\E_{\tau \sim G}[\Phi_k(\tau)] < \E_{\tau \sim G}[\Phi_{k-1}(\tau)]
$$

Where the objective function $\Phi$ is as defined in "Proposed Objective: Hybrid
Approach" above (weights adapt to $U$ per the table there; not repeated here).

**Interpretation**: On average across random task sets, multi-level scheduling with more levels achieves a better objective value.

**Validation Method (Ensemble Test)**:
```python
def validate_statistical_improvement(gen_distribution, k_levels, n_samples=1000):
    phi_k_sum = 0
    phi_km1_sum = 0

    for _ in range(n_samples):
        taskset = gen_distribution.sample()
        phi_k = compute_objective(simulate_multi_level(taskset, k_levels))
        phi_km1 = compute_objective(simulate_multi_level(taskset, k-1_levels))
        phi_k_sum += phi_k
        phi_km1_sum += phi_km1

    mu_k = phi_k_sum / n_samples
    mu_km1 = phi_km1_sum / n_samples

    # t-test for H0: mu_k >= mu_km1 vs H1: mu_k < mu_km1
    return mu_k < mu_km1 - epsilon  # Require meaningful gap
```

## Statistical Validation Procedure

### Step 1: Single-Task Set Verification
For a specific task set $\tau$:
1. Compute JNC, TiD, WastedCPU (all normalised) for $k=2,3,4,5$
2. Verify monotonicity: each metric decreases or stays same as $k$ increases
3. Record whether all three criteria are satisfied

### Step 2: Ensemble Validation (Paired Hypothesis Testing)

**The comparison must be paired**, and this is not a refinement — it decides
whether the procedure works at all. Configurations are evaluated on the *same*
task sets and the *same* seeds (common random numbers), and the statistic is the
mean of per-task-set differences, not the difference of two means. Measured on
this simulator at a fixed budget of 20 task sets x 20 seeds:

| Design | Std err | Smallest resolvable difference |
|---|---|---|
| A fresh population per configuration | 19.504 | 89.8% |
| Same population, analysed unpaired | 7.719 | 35.5% |
| Same population, analysed paired | 0.866 | **4.0%** |

Only the last row can support the 5% threshold of Step 3. Reaching 5% at 95%
power needs roughly **43 task sets paired**, against roughly **3,400 unpaired**.
Pairing cannot be added retrospectively — it has to exist when the runs are
generated — so an unpaired study is not merely weaker, it is unrecoverable.

Aggregate each task set's seeds to one value *before* differencing: two
configurations consume the random stream differently, so their individual runs
decorrelate, while task-set means stay correlated through the shared task set.
`amc_tasksim.experiments.contract` implements this and refuses unequal-length
samples rather than silently falling back to an unpaired test.

1. Generate $N$ task sets at utilisation $U$ from distribution $G$, once, and
   reuse them for every configuration
2. Run multi-level scheduling for each $k \in \{2,3,4,5\}$
3. Compute sample means:
   $$
   \hat{\mu}_k = \frac{1}{N}\sum_{i=1}^N \Phi^{(i)}_k
   $$
4. Apply one-tailed t-test: $H_0: \mu_k \geq \mu_{k-1}$ vs $H_1: \mu_k < \mu_{k-1}$
5. Reject null hypothesis if p-value $< 0.05$

### Step 3: Threshold Check (Practical Significance)
A configuration is "meaningfully better" if:
$$
\frac{\mu_{k-1} - \mu_k}{\mu_{k-1}} > \epsilon
$$

for threshold $\epsilon = 0.05$ (5% improvement). This ensures the improvement is not just statistically significant but also practically meaningful.

## Implementation Notes

### Computing Wasted CPU in Simulation
```python
# Pseudocode for wasted CPU tracking
def on_job_abandon(job, current_time):
    if job.executed > 0:
        result.wasted_cpu += job.executed
```

### Tracking Level Transitions
```python
# Maintain previous level and increment on change
prev_level = current_level
current_level = get_degradation_level()
if prev_level != current_level:
    result.level_transitions += 1
```

## References

- AMC-RH Section V-A: Metrics definition
- The weighted cost approach follows standard multi-objective optimization techniques
- Statistical validation follows the hypothesis testing methodology in Baruah et al. (DRS, RTSS 2020)
