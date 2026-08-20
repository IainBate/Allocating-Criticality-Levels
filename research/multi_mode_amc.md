# Multi-Mode Mixed-Criticality Scheduling: Research Plan

## Related Documentation

The formal task set model used in this research is documented separately:

- **[Task Set Model (LaTeX)](./task_set_model.tex)** - Formal mathematical specification with definitions, notation reference, and proofs
- **[Task Set Model (Markdown)](./task_set_model.md)** - Explanatory document with code snippets and examples

**See Also**: [[task-set-model]] memory entry for cross-referencing.

## Executive Summary

This research explores extending traditional two-mode AMC (Adaptive Mixed-Criticality) scheduling to support **multiple criticality levels**. Instead of binary normal/degraded modes, the system would have a hierarchy of degradation levels, allowing low-criticality tasks to be dropped in stages rather than all at once when high-criticality behavior is detected.

---

## 1. Background & Motivation

### Current AMC Limitations
- **AMC-RH** operates with two modes: normal and degraded
- Upon detecting HI-behavior (when a HI task reaches `R_i(LO)` from its busy period start), the system immediately enters degraded mode
- In degraded mode, **all** LO-criticality jobs are abandoned
- This is restrictive: even tasks that could meet their deadlines under partial load get dropped

### Research Goal
Introduce multiple criticality/degradation levels where:
- Each level has a different trigger point `R_level_X` for moving to the next level
- Low-criticality tasks are dropped incrementally as degradation escalates
- The system can balance between protecting HI tasks and preserving LO service

---

## 2. System Model

See the **[Task Set Model](./task_set_model.tex)** for complete formal definitions of:
- Task properties (T_i, D_i, C_i^lo, C_i^hi, BCET_i)
- Criticality assignments and utilisation calculations
- Response time analysis (RTA) for normal and HI-criticality modes
- AMC-rtb schedulability test

### Task Properties (unchanged from current model)
```
Task i:
  - T_i: period
  - D_i: deadline (= T_i for implicit-deadline)
  - C_lo[i]: execution time budget in normal mode
  - C_hi[i]: execution time budget in degraded mode (≥ C_lo[i])
  - BCET[i]: best-case execution time
  - criticality: "HI" or "LO"
```

### Multi-Level Model Extension

| Level | Name | Trigger Condition | Behavior |
|-------|------|-------------------|----------|
| L_0 | Normal | None | All tasks run normally |
| L_1 | Degraded-1 | Some HI task reaches R_1 | Drop some LO tasks (e.g., lowest priority) |
| L_2 | Degraded-2 | Some HI task reaches R_2 | Drop more LO tasks |
| ... | ... | ... | ... |
| L_k | Fully Degraded | HI task reaches R_k (≈ R_trigger) | Drop all LO tasks |

### Key Design Questions

1. **When to trigger level transitions?**
   - Option A: `R_level_X <= R_trigger` for each level X
   - Option B: Different thresholds per level based on system state
   
2. **Which tasks to drop at each level?**
   - Priority-based (lowest-priority LO tasks first)
   - Deadline-based (tightest-deadline LO tasks first)
   - Random selection
   - Proportional reduction based on utilization

3. **Exit strategy?**
   - Can exit directly to L_0, or must cascade through intermediate levels?
   - Should exit be state-dependent or trigger-based?

---

## 3. Staged Research Approach

### Phase 1: Foundational Analysis

#### Task 1.1: Precise Task Set Specification
**Objective**: Establish a well-documented baseline task set from existing literature for reproducible experiments.

**Approach**:
- Select one of the papers' worked examples (e.g., Appendix A from AMC-RH)
- Document exact parameters: periods, deadlines, execution times, criticality
- Verify current implementation reproduces expected R_i(LO) and R_i(HI) values
- Use this task set as the reference for all subsequent analysis

**Deliverable**: `docs/task_set_specification.md` with:
- Complete task table
- Manual calculation of response times
- Verification against existing code output

#### Task 1.2: Safety Proof for R_level_X <= R_trigger
**Objective**: Prove that using lower trigger points at intermediate levels maintains schedulability guarantees.

**Approach**:
- Formalize the multi-level model mathematically
- Show that if `R_1 <= R_2 <= ... <= R_k = R_trigger`, then:
  - HI tasks still meet their deadlines (by construction)
  - The system never enters a "worse than AMC-RH" state
  - LO task abandonment is monotonically non-decreasing with level
  
**Key Lemma**: If `R_level_X <= R_trigger`, then the set of tasks dropped at level X is a subset of those dropped at level Y for X < Y.

**Deliverable**: `docs/safety_proof.md` with formal proof

#### Task 1.3: Determine Required Proofs
**Objective**: Identify all necessary correctness proofs beyond the safety bound.

**Questions to Address**:
1. Is it sufficient to prove safety, or do we need optimality guarantees?
2. Does multi-level dropping affect HI-task schedulability analysis?
3. Are there new interference patterns introduced by partial LO dropping?

**Deliverable**: `docs/proof_requirements.md` listing:
- Proven properties
- Assumed properties
- Open proof questions

---

### Phase 2: Objective Function & Metrics

#### Task 2.1: Extend Metrics Beyond JNE, TiD, NiHCM
**Objective**: Define a comprehensive objective function that captures the benefits of multi-level scheduling.

**Proposed Metrics**:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **JNE** (existing) | LO jobs dropped in degraded mode | Lost low-criticality work |
| **TiD** (existing) | Time fraction in degraded mode | System degradation duration |
| **Wasted CPU** (new) | Σ(executed but abandoned LO work) | CPU cycles wasted on dropped tasks |
| **Service Ratio** (new) | (LO completions) / (LO releases) | Fraction of LO service preserved |

**Objective Function Candidates**:
1. Minimize: `α·JNE + β·TiD + γ·WastedCPU`
2. Maximize: `ServiceRatio - α·JNE - β·TiD`
3. Multi-objective: Pareto front across (JNE, TiD, WastedCPU)

#### Task 2.2: Define "Meaningful" Improvements
**Objective**: Establish criteria for when multi-level scheduling is beneficial.

**Proposed Criteria**:
1. **Service Preservation**: No task's JNE increases compared to one fewer mode
2. **Waste Reduction**: Total wasted CPU cycles decrease (or stay same)
3. **Statistical Improvement**: Expected value of objective function improves across random task sets

**Deliverable**: `docs/metrics_objective.md` with:
- Final metric definitions
- Objective function formula
- Validation criteria for "meaningful improvement"

#### Task 2.3: Overhead Analysis
**Objective**: Confirm multi-level scheduling doesn't introduce significant computational overhead.

**Analysis Points**:
1. **Trigger computation**: O(n) to check all active HI jobs (same as current)
2. **Drop decision**: O(m log m) for sorting LO tasks by drop priority, where m = active LO count
3. **State tracking**: O(k) per job for level information (k = number of levels, typically small)

**Deliverable**: `docs/complexity_analysis.md` showing:
- Time complexity per event type
- Space overhead per job
- Comparison to current implementation

---

### Phase 3: Optimal Mode Design

> **Status.** Superseded by the revised protocol in `../research/mode_optimization.tex`
> §"Revised Protocol" — see `docs/mode_optimization.md` §"Superseded" for the full account.
> In short: Tasks 3.1/3.2 below (trigger spacing, mode count, and choosing among the
> methods table in Task 3.3) turned out to have a vacuous answer under the adopted
> shed-early policy, so the method-comparison question retired *with evidence* rather than
> being run. What Phase 3 actually established is the regime map (Stage 2): the scheme
> beats two-level AMC-RA by +2.4% to +27.3%, driven by drop policy and operating point, not
> by anything in this section.

#### Task 3.1: Identify Optimal Number of Modes (R_level_X = R_trigger)
**Objective**: Determine the optimal number of degradation levels when all intermediate triggers equal the full trigger point.

**Approach**:
```
For k ∈ {2, 3, 4, 5, ...}:
    For each level i ∈ {1, ..., k-1}: R_i = R_trigger
    Simulate task sets under k-level scheme
    Record: JNE, TiD, WastedCPU, ServiceRatio
    
Compare k vs (k-1) levels using objective function
Select k that maximizes objective while keeping overhead acceptable
```

**Key Insight**: When `R_level_X = R_trigger` for all X < k, intermediate levels don't trigger earlier - they only change *which* tasks are dropped. Optimal design may be about **selective dropping strategy**, not trigger timing.

#### Task 3.2: Identify Optimal Number of Modes (R_level_X <= R_trigger)
**Objective**: Find optimal level count and trigger points when intermediate triggers can be less than R_trigger.

**Approach**:
1. **Parameter space**: For k levels, we have (k-1) trigger points to optimize
2. **Optimization strategy**:
   - Grid search for small k (k ≤ 4)
   - Genetic algorithm or simulated annealing for larger k
   - Reinforcement learning for very large k ( exploratory)

**Hypothesis**: The optimal number is small (2-4 levels) due to:
- Diminishing returns from additional thresholds
- Increased complexity without proportional benefit
- Potential for new failure modes

#### Task 3.3: Research Optimal Mode Finding Methods
**Objective**: Identify practical methods for finding good (not necessarily globally optimal) mode configurations.

**Methods to Evaluate**:

| Method | Pros | Cons |
|--------|------|------|
| **Exhaustive search** | Guaranteed optimal | Combinatorial explosion |
| **Grid search** | Simple, interpretable | Curse of dimensionality |
| **Genetic algorithm** | Handles large spaces well | No optimality guarantee |
| **Bayesian optimization** | Sample-efficient | Complex to implement |
| **Simulated annealing** | Good for continuous spaces | Slow convergence |
| **Reinforcement learning** | Adaptive policy learning | High sample complexity |

**Proposed Strategy**:
1. For k ≤ 4: Use grid search with adaptive resolution
2. For 4 < k ≤ 8: Use genetic algorithm with population size 50-100
3. For k > 8: Use Bayesian optimization with Gaussian process

**Deliverable**: `docs/mode_optimization.md` with:
- Recommended method(s) based on k range
- Implementation outline
- Validation procedure

---

### Phase 4: Drop Strategy Research

> **Status.** Task 4.1 is answered: Stage 3 of the revised protocol
> (`mode_optimization.tex` §"Bounded-Gain Configuration Selection") exhaustively bounds
> any drop-strategy search's gain over the default (utilisation-ordered shed-early)
> configuration at ≤0.79%, an order of magnitude under this study's 5% threshold — closing
> the comparison without needing to run the strategies enumerated below individually.
> Priority-based dropping specifically is *not* competitive (73–78% more shedding than
> necessary), which corrects §5 Q3 below. Task 4.2 (exit strategy) is now measured, not
> just piloted (`docs/exit_strategy_analysis.md`, `mode_optimization.tex` §"Stage 5"):
> `exit_policy="amc_rh"` in `simulation.multilevel` is proven safe for full exit
> (`safety_proof.md` Corollary 2) and gives +2% to +27% service ratio over today's
> idle-only exit, 11 of 16 cells practically significant — the largest exit-related effect
> this project has measured. It costs 20–87% more mode changes, tracked and reported
> alongside the gain rather than netted against it (no validated `Φ` implementation
> exists to net them against). Separately, `progressive`'s cascade question is closed with
> evidence: a real cascade mechanism could add at most 0.04–0.86 percentage points beyond
> full exit alone — not worth building.

#### Task 4.1: LO Task Assignment to Criticality Levels
**Objective**: Determine optimal assignment of low-criticality tasks to degradation levels.

**Questions**:
1. Should tasks be assigned to levels at generation time, or dynamically?
2. Should assignment consider task parameters (utilization, deadline tightness)?
3. Can we use priority as the drop ordering (inherent in FPPS)?

**Drop Strategy Options**:

| Strategy | Description | Complexity |
|----------|-------------|------------|
| **Priority-based** | Drop lowest-priority LO tasks first | O(1) per drop decision |
| **Utilization-based** | Drop highest-utilization LO tasks first | O(n log n) per level entry |
| **Deadline-based** | Drop tightest-deadline LO tasks first | O(n log n) per level entry |
| **Proportional** | Drop fraction f of each task's jobs | Requires job-level tracking |

#### Task 4.2: Exit Strategy Research
**Objective**: Determine optimal exit strategy from degraded modes.

**Exit Strategy Options**:

| Strategy | Description | Trade-offs |
|----------|-------------|------------|
| **Direct to L_0** | Exit immediately when trigger clears | Faster recovery, more oscillation |
| **Cascade exit** | Exit one level at a time | Smoother transition, slower recovery |
| **Hysteresis-based** | Require trigger < threshold - hysteresis to exit | Reduces oscillation, adds delay |
| **Adaptive exit** | Exit based on expected future triggers | Complex, requires prediction |

**Research Question**: Does hysteresis improve the objective function by reducing mode-switching overhead?

---

### Phase 5: Implementation & Validation

> **Status.** Tasks 5.1 and 5.3 are built, not just planned: the k-level engine
> (`amc_tasksim/simulation/multilevel.py`) and the paired sweep framework
> (`amc_tasksim/experiments/contract.py`, `sweep.py`) ran all four stages of the revised
> protocol. Task 5.2's validation suite is only partially confirmed against the specific
> checks below — see `docs/README.md`'s Phase 5 status note.

#### Task 5.1: Core Multi-Level Protocol
**Implementation Tasks**:
1. Extend `ModeChangeProtocol` interface to support multiple levels
2. Add `get_degradation_level()` method
3. Add `should_exit_level(level)` method
4. Track per-job degradation level state
5. Implement drop决策 logic for each level

#### Task 5.2: Validation Suite
**Tests to Run**:
1. **Correctness**: Reproduce two-mode results when k=2 matches existing AMC-RH
2. **Monotonicity**: More levels never increases JNE for same task set
3. **Consistency**: Objective function improvements across ensembles
4. **Edge cases**: k=1 (single mode), very large k, extreme utilizations

#### Task 5.3: Experiment Framework
**Sweep Parameters**:
- U ∈ [0.3, 0.5, 0.7, 0.8, 0.9] (key operating points)
- k ∈ {2, 3, 4, 5} (number of levels)
- Drop strategies: priority, utilization, deadline
- Exit strategies: direct, cascade, hysteresis

**Output**: Compare objective function values across configurations

---

## 4. Research Timeline & Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 2-3 weeks | Safety proof, task set spec, metrics |
| Phase 2 | 2-3 weeks | Objective function, complexity analysis |
| Phase 3 | 4-6 weeks | Optimal mode analysis, optimization methods |
| Phase 4 | 3-4 weeks | Drop strategies, exit strategy research |
| Phase 5 | 4-6 weeks | Implementation, validation, experiments |

**Total**: ~15-22 weeks for comprehensive study

---

## 5. Key Research Questions (Summary)

> **Status.** The answers below were the working hypotheses at project inception.
> Several are now known wrong or answered differently by the revised protocol
> (`../research/mode_optimization.tex`). Corrections are appended rather than silently
> edited in, per this project's convention (see `task_model.tex`, `safety_proof.md`).

### Primary Questions
1. **When is it meaningful to introduce additional modes?**
   - When the reduction in wasted CPU/service loss justifies complexity overhead
   - *Standing, refined*: additional *modes* (k) turned out not to be the lever at all
     (see Q2). The meaningful choice is drop *policy* (shed-early vs. progressive) and
     *operating point* — Stage 2's regime map shows these alone produce +2.4% to +27.3%
     over two-level AMC-RA.

2. **How many modes are meaningful?**
   - ~~Likely 2-4; diminishing returns beyond that~~
   - *Corrected*: under the adopted shed-early policy, $k$ is vacuous — results agree to
     six decimal places across $k \in \{2,3,4,5\}$ (`mode_optimization.md` §"Superseded").
     There is no "how many" question left to answer for this policy.

3. **How do we allocate low-criticality tasks to modes?**
   - ~~Priority-based drop is simplest and most natural for FPPS~~
   - *Corrected*: priority-based dropping sheds 73–78% more than necessary. The adopted
     default is utilisation-ordered shed-early (`scheduling/drop_sets.py`), and Stage 3
     bounds any alternative's further gain at ≤0.79%.

4. **How do we assign R_level_X to highest criticality tasks?**
   - ~~Monotonic sequence ≤ R_trigger; optimal spacing needs research~~
   - *Corrected*: the research ran (Stage 0/1 structural pilot) and found spacing has zero
     effect on the drop set under shed-early. Monotonic sequence is still required for the
     safety proof (`safety_proof.md` ladder properties), but there is no spacing to
     optimise.

5. **Does it make sense to assign R_level_X to other tasks?**
   - Later phase - potentially but adds complexity
   - *Standing, partially addressed*: `task_model.tex`'s "Mode Transition Protocol"
     extension lets LO-criticality tasks trigger a rung too; `safety_proof.md`'s
     Corollary 1' covers the safety consequences (graded, not full containment). Not
     revisited as an optimisation target.

6. **Can exit be staged?**
   - ~~Yes, with hysteresis providing smooth transitions~~
   - *Corrected, then answered*: this was originally asserted rather than researched.
     "Staged" in the sense of earlier, evidence-based exit (no intermediate level, just
     leaving degraded mode sooner) is now built, proven safe, and measured — +2% to +27%
     service ratio, at a real cost of 20–87% more mode changes
     (`docs/exit_strategy_analysis.md`). "Staged" in the sense of genuine intermediate-level
     cascade is closed the other way, with evidence rather than by continued absence of
     research: decomposing the same measurement shows a real cascade mechanism could add at
     most 0.04–0.86 percentage points beyond full exit alone, so it is not being built.
     Hysteresis (a *timed* hold-off, as originally meant here) remains genuinely
     unbuilt — `ModeChangeProtocol` still has no `exit_time()` — and is now a lower
     priority than when this question was written, given how much larger the plain
     evidence-cleared effect turned out to be.

### Secondary Questions
- Does multi-level scheduling reduce oscillation between modes?
- Can we prove bounds on service degradation vs. HI protection?
- What's the relationship between number of levels and statistical stability?

---

## 6. References

1. **AMC-RH**: "Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling", Bate et al., RTAS 2022
2. **AMC**: "Compensating Adaptive Mixed-Criticality Scheduling", Bate et al., RTNS 2022
3. **DRS**: "Generating Utilization Vectors for the Evaluation of Real-Time Scheduling Algorithms", Baruah et al., RTSS 2020

---

## 7. References and Related Documents

The formal task set model used in this research is documented separately:

- **[Task Set Model (LaTeX)](./task_set_model.tex)** - Complete formal specification with mathematical definitions
- **[Task Set Model (Markdown)](./task_set_model.md)** - Explained version with examples

See also the [[task-set-model]] memory entry.

---

## Appendix: Notation Reference

| Symbol | Meaning |
|--------|---------|
| `n` | Number of tasks |
| `k` | Number of degradation levels (≥ 2) |
| `R_i(LO)` | Response time for task i in normal mode |
| `R_level_X` | Trigger point for level X (X ∈ {1, ..., k-1}) |
| `JNE` | Jobs Not Executed (LO jobs dropped) |
| `TiD` | Time in Degraded mode (fraction) |
| `WastedCPU` | CPU cycles spent on abandoned LO work |
