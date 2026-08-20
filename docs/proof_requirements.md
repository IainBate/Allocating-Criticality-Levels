# Proof Requirements for Multi-Level Mixed-Criticality Scheduling

## Overview

This document identifies the necessary correctness proofs for the multi-level mixed-criticality scheduling framework and categorizes them by status.

## Proven Properties (Verified)

All four rest on the *severity ladder* trigger family and on drop sets being
**admissible** (see `safety_proof.md`). Under the earlier fraction-of-$R_i(LO)$
design two of them were false; the status column records what is established
now, and by what.

| Property | Status | Established by |
|----------|--------|----------------|
| HI tasks meet deadlines at every level | **Proven**, conditional on admissible drop sets | Theorem 1 + admissibility clause 1; drop sets computed by `scheduling/drop_sets.py`, tested in `tests/scheduling/test_drop_sets.py` |
| No LO job ever completes after its deadline | **Proven**, unconditionally | Corollary 3.1. True by construction under deadline termination — stronger than the old clause-2 argument, which held only for retained tasks and only when the drop set was admissible |
| LO task abandonment is monotonic with level | **Proven** | Lemma 1 — a property of the greedy construction (levels are prefixes of one shed sequence), not an assumption about the drop policy |
| JNE is bounded by AMC-RH performance, while only HI tasks may trigger | **Proven** | Corollary 1, *and only because of* ladder property (C): no level fires before $R_i(LO)$. Was false under the fraction design (measured: 356 abandoned jobs against AMC-RH's zero, with no fault present) |
| — the same, once LO tasks may also trigger a rung | **Retracted as unconditional, replaced by a graded bound** | Corollary 1' (`safety_proof.md`): an explicit counterexample shows containment fails once a LO task can pull the trigger (`task_model.tex`, "Mode Transition Protocol"). What survives: no-fault silence (a), ordered sacrifice (b), and containment *above* the deepest tier a LO task can trigger (c). Below that tier the comparison is favourable but reported empirically (JNC), not proven by containment |
| Time at the *deepest* level is bounded by AMC-RH's degraded time | **Proven** | Theorem 2 via property (C) |
| Total time at *any* degraded level is bounded by AMC-RH's | **Not claimed** | Shedding less can lengthen the drain, so TiD may exceed AMC-RH's. Measure it; do not argue it |

### Ladder properties these depend on

| Property | Status | Established by |
|----------|--------|----------------|
| (A) $R_i(0) = R_i(LO)$ exactly | **Verified** | `tests/scheduling/test_severity.py`, 40 task sets x 10 severities |
| (B) thresholds monotone in severity | **Verified** | same; requires $+\infty$ saturation for unreachable levels, without which it fails |
| (C) $R_i(\chi) \geq R_i(LO)$ always | **Verified** | same — the property the fraction design violated by construction |

## Assumed Properties (Needed for Framework)

### Schedulability Analysis
| Property | Description | Reason |
|----------|-------------|--------|
| **A.1**: AMC-rtb correctness | The response time analysis equations (1)-(2) correctly compute $R_i^{\mathrm{lo}}$ and $R_i^{\mathrm{hi}}$ | Foundation for determining when to trigger degradation levels |

### Task Model Assumptions
| Property | Description | Reason |
|----------|-------------|--------|
| **A.2**: Implicit deadlines ($D_i = T_i$) | Simplifies analysis; can be extended to constrained deadlines | Current implementation assumption |
| **A.3**: Preemptive FPPS | Priority-based preemption model | Basis for response time analysis |
| **A.4**: Sporadic task arrivals | Tasks arrive independently, periodic replications | Standard MC scheduling model |

### Execution Model Assumptions
| Property | Description | Reason |
|----------|-------------|--------|
| **A.5**: BCET $\leq$ actual execution $\leq C^{\mathrm{lo}}$ (normal) | Execution times bounded within expected range | Ensures predictable behavior |
| **A.6**: HI-behavior draws $U[C^{\mathrm{lo}}, C^{\mathrm{hi}}]$ | Captures worst-case when HI behavior occurs | Matches AMC-RH model |

## Open Proof Questions

> **Status as of the revised protocol** (`research/mode_optimization.tex`, "Revised
> Protocol" and "Stage Results"): three of the six rows below were written before that
> protocol ran and asked the wrong question, or a question the protocol answered by a
> different route than the one proposed here. Each is annotated rather than deleted, per
> this project's convention of recording a correction instead of silently rewriting it
> (see `task_model.tex` and `safety_proof.md`'s own corrections).

### Quality of Service Proofs
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.1**: Is multi-level scheduling *strictly better* than two-level? | Can we prove that expected JNE decreases as $k$ increases (for optimal drop policies)? | Medium - requires stochastic analysis of drop policies | **Answered empirically, not by proof.** The regime map (`mode_optimization.tex` §Stage 2) shows the scheme beats two-level AMC-RA in all 32 configurations tested, $+2.4\%$ to $+27.3\%$, 22/32 practically significant. This is at the tested grid points, not a general stochastic-ordering proof — see the two scope limits in `mode_optimization.tex` §"What This Protocol Does and Does Not Yet Establish". It is *not* driven by $k$: $k$ and trigger spacing were subsequently found to be vacuous (see O.2) — the effect comes from drop policy and operating point instead. |
| **O.1a**: Prove monotonic JNE for priority-based dropping | Show $\E[\mathrm{JNE}_k] \leq \E[\mathrm{JNE}_{k-1}]$ for the specific drop policy used | Medium - needs stochastic ordering proof | **Superseded — wrong policy.** Priority-based dropping is not the adopted default: it sheds 73–78% more than necessary against the utilisation-ordered shed-early policy (`drop_sets.py`, cited in `mode_optimization.md` §"Objective Function for Optimization"). Monotonicity of abandonment *with level* is already proven policy-independently by Lemma 1 (`safety_proof.md`) — what remains open is the stochastic-ordering claim *across $k$* for the adopted policy specifically, which is supported empirically (O.1) but not proven. |

### Optimality Guarantees
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.2**: What is the optimal trigger point spacing $\{R_1, \dots, R_{k-1}\}$? | For a given objective function, what spacing minimizes expected cost? | Hard - optimization over probability space | **Resolved: there is no optimisation problem here.** Under the shed-early policy, trigger spacing has *zero* effect on the drop set — confirmed 16/16 task sets, agreeing to six decimal places across $k \in \{2,3,4,5\}$ (`mode_optimization.md` §"Superseded"). The question was well-posed but the premise (spacing matters) is false for the adopted policy. |
| **O.2a**: Prove convexity of objective function in trigger spacings | Enables gradient-based optimization | Hard - requires analysis of probability integrals | **Moot.** The objective is constant, not merely convex, in trigger spacing under shed-early (see O.2) — there is nothing for a convexity proof to enable. |

### Complexity Bounds
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.3**: What is the competitive ratio of greedy drop policies? | How well does priority-based dropping perform vs. optimal offline? | Medium - approximation theory | **Empirically bounded, not formally proven.** Stage 3's exhaustive search (`mode_optimization.tex` §"Bounded-Gain Configuration Selection") bounds *any* method's gain over the adopted default at $\leq 0.79\%$ service ratio, everywhere tested — an upper bound by construction, since nothing outperforms exhaustive enumeration. A formal competitive-ratio proof would now only need to explain a small number. |
| **O.3a**: Tight bound for online multi-level scheduling | Compare online algorithm to clairvoyant offline optimal | Medium - adversarial analysis | **Open.** No formal proof exists; scope is now known to be small (O.3), which may make a loose bound sufficient rather than motivating a tight one. |

## Completed Work


### Metrics and Objective Function
| Component | Status | Reference |
|-----------|--------|-----------|
| Wasted CPU metric | **Defined** | docs/metrics_objective.md |
| Service Ratio metric | **Defined** | docs/metrics_objective.md |
| Level transitions metric | **Defined** | docs/metrics_objective.md |
| Weighted objective function | **Defined** | docs/metrics_objective.md |
| Adaptive weight selection | **Defined** | docs/metrics_objective.md |

### Meaningful Improvement Criteria (Task 2.2)
| Criterion | Status | Formal Expression |
|-----------|--------|-------------------|
| Service Preservation | **Defined** | $\E[\mathrm{JNE}_k] \leq \E[\mathrm{JNE}_{k-1}]$ |
| Waste Reduction | **Defined** | $\E[\mathrm{WastedCPU}_k + \mathrm{JNE}_k] \leq \E[\mathrm{WastedCPU}_{k-1} + \mathrm{JNE}_{k-1}]$ |
| Statistical Improvement | **Defined** | $\E_G[\Phi_k] < \E_G[\Phi_{k-1}]$ |

### Complexity Analysis (Task 2.3)
| Component | Baseline | Multi-Level | Reference |
|-----------|----------|-------------|-----------|
| Trigger check | O(n) | O(k × n) | docs/complexity_analysis.md |
| Drop decision | - | O(m) with priority-based | docs/complexity_analysis.md |
| State tracking | O(1) per job | +O(1) per level state | docs/complexity_analysis.md |
| Space overhead | O(n) | O(n) + O(k × n) | docs/complexity_analysis.md |

### Validation Procedures
| Procedure | Status | Description |
|-----------|--------|-------------|
| Single-task set monotonicity check | **Defined** | Verify all criteria for specific task sets |
| Ensemble hypothesis testing | **Defined** | t-test comparing k vs k-1 levels across N samples |
| Practical significance threshold | **Defined** | 5% improvement relative to baseline |

## Summary

| Category | Count |
|----------|-------|
| Proven (conditional on admissible drop sets) | 6 |
| Explicitly not claimed | 1 |
| Ladder properties verified by test | 3 |
| Assumed | 6 |
| Answered empirically (not by proof) | 2 (O.1, O.3) |
| Resolved / moot — premise was false | 2 (O.2, O.2a) |
| Superseded — asked about the wrong policy | 1 (O.1a) |
| Open, no answer at all | 1 (O.3a) |

**Status**: the safety argument is now carried by executable checks rather than
prose — ladder properties (A)-(C) and drop-set admissibility are enforced by
`tests/scheduling/`, so a regression fails the suite rather than surviving in a
document. Of the original six open questions, four have since been answered or
retired by the revised protocol (`mode_optimization.tex`) rather than by the
proof techniques originally proposed for them; only O.3a remains genuinely
open. See the annotated table above for what changed and why.

## Proof Framework for Remaining Questions

Only O.3a still lacks any answer, empirical or formal (see Summary). The frameworks below
are kept for the record, annotated with why each stopped being the live plan.

### For O.1 / O.1a (Strict Improvement)
```
Goal: E[JNE_k] ≤ E[JNE_{k-1}]

Approach:
1. Define stochastic model of HI behavior occurrences
2. Express JNE as integral over probability space
3. Show adding a level splits the integral into smaller pieces
4. Apply Jensen's inequality to convex cost function
```
Not pursued: the regime map answered the practical question empirically (O.1) before this
was attempted, and $k$ turned out not to be the lever that matters (O.2) — so a proof
indexed on $k$ would be proving the wrong variable.

### For O.2 (Optimal Trigger Spacing) — moot, kept for the record
```
Goal: argmin_{R_1≤...≤R_{k-1}} Φ(R_1,...,R_{k-1})

Approach:
1. Model trigger times as order statistics of HI-behavior events
2. Use dynamic programming for small k (k ≤ 4)
3. For larger k, apply convex optimization if objective is concave
```
Superseded: there is no argmin to find. Φ does not vary with trigger spacing under the
adopted shed-early policy (O.2), so this framework solves a problem that does not exist
for the policy actually adopted.

### For O.3 / O.3a (Competitive Ratio)
```
Goal: JNE_greedy / JNE_optimal ≤ c for some constant c

Approach:
1. Define optimal offline algorithm with full future knowledge
2. Use potential function method to compare states
3. Bound the difference in dropped tasks at each level transition
```
Still the right approach for O.3a specifically. Its urgency is lower than when written,
since Stage 3's exhaustive bound already shows $c$ is close to 1 (gain $\leq 0.79\%$) at
every point tested — a formal proof would confirm a small number, not discover a large one.

## Implementation Verification Checklist

Before deploying multi-level scheduling, verify:

- [ ] **V.1**: HI-criticality response time analysis matches paper (AMC-RH Appendix A)
- [ ] **V.2**: Trigger point $R_x$ correctly computed from busy period start times
- [ ] **V.3**: Drop policy at level $L_x$ produces subset of drops at $L_{x+1}$
- [x] **V.4**: Exit condition for $L_{k-1}$ matches AMC-RH (idle instant) — done for
      `exit_policy="idle"` (unchanged shipped behaviour) and, additionally, verified
      bit-identical to `engine.py`'s `AMC_RH` for `exit_policy="amc_rh"` at k=2
      (`test_k2_full_drop_amc_rh_exit_reproduces_amc_rh_exactly`). Proven safe by
      `safety_proof.md`'s Corollary 2, scoped to full exit only.
- [ ] **V.5**: No priority inversion in intermediate levels
- [ ] **V.6**: BCET constraints maintained after level transitions

> **No longer blocked — a timed exit rule is built too.** `safety_proof.md` (Theorem 2 scope
> note, superseded by Corollary 3) distinguished evidence-cleared exit (event-triggered, exact
> without `exit_time()`) from a hysteresis/hold-off rule (time-triggered, needing a genuine
> scheduled-event mechanism). Both are now implemented: `exit_policy="hysteresis"` in
> `simulation.multilevel` adds `_next_evidence_reappearance` — an extension of the existing
> next-event lookahead, not a new `exit_time()` abstraction as originally anticipated — so the
> loop cannot skip past evidence reappearing between events. Proven safe for any `hold_off`
> (Corollary 3): every exit is gated on a *fresh* check at the instant it fires, so the
> lookahead's precision affects only how faithfully `hold_off` is honoured, never safety.
> `docs/exit_strategy_analysis.md` (Result 4) sweeps it for real: a genuine sweet spot exists
> between the service-ratio gain and the oscillation cost, and it shifts with utilisation —
> capping the oscillation increase at ≤10% keeps 28–43% of the gain at U=0.6–0.7 but only
> ~10.5% at U=0.9. V.5/V.6 above should extend to cover `exit_policy="hysteresis"` explicitly,
> not just `"idle"`/`"amc_rh"`.


