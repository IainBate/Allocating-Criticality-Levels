# Proof Requirements for Multi-Level Mixed-Criticality Scheduling

## Overview

This document identifies the necessary correctness proofs for the multi-level mixed-criticality scheduling framework and categorizes them by status.

## Proven Properties (Verified)

| Property | Status | Reference |
|----------|--------|-----------|
| HI tasks meet deadlines when $R_x \leq R_{\mathrm{trigger}}$ | **Proven** | Theorem 1 in `safety_proof.md` |
| System never enters worse state than AMC-RH | **Proven** | Theorem 2 in `safety_proof.md` |
| LO task abandonment is monotonic with level | **Proven** | Lemma 1 in `safety_proof.md` |
| JNE is bounded by AMC-RH performance | **Proven** | Corollary 1 in `safety_proof.md` |

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

### Quality of Service Proofs
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.1**: Is multi-level scheduling *strictly better* than two-level? | Can we prove that expected JNE decreases as $k$ increases (for optimal drop policies)? | Medium - requires stochastic analysis of drop policies | Partially addressed by Criterion 1 in metrics_objective.md |
| **O.1a**: Prove monotonic JNE for priority-based dropping | Show $\E[\mathrm{JNE}_k] \leq \E[\mathrm{JNE}_{k-1}]$ for the specific drop policy used | Medium - needs stochastic ordering proof | **Open** |

### Optimality Guarantees
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.2**: What is the optimal trigger point spacing $\{R_1, \dots, R_{k-1}\}$? | For a given objective function, what spacing minimizes expected cost? | Hard - optimization over probability space | Addressed via adaptive weights in metrics_objective.md |
| **O.2a**: Prove convexity of objective function in trigger spacings | Enables gradient-based optimization | Hard - requires analysis of probability integrals | **Open** |

### Complexity Bounds
| Question | Description | Difficulty | Status |
|----------|-------------|------------|--------|
| **O.3**: What is the competitive ratio of greedy drop policies? | How well does priority-based dropping perform vs. optimal offline? | Medium - approximation theory | Related to complexity_analysis.md |
| **O.3a**: Tight bound for online multi-level scheduling | Compare online algorithm to clairvoyant offline optimal | Medium - adversarial analysis | **Open** |

## Completed Work

### Proven Properties (Verified)
| Property | Status | Reference |
|----------|--------|-----------|
| HI tasks meet deadlines when $R_x \leq R_{\mathrm{trigger}}$ | **Proven** | Theorem 1 in `safety_proof.md` |
| System never enters worse state than AMC-RH | **Proven** | Theorem 2 in `safety_proof.md` |
| LO task abandonment is monotonic with level | **Proven** | Lemma 1 in `safety_proof.md` |
| JNE is bounded by AMC-RH performance | **Proven** | Corollary 1 in `safety_proof.md` |

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
| Proven | 4 |
| Assumed | 6 |
| Open | 5 |
| Completed (Tasks 2.1-2.3) | 8 |

**Status**: Implementation framework complete with formal definitions for meaningful improvement criteria and complexity analysis. The remaining open proofs can be addressed incrementally as research extensions.

## Proof Framework for Remaining Questions

### For O.1 (Strict Improvement)
```
Goal: E[JNE_k] ≤ E[JNE_{k-1}]

Approach:
1. Define stochastic model of HI behavior occurrences
2. Express JNE as integral over probability space
3. Show adding a level splits the integral into smaller pieces
4. Apply Jensen's inequality to convex cost function
```

### For O.2 (Optimal Trigger Spacing)
```
Goal: argmin_{R_1≤...≤R_{k-1}} Φ(R_1,...,R_{k-1})

Approach:
1. Model trigger times as order statistics of HI-behavior events
2. Use dynamic programming for small k (k ≤ 4)
3. For larger k, apply convex optimization if objective is concave
```

### For O.3 (Competitive Ratio)
```
Goal: JNE_greedy / JNE_optimal ≤ c for some constant c

Approach:
1. Define optimal offline algorithm with full future knowledge
2. Use potential function method to compare states
3. Bound the difference in dropped tasks at each level transition
```

## Implementation Verification Checklist

Before deploying multi-level scheduling, verify:

- [ ] **V.1**: HI-criticality response time analysis matches paper (AMC-RH Appendix A)
- [ ] **V.2**: Trigger point $R_x$ correctly computed from busy period start times
- [ ] **V.3**: Drop policy at level $L_x$ produces subset of drops at $L_{x+1}$
- [ ] **V.4**: Exit condition for $L_{k-1}$ matches AMC-RH (idle instant)
- [ ] **V.5**: No priority inversion in intermediate levels
- [ ] **V.6**: BCET constraints maintained after level transitions

## Summary

| Category | Count |
|----------|-------|
| Proven | 4 |
| Assumed | 6 |
| Open | 3 |

**Recommendation**: Proceed with implementation using the proven properties as safety guarantees. The open questions can be addressed incrementally as research extensions.
