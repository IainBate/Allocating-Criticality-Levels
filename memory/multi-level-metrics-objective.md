---
name: multi-level-metrics-objective
description: Metrics and objective function for multi-level scheduling
metadata:
  type: reference
---

# Multi-Level Scheduling Metrics and Objective Function

## Metrics (docs/metrics_objective.md)

### Wasted CPU
Measures CPU cycles spent on LO tasks that were later abandoned:

$$\mathrm{WastedCPU} = \sum_{i \in \tau^{\mathrm{LO}}} \sum_{j=1}^{\infty} 
\left( \min(\text{executed}_{ij}, C_i^{\mathrm{lo}}) \cdot \Ind[\text{job } j \text{ abandoned}] \right)$$

### Service Ratio
Fraction of LO-criticality work that completes successfully:

$$\mathrm{ServiceRatio} = \frac{\sum_{i \in \tau^{\mathrm{LO}}} (\text{releases}_i - \text{drops}_i)}{\sum_{i \in \tau^{\mathrm{LO}}} \text{releases}_i}
= 1 - \frac{\text{JNE}}{\text{Total LO releases}}$$

### Level Transition Count
Number of times the system changes degradation levels:

$$\mathrm{LevelTrans} = \sum_{t} \Ind[\text{level}(t-1) \neq \text{level}(t)]$$

## Objective Function (Weighted Cost Minimization)

$$\Phi = \alpha(U) \cdot \E[\mathrm{JNE}] + \beta(U) \cdot \E[\mathrm{TiD}] + \gamma \cdot \E[\mathrm{WastedCPU}]$$

Weights adapt to total utilisation $U$:

| Utilisation | $\alpha$ | $\beta$ | $\gamma$ |
|-------------|----------|---------|----------|
| $U < 0.5$ | 1.0 | 0.5 | 0.1 |
| $0.5 \leq U < 0.7$ | 0.8 | 0.8 | 0.2 |
| $0.7 \leq U < 0.85$ | 0.5 | 1.0 | 0.3 |
| $U \geq 0.85$ | 0.3 | 1.0 | 0.5 |

## Meaningful Improvement Criteria

### Criterion 1: Service Preservation
$$\E[\mathrm{JNE}_k] \leq \E[\mathrm{JNE}_{k-1}]$$
Adding a level should never increase lost LO work.

### Criterion 2: Waste Reduction
$$\E[\mathrm{WastedCPU}_k + \mathrm{JNE}_k] \leq \E[\mathrm{WastedCPU}_{k-1} + \mathrm{JNE}_{k-1}]$$
Total "bad" work should not increase.

### Criterion 3: Statistical Improvement
$$\E_{G}[\Phi_k] < \E_{G}[\Phi_{k-1}]$$
On average across random task sets, multi-level is better.

## Validation Procedure

1. **Single-task set**: Verify all three criteria hold for specific task sets
2. **Ensemble test**: Generate $N$ task sets at utilisation $U$, run for each $k \in \{2,3,4,5\}$, apply t-test
3. **Threshold check**: Require $(\mu_{k-1} - \mu_k)/\mu_{k-1} > 0.05$ (5% improvement)

## References

- docs/metrics_objective.md: Full derivation and implementation notes
- docs/complexity_analysis.md: Overhead analysis showing O(k×n) trigger check, O(m) drop decision
- docs/safety_proof.md: Proven properties that multi-level is no worse than AMC-RH
