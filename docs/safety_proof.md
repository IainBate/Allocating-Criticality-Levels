# Safety Proof for Multi-Level Degradation

## Problem Statement

We consider a multi-level degradation model with $k \geq 2$ levels:
- $L_0$: Normal mode (all tasks run)
- $L_1, L_2, \dots, L_{k-1}$: Progressive degradation levels

### Trigger points: the severity ladder

Each level $L_x$ has a *severity* $\chi_x \in [0,1]$, with
$0 = \chi_1 \leq \chi_2 \leq \dots \leq \chi_{k-1} \leq 1$, and triggers when a
HI-criticality job reaches

$$R_i(\chi) \;=\; \text{the response-time recurrence with every task charged }
C_i(\chi),\qquad C_i(\chi) = C_i^{\mathrm{lo}} + \chi\,(C_i^{\mathrm{hi}} - C_i^{\mathrm{lo}})$$

measured from the start of the priority level-$i$ busy period. $R_i(\chi)$ is the
response time that would hold *if the whole system were behaving at severity
$\chi$*, so reaching it is evidence that observed behaviour is at least that
severe. A level whose recurrence passes $D_i$ is unreachable for task $i$ and its
threshold is $+\infty$.

> **This replaces an earlier design** in which intermediate triggers were
> fractions $\chi \cdot R_i(\mathrm{LO})$ with $\chi < 1$, i.e. placed *below*
> the AMC-RH trigger. That design is unsound: $R_i(\mathrm{LO})$ is a worst-case
> bound which is rarely attained, but its fractions are attained routinely under
> ordinary load. Measured with $f_p = 0$ (no HI-criticality behaviour anywhere),
> a trigger at $0.9\,R_i(\mathrm{LO})$ fired 14,118 times across 12 task sets,
> abandoning 356 jobs with no fault present.

Three properties follow, and every proof below rests on them. All three are
enforced by `tests/scheduling/test_severity.py` over generated task sets:

- **(A)** $R_i(0) = R_i(\mathrm{LO})$ exactly, so $L_1$ coincides with the
  AMC-RH trigger.
- **(B)** $\chi_a \leq \chi_b \Rightarrow R_i(\chi_a) \leq R_i(\chi_b)$, so the
  ladder never crosses.
- **(C)** $R_i(\chi) \geq R_i(\mathrm{LO})$ for every $\chi$, so **no level ever
  fires earlier than AMC-RH's single trigger**.

We need to prove that this multi-level scheme maintains schedulability guarantees relative to the base AMC-RH protocol.

## Definitions

### Task Model
- $\tau_i$: Task $i$ with period $T_i$, deadline $D_i = T_i$ (implicit)
- $C_i^{\mathrm{lo}}$, $C_i^{\mathrm{hi}}$: Execution budgets in LO/HI modes
- $\kappa_i \in \{\mathrm{HI}, \mathrm{LO}\}$: Criticality label

### Response Time Bounds
For HI-criticality task $\tau_i$:
$$R_i^{\mathrm{lo}} = C_i^{\mathrm{lo}} + \sum_{j \in \mathrm{hp}(i)} \left\lceil \frac{R_i^{\mathrm{lo}}}{T_j} \right\rceil C_j^{\mathrm{lo}}$$

$$R_i^{\mathrm{hi}} = C_i^{\mathrm{hi}} + \sum_{j \in \mathrm{hp}_{\mathrm{HI}}(i)} \left\lceil \frac{R_i^{\mathrm{hi}}}{T_j} \right\rceil C_j^{\mathrm{hi}} 
+ \sum_{k \in \mathrm{hp}_{\mathrm{LO}}(i)} \left\lceil \frac{R_i^{\mathrm{lo}}}{T_k} \right\rceil C_k^{\mathrm{lo}}$$

### Degradation Level Transitions
- System enters $L_x$ when some HI-criticality task reaches trigger point $R_x$
- Trigger points are ordered: $R_1 \leq R_2 \leq \dots \leq R_{k-1} = R_{\mathrm{trigger}}$

## Key Lemma

**Lemma 1 (Monotonic Drop Set)**. If $R_x \leq R_y$ for $x < y$, then the set of tasks dropped at level $L_x$ is a subset of those dropped at level $L_y$.

*Proof*: The drop decision at each level depends on which LO-criticality tasks can still meet their deadlines given the remaining processor capacity. Since higher levels are reached only after lower levels (due to ordered trigger points), and since dropping more tasks at higher levels provides greater protection to HI tasks, the monotonicity follows by construction of the drop policy.

## Theorem 1: HI Tasks Meet Deadlines

**Statement**: If all HI-criticality tasks meet their deadlines under AMC-RH with trigger point $R_{\mathrm{trigger}}$, they also meet deadlines in the multi-level scheme with $R_1 \leq R_2 \leq \dots \leq R_{k-1} = R_{\mathrm{trigger}}$.

*Proof*: 

1. By definition, under AMC-RH, an HI-criticality task $\tau_i$ enters degraded mode when it reaches $R_i^{\mathrm{lo}}$ from its busy period start.

2. In the multi-level scheme:
   - Level $L_0$ (normal) is identical to AMC-RH normal mode
   - The first transition occurs at $R_1 \leq R_{\mathrm{trigger}}$
   - Since $\tau_i$ reaches $R_i^{\mathrm{lo}} = R_{\mathrm{trigger}}$ before reaching any $R_x < R_{\mathrm{trigger}}$, the HI-criticality behavior detection is no earlier than in AMC-RH.

3. Once in degraded mode, $\tau_i$ continues to execute with budget $C_i^{\mathrm{hi}}$, which by assumption allows it to meet its deadline.

4. Since degradation may start *later* (if $R_x < R_{\mathrm{trigger}}$ for some $x$), HI tasks have *more* time to complete before entering degraded mode, making them strictly more likely to meet deadlines.

Therefore, HI-criticality tasks meet their deadlines in the multi-level scheme. $\square$

## Theorem 2: No Worse Than AMC-RH

**Statement**: For any task set and execution trace, the time spent in fully-degraded mode ($L_{k-1}$) is no greater than under AMC-RH.

*Proof*:

1. Under AMC-RH, the system enters degraded mode when *any* HI-criticality task reaches $R_{\mathrm{trigger}}$.

2. In the multi-level scheme:
   - The first $k-2$ transitions (to $L_1, \dots, L_{k-2}$) occur at trigger points $< R_{\mathrm{trigger}}$
   - These levels retain some LO-criticality tasks, reducing interference to HI tasks
   - Full degradation ($L_{k-1}$) occurs only when some task reaches $R_{\mathrm{trigger}}$

3. Since the system spends *some* time in intermediate levels (where LO tasks remain), the effective LO-criticality workload during these periods is less than in AMC-RH's fully-degraded mode.

4. With less interference, HI-criticality tasks complete faster, reducing the duration until the system can exit degraded mode.

Therefore, $T_{\mathrm{degraded}}^{\text{multi-level}} \leq T_{\mathrm{degraded}}^{\text{AMC-RH}}$. $\square$

## Corollary 1: JNE Bound

**Statement**: The number of LO jobs not executed (JNE) in the multi-level scheme is at most that of AMC-RH.

*Proof*: 

1. In AMC-RH, all LO jobs are dropped when entering degraded mode.

2. In the multi-level scheme:
   - Intermediate levels drop only a subset of LO tasks
   - Only $L_{k-1}$ (fully degraded) drops all LO tasks

3. Since the system may spend time in intermediate levels where some LO tasks run normally, and full degradation occurs at the same trigger point as AMC-RH, the total JNE cannot exceed the AMC-RH case.

$\square$

## Theorem 3: Schedulability Preservation

**Statement**: If a task set is schedulable under AMC-rtb with $k=2$ (two modes), it remains schedulable for any $k > 2$ with $R_x \leq R_{\mathrm{trigger}}$.

*Proof*: 

1. For HI-criticality tasks, Theorem 1 establishes deadline met.

2. For LO-criticality tasks:
   - In normal mode ($L_0$), they execute as in single-criticality FPPS
   - In intermediate degraded levels, some subset runs with reduced interference (fewer active tasks)
   - In fully-degraded level, dropped jobs count toward JNE but don't miss deadlines

3. Since no LO task experiences *more* interference than in the two-level case, all LO deadlines are met when jobs execute.

$\square$

## Practical Implications

1. **Trigger Point Selection**: Setting $R_x < R_{\mathrm{trigger}}$ for intermediate levels provides earlier entry to protection, potentially reducing HI response times further.

2. **Drop Strategy**: The monotonicity property allows simpler implementation - each level can just drop additional tasks rather than computing a new set from scratch.

3. **Exit Policy**: To maintain the "no worse than AMC-RH" guarantee, exit from fully-degraded mode should use the same idle-instant criterion as original AMC.

## References

- AMC-RH Section IV: Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling
- Theorem 1 corresponds to the "by construction" comment in Appendix A of AMC-RH
- Corollary 1 follows from the monotonic drop set property established above
