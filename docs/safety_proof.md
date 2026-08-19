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
- Thresholds are ordered by property **(B)**: $R_i(\chi_1) \leq \dots \leq R_i(\chi_{k-1})$, with $R_i(\chi_1) = R_i(\mathrm{LO})$ by property **(A)**

## Key Lemma

**Lemma 1 (Monotonic Drop Set)**. For $\chi_x \leq \chi_y$, $S_x \subseteq S_y$.

*Proof*: A drop set is built by shedding tasks, one at a time, from the full
LO-criticality set until admissible, choosing the next by an *ordering* --
a function of the task set and the remaining candidates. Every level begins
from the same full candidate set, so the ordering yields the *same sequence* of
tasks at every severity. Raising $\chi$ only increases interference, so any set
admissible at $\chi_y$ is also admissible at $\chi_x \leq \chi_y$; the shedding
therefore stops no later at $\chi_x$ than at $\chi_y$ along that shared
sequence. Each level is a prefix of one sequence, and prefixes nest. $\square$

This is a property of the construction, not an assumption about it, and it is
checked in `tests/scheduling/test_drop_sets.py`. It fails only for an ordering
that consults the severity or the partial drop set, which is why
`drop_ladder()` carries each level's set forward as a safeguard.

## Theorem 1: HI Tasks Meet Deadlines

**Statement**: If all HI-criticality tasks meet their deadlines under AMC-RH, they also meet them in the multi-level scheme with severities $0 = \chi_1 \leq \dots \leq \chi_{k-1}$ and admissible drop sets.

*Proof*: 

1. By definition, under AMC-RH, an HI-criticality task $\tau_i$ enters degraded mode when it reaches $R_i^{\mathrm{lo}}$ from its busy period start.

2. In the multi-level scheme:
   - Level $L_0$ (normal) is identical to AMC-RH normal mode
   - By property **(A)**, $L_1$ triggers at $R_i(0) = R_i(\mathrm{LO})$, which is exactly AMC-RH's trigger, so protection begins at the same instant, never later.

3. Once in degraded mode, $\tau_i$ continues to execute with budget $C_i^{\mathrm{hi}}$, which by assumption allows it to meet its deadline.

4. Deeper levels trigger later (property **(B)**) and shed more (Lemma 1), so each provides at least the protection of the level below it. Clause 1 of admissibility discharges the deadline obligation at every level.

Therefore, HI-criticality tasks meet their deadlines in the multi-level scheme. $\square$

## Theorem 2: No Worse Than AMC-RH

**Statement**: On the same arrival sequence and execution times, the time spent
at the deepest level $L_{k-1}$ is no greater than the time AMC-RH spends in
degraded mode.

*Proof*: AMC-RH enters degraded mode when a HI-criticality job reaches
$R_i(\mathrm{LO})$. The multi-level scheme enters $L_{k-1}$ when a job reaches
$R_i(\chi_{k-1})$, and by property **(C)** $R_i(\chi_{k-1}) \geq
R_i(\mathrm{LO})$. So every instant at which the scheme is at $L_{k-1}$ is an
instant at which AMC-RH is already degraded, given a common exit rule. The set of
$L_{k-1}$ instants is contained in the set of AMC-RH degraded instants, so its
measure is no greater. $\square$

> **Scope.** This bounds occupancy of the *deepest* level only. Total time spent
> at *any* degraded level is **not** bounded by AMC-RH's, and should not be
> claimed: $L_1$ fires at exactly AMC-RH's trigger (property **(A)**), and
> shedding less means the system can take longer to drain, so it may remain
> degraded longer. That is the real cost of grading the response, and it is a
> quantity to measure (TiD), not to bound by argument.
>
> The exit rule also has to be stated, because the containment above assumes
> both schemes leave on the same condition. Note further that an exit rule with
> a *temporal* component (a hysteresis or hold-off) is not currently expressible
> exactly in the simulator: `ModeChangeProtocol` exposes `entry_time()` as a
> scheduled instant but only a `should_exit()` predicate, which is sampled at
> whatever event the loop reaches next. For the three shipped protocols this is
> exact, because their exit conditions can only become true when the run-queue
> shrinks — itself an event. A hysteresis rule breaks that and inflates
> `degraded_ticks` by roughly 15–20%. Adding `exit_time()` is a prerequisite for
> any k-level demotion policy with a timer.

## Corollary 1: JNE Bound

**Statement**: The number of LO jobs not executed (JNE) in the multi-level scheme is at most that of AMC-RH.

*Proof*: Compare the two schemes on the same arrival sequence and the same
execution times.

1. By property **(C)**, no level of the ladder fires before $R_i(\mathrm{LO})$,
   which is AMC-RH's trigger. So at every instant $t$ at which the multi-level
   scheme is abandoning LO-criticality releases, AMC-RH is abandoning them too.

2. By property **(A)**, $L_1$ fires exactly at $R_i(\mathrm{LO})$, so the
   converse also holds: the two schemes begin shedding at the same instant.

3. At any such instant the multi-level scheme abandons only releases of tasks in
   $S_x \subseteq \mathrm{LO}$, whereas AMC-RH abandons releases of every
   LO-criticality task.

So at every instant the multi-level drop set is a subset of AMC-RH's, and the
abandoned-job count is a pointwise-dominated sum. $\square$

> **This corollary was previously false and is now true only because of the
> ladder.** Under the earlier fraction-based triggers, step 1 fails: levels
> fired *below* $R_i(\mathrm{LO})$, so the scheme abandoned jobs at instants
> when AMC-RH was still running everything, and JNE went **up**, not down —
> measured at 356 abandoned jobs against AMC-RH's zero, with no fault present.
> Property (C) is precisely what step 1 needs.

## Theorem 3: Schedulability Preservation

The previous statement of this theorem argued that "no LO task experiences
*more* interference than in the two-level case". That is **false**: a
LO-criticality task retained at an intermediate level runs alongside
HI-criticality jobs charged up to $C^{\mathrm{hi}}$, whereas two-level AMC would
have abandoned it. Retaining work necessarily means absorbing interference that
abandoning it would have avoided.

The repair is not to weaken the claim but to *impose it*. Interference is not a
free variable: it is determined by the drop set, and the drop set is ours to
choose. So the property becomes an obligation on $S_x$, discharged by
response-time analysis at design time.

**Definition (admissible drop set).** $S_x \subseteq \mathrm{LO}$ is *admissible*
at severity $\chi_x$ if, charging every task $C_i(\chi_x)$, treating tasks in
$S_x$ as absent and retained LO-criticality tasks at $C^{\mathrm{lo}}$:

1. every HI-criticality task $\tau_i$ has $R_i \leq D_i$, and
2. every retained LO-criticality task $\tau_j \notin S_x$ has $R_j \leq D_j$.

**Statement**: If every level's drop set is admissible, then at every level no
HI-criticality job and no *retained* LO-criticality job misses its deadline.

*Proof*: Both clauses are exactly the response-time conditions of the standard
fixed-priority test applied at level $L_x$'s budgets and active task set. Level
entry only ever removes tasks (Lemma 1), so the interference each retained task
faces at run time is bounded by the interference the analysis charged it.
Clause 1 gives the HI-criticality obligation; clause 2 gives the retained
LO-criticality obligation. A task in $S_x$ has no deadline obligation: its jobs
are abandoned on release and count toward JNE, not LDM. $\square$

**Corollary 3.1 (no LDM).** LDM $= 0$ by construction, so there is no
JNE-against-LDM trade-off to measure. The scheme trades JNE against *severity
coverage*, not against missed deadlines.

**Existence.** An admissible set always exists whenever the task set passes
AMC-rtb, since $S_x = \mathrm{LO}$ reduces clause 1 to the AMC-rtb HI-mode test
and vacates clause 2. So the question is never *whether* a level is feasible,
only how little must be shed — which is what makes the drop set an optimisation
rather than a guess.

### Measured cost

Minimal admissible sets over 50 AMC-rtb-schedulable, non-trivial task sets
($n=12$, $U=0.7$, $CF=2$), computed by `amc_tasksim.scheduling.drop_sets`:

| Severity $\chi$ | LO tasks that must be shed |
|---|---|
| $\leq 0.25$ | **0.0%** |
| 0.50 | 10.0% |
| 0.75 | 33.7% |
| 1.00 | **56.3%** |

Two-level AMC abandons every new LO-criticality release at its single trigger,
i.e. the 100% row. Mild overruns need no degradation at all, and even at full
$C^{\mathrm{hi}}$ behaviour the scheme retains 43.7% of LO-criticality tasks
that AMC-RH would have abandoned — with a proof that they still meet their
deadlines. **This is the quantitative case for grading the response.**

> **A note on a stronger alternative.** One could instead require that no
> retained LO task sees more interference than in *normal mode*
> ($R_j \leq R_j(\mathrm{LO})$ rather than $R_j \leq D_j$). That is achievable
> and would make the original wording true, but it is far stricter than
> necessary: it forces shedding 80–88% of LO-criticality tasks, against 0–56%
> for the deadline criterion. Deadline preservation is the property that
> matters, and demanding interference preservation discards most of the benefit
> for no additional safety.

## Practical Implications

1. **Trigger Point Selection**: intermediate levels sit at or *above* $R_i(\mathrm{LO})$, never below it. A threshold below $R_i(\mathrm{LO})$ fires under ordinary load and costs LO-criticality work with no fault present; property (C) forbids it.

2. **Drop Strategy**: The monotonicity property allows simpler implementation - each level can just drop additional tasks rather than computing a new set from scratch.

3. **Exit Policy**: To maintain the "no worse than AMC-RH" guarantee, exit from fully-degraded mode should use the same idle-instant criterion as original AMC.

## References

- AMC-RH Section IV: Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling
- Theorem 1 corresponds to the "by construction" comment in Appendix A of AMC-RH
- Corollary 1 follows from the monotonic drop set property established above
