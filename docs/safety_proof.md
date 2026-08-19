# Safety Proof for Multi-Level Degradation

## Problem Statement

We consider a multi-level degradation model with $k \geq 2$ levels:
- $L_0$: Normal mode (all tasks run)
- $L_1, L_2, \dots, L_{k-1}$: Progressive degradation levels

### Trigger points: the severity ladder

Each level $L_x$ has a *severity* $\chi_x \in [0,1]$, with
$0 = \chi_1 \leq \chi_2 \leq \dots \leq \chi_{k-1} \leq 1$, and triggers when a
job that the level would *protect* reaches

$$R_i(\chi) \;=\; \text{the response-time recurrence with every task charged }
C_i(\chi),\qquad C_i(\chi) = C_i^{\mathrm{lo}} + \chi\,(C_i^{\mathrm{hi}} - C_i^{\mathrm{lo}})$$

measured from the start of the priority level-$i$ busy period. $R_i(\chi)$ is the
response time that would hold *if the whole system were behaving at severity
$\chi$*, so reaching it is evidence that observed behaviour is at least that
severe. A level whose recurrence passes $D_i$ is unreachable for task $i$ and its
threshold is $+\infty$.

**Who may pull the trigger.** Rung $x$ admits $\tau_i$ as a trigger when
$\ell_i \geq x$ (the rung does not abandon $\tau_i$) *and* either
$\kappa_i = \mathrm{HI}$ or $x \leq x_{\mathrm{LO}}$. The parameter
$x_{\mathrm{LO}}$ is the deepest rung a LO-criticality task may fire. At
$x_{\mathrm{LO}} = 0$ this is the classic rule (HI-criticality tasks only).
Above $0$, a LO-criticality task uses its *own* $R_i(\chi_x)$, computed
identically — the soundness argument is unchanged, since if every task had
complied with $C(\chi_x)$ then $\tau_i$ would have completed by
$R_i(\chi_x)$. Properties (A), (B) and (C) below hold verbatim for
LO-criticality tasks (checked: zero monotonicity violations across all tasks).

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
+ \sum_{k \in \mathrm{hp}_{\mathrm{LO}}(i)} \min\!\left( \left\lceil \frac{R_i^{\mathrm{hi}}}{T_k} \right\rceil, \left\lceil \frac{R_i^{\mathrm{lo}}}{T_k} \right\rceil \right) C_k^{\mathrm{lo}}$$

The $\min$ is a sound tightening of the canonical AMC-rtb form: interference
over a window of length $R_i^{\mathrm{hi}}$ cannot exceed
$\lceil R_i^{\mathrm{hi}}/T_k \rceil$ jobs however early the switch occurs.
This matches `amc_tasksim.scheduling.amc_rtb` exactly.

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

> **RETRACTED as stated, and replaced by Corollary 1' below.** The original
> statement — "the number of LO jobs not executed (JNE) in the multi-level
> scheme is at most that of AMC-RH" — is true only while HI-criticality tasks
> alone may pull the trigger. Once a LO-criticality task may demand a rung
> (`task_model.tex`, "Mode Transition Protocol"), it is **false**, and the
> counterexample below is explicit rather than hypothetical.

### The counterexample

| Task | $\kappa_i$ | Priority | $T_i = D_i$ | $C_i^{\mathrm{lo}}$ | $C_i^{\mathrm{hi}}$ |
|---|---|---|---|---|---|
| $\tau_1$ | LO | 0 (high) | 10 | 3 | — |
| $\tau_2$ | HI | 1 | 20 | 2 | 4 |
| $\tau_3$ | LO | 2 (low) | 40 | 5 | — |

$R_2^{\mathrm{lo}} = 5$, $R_3^{\mathrm{lo}} = 10$. Release $\tau_2, \tau_3$
at $t = 0$ and $\tau_1$ at $t = 5$ (its previous job ran in $[-5,-2)$, so the
processor idles at $t=-1$ and $\tau_3$'s busy period genuinely starts at
$t=0$). Let $\tau_2$ execute its full $C_2^{\mathrm{hi}} = 4$:

```
t:  0 1 2 3 4 5 6 7 8 9 10 11 12 ...
    2 2 2 2 3 1 1 1 3  3  3  3
```

- $\tau_2$ **overruns** — it executes twice its normal budget — yet completes
  at $t = 4$, *before* its own trigger $R_2^{\mathrm{lo}} = 5$. It beat the
  bound because $R_2^{\mathrm{lo}}$ budgets for interference from $\tau_1$
  that this release did not receive. **AMC-RH therefore never degrades.**
- $\tau_3$ has executed 3 of its 5 ticks at $t = 10 = R_3^{\mathrm{lo}}$, so
  it *would* fire a rung — at an instant when AMC-RH is running everything.

Step 1 of the original proof ("at every instant the multi-level scheme is
abandoning releases, AMC-RH is abandoning them too") fails here. Note the
failure is at **every** tier, not only the lowest, so no per-tier restatement of
containment survives unconditionally either.

## Corollary 1': Graded JNE

Let $\ell_i$ be $\tau_i$'s allocated criticality level and let
$x_{\mathrm{LO}}$ be the deepest rung a LO-criticality task may fire.

**(a) No-fault silence.** If every job complies with $C^{\mathrm{lo}}$, no rung
fires and JNE $= 0$ — identical to AMC-RH.

*Proof*: property **(C)** extended to LO-criticality tasks. Every threshold
$R_i(\chi) \geq R_i(\mathrm{LO})$, and $R_i(\mathrm{LO})$ is by definition
not reached when every task complies. Checked over all tasks, HI and LO, with
zero monotonicity violations. $\square$

**(b) Ordered sacrifice.** At every instant the abandoned set is a down-closed
prefix of the criticality-level order: if $\tau_j$ is abandoned at $t$ and
$\ell_i > \ell_j$, then $\tau_i$ is not abandoned at $t$. Abandoning a
tier-$\ell$ task requires evidence of severity-$\chi_{\ell+1}$ behaviour from
a task of tier $> \ell$.

*Proof*: $S_x = \{i \mid \ell_i < x\}$ is down-closed by construction, and
the trigger rule admits only tasks with $\ell_i \geq x$ to fire rung $x$.
$\square$

**(c) Containment above the knob.** For every tier $m \geq x_{\mathrm{LO}}$,
$$\mathrm{JNE}(\{\tau_j : \ell_j \geq m\}) \leq \mathrm{JNE}_{\text{AMC-RH}}.$$

*Proof*: a task of tier $\geq m$ is abandoned only at a level $x > m \geq
x_{\mathrm{LO}}$. Rungs deeper than $x_{\mathrm{LO}}$ admit only
HI-criticality tasks as triggers, so such a rung fired because some
$\tau_i \in \tau^{\mathrm{HI}}$ reached $R_i(\chi_x) \geq
R_i(\mathrm{LO})$ — meaning it had already passed $R_i(\mathrm{LO})$ earlier
in the same busy period, at which instant AMC-RH degraded. Both schemes exit
only on an idle instant and the busy period has had none, so AMC-RH is still
degraded now. The abandoned set is therefore pointwise dominated. $\square$

**(d) Recovery.** At $x_{\mathrm{LO}} = 0$, (c) covers every tier and the
original Corollary 1 is recovered verbatim.

> **What this costs and buys.** $x_{\mathrm{LO}}$ is a dial, not a defect.
> Raising it surrenders containment in exchange for detecting overruns that
> two-level AMC structurally cannot see — those where a HI-criticality task
> exceeds $C^{\mathrm{lo}}$ yet completes before $R_i(\mathrm{LO})$, as in the
> counterexample. Total JNE may rise; it rises only in tiers the allocation has
> deliberately ranked lowest, and it buys a deadline guarantee for the tiers
> above. Whether that trade pays is measured (sweep $x_{\mathrm{LO}}$ and
> report JNE per tier against AMC-RH), not argued.

> **Raising the knob also tightens the analysis.** The carry-in bound of
> `task_model.tex` §"Shed-Aware Response Time" is available to a task only for
> a rung it may itself fire. A retained LO-criticality task therefore gets a
> finite shed instant exactly for rungs at or below $x_{\mathrm{LO}}$; above
> it, shed tasks are charged in full. So $x_{\mathrm{LO}}$ appears twice — once
> as the mechanism's reach, once as the analysis's precision.

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
at severity $\chi_x$ if, charging every task $C_i(\chi_x)$, charging each shed
task its **carry-in** (below) rather than treating it as absent, and retained
LO-criticality tasks at $C^{\mathrm{lo}}$:

1. every HI-criticality task $\tau_i$ has $R_i \leq D_i$, and
2. every retained LO-criticality task $\tau_j \notin S_x$ has $R_j \leq D_j$.

**Statement**: If every level's drop set is admissible, then at every level no
HI-criticality job and no *retained* LO-criticality job misses its deadline.

*Proof*: Both clauses are exactly the response-time conditions of the standard
fixed-priority test applied at level $L_x$'s budgets and active task set. Level
entry only ever removes tasks (Lemma 1), so the interference each retained task
faces at run time is bounded by the interference the analysis charged it —
provided a shed task is charged for the jobs it released *before* it stopped,
which is what the carry-in term supplies and what the earlier form of this
proof omitted.
Clause 1 gives the HI-criticality obligation; clause 2 gives the retained
LO-criticality obligation. A task in $S_x$ has no deadline obligation: its jobs
are abandoned on release and count toward JNE, not LDM. $\square$

**Corollary 3.1 (no LDM among retained tasks).** LDM $= 0$ for every *retained*
LO-criticality task, by clause 2.

> **The previous statement, "LDM $= 0$ by construction", was too strong.** It
> reasoned that a task in $S_x$ "has no deadline obligation: its jobs are
> abandoned on release and count toward JNE, not LDM". That holds for jobs
> released *after* the rung fires. It fails for the job already in flight when
> it fires: abandonment is on release, so that job runs to completion, and it
> can miss. This is the same instantaneous-shedding assumption that carry-in
> corrects on the interference side, surfacing on the deadline side.
>
> Measured over 60 task sets $\times$ 3 seeds at $f_p = 0.2$ (300k ticks): 23
> LO-criticality deadline misses, **all 23 on tasks in the drop set**, and
> **zero on retained tasks**. Clause 2 is therefore exactly sound as stated;
> only the extrapolation to "no LDM at all" was not. The residual is bounded by
> the same quantity as the carry-in term — at most one in-flight job per shed
> task per degradation episode.

So the scheme trades JNE against *severity coverage* plus a bounded LDM
residual on tasks it has already elected to abandon, not against missed
deadlines for anything it promised to keep.

**Existence.** An admissible set always exists whenever the task set passes
AMC-rtb, since $S_x = \mathrm{LO}$ reduces clause 1 to the AMC-rtb HI-mode test
and vacates clause 2. So the question is never *whether* a level is feasible,
only how little must be shed — which is what makes the drop set an optimisation
rather than a guess.

### Carry-in: what a shed task still costs

Abandonment is on release, so a job of a shed task already in flight when its
level is entered runs to completion and still interferes. Charging it nothing —
which every earlier version of this proof did — is sound only in the limit
where shedding is instantaneous. Charging it honestly gives, for a shed task
$\tau_k$ that stopped releasing at instant $s_k$ within $\tau_i$'s busy period,

$$\min\!\left( \left\lceil \frac{w}{T_k} \right\rceil, \left\lceil \frac{s_k}{T_k} \right\rceil \right) C_k^{\mathrm{lo}}$$

— AMC-rtb's own term, generalised to a per-task shed instant. See
`task_model.tex` §"Shed-Aware Response Time".

$s_k$ is bounded by $R_i(\chi_{\ell_k + 1})$, because if $\tau_i$ is still
incomplete at that instant it fires the rung itself. The argument needs $\tau_i$
to be *permitted* to fire that rung — always so for HI-criticality tasks, and
for a retained LO-criticality task exactly when $\ell_k + 1 \leq
x_{\mathrm{LO}}$. Where it is unavailable, $s_k = \infty$ and $\tau_k$ is
charged in full.

### Measured cost

This changes the design conclusion, not just the numbers. $R_i(\chi) = \infty$
for **33% of HI-criticality tasks at $\chi = 1$**, so a task first shed at a
*deep* rung frequently has no finite $s_k$ at all — and no further shedding can
repair that. Over 199 non-trivial AMC-rtb-schedulable task sets ($n=12$,
$U=0.7$, $CF=2$), trigger severities $(0, 0.5, 1)$:

| Policy | Task sets uncertifiable | LO tasks shed |
|---|---|---|
| Progressive (each task shed at its own rung) | **93 (47%)** | 71.5% |
| Shed-early (one set, all shed at rung 1) | **0 (0%)** | **68.3%** |
| Classic two-level (single trigger) | 0 (0%) | 100% |

On the 106 sets where both are certifiable, shedding early sheds strictly less
on 12, the same on 94, and more on **none**. And shed-early is certifiable
exactly when the task set passes AMC-rtb — shedding everything at
$R_i(\mathrm{LO})$ *is* AMC-rtb's equation (2) — so it never demands more than
the classic test and never rescues a set the classic test rejects.

**Conclusion: grading belongs in the budgets, not in the drop sets.**
Progressive shedding buys nothing the analysis can certify and costs
certifiability on nearly half the population. The scheme still retains 31.7% of
the LO-criticality tasks AMC-RH abandons, with a proof they meet their
deadlines — the quantitative case for grading survives, at a smaller margin
than the optimistic analysis suggested.

For comparison, the figures under the instantaneous-shedding assumption (0.0%
shed at $\chi \leq 0.25$, 10.0% at 0.50, 33.7% at 0.75, 56.3% at 1.00) are
optimistic by exactly the carry-in term.

**Remaining pessimism.** The bound charges every task at the rung's operating
severity for the whole busy period, including the interval when the system was
at a shallower rung. A windowed two-phase form would tighten this. It is
pessimism, not unsoundness: the table above is an upper bound on what must be
shed.

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

2. **Drop Strategy**: shed once, at the shallowest rung. Progressive shedding is not certifiable on nearly half the population once carry-in is charged, and never sheds less. Nesting still holds trivially (every level's set is the same set).

2a. **Budget semantics**: $C_i(\chi_x)$ is an *analysis* quantity, never a run-time throttle. Enforcing it would cap a HI-criticality job at $C_i^{\mathrm{lo}}$ in $L_1$, making overruns undetectable — measured: 1,469 truncated jobs and **zero** level transitions over 2M ticks. The enforced budget is $C_i^{\mathrm{hi}}$.

3. **Exit Policy**: To maintain the "no worse than AMC-RH" guarantee, exit from fully-degraded mode should use the same idle-instant criterion as original AMC.

## References

- AMC-RH Section IV: Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling
- Theorem 1 corresponds to the "by construction" comment in Appendix A of AMC-RH
- Corollary 1 follows from the monotonic drop set property established above
