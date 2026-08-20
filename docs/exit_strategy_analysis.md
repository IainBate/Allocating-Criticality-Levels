# Exit Strategy: Evidence-Cleared and Tempered Exit (Phase 4, Task 4.2)

## Status

**Measured, not just piloted.** This document originally reported a diagnostic upper bound. The
mechanism it recommended has since been built (`simulation.multilevel`, `exit_policy="amc_rh"`),
proven safe for the case it covers (`safety_proof.md`, "Corollary 2: Evidence-Cleared Exit Is
Safe"), and measured as a real, paired comparison (`amc_tasksim.experiments.exit_opportunity.early_exit_trial`).
A tempered variant (`exit_policy="hysteresis"`) has since been built too, proven safe for any
`hold_off` ("Corollary 3: Tempered Exit Is Safe"), and swept as a real paired trial across
`hold_off` and utilisation (Result 4, below) — the oscillation cost from evidence-cleared exit has
a real, quantified remedy, and its sweet spot is now known to shift with utilisation rather than
being one fixed value. The paper section is `research/mode_optimization.tex` §"Stage 5:
Evidence-Cleared Exit". This document is the fuller supporting account, including the confound
this line of work had to work through before the headline numbers were trustworthy.

## Why this exists

The rest of this project's optimisation work fixed **entry**: `research/mode_optimization.tex`'s
four-stage revised protocol establishes the scheme's positive result (+2.4% to +27.3% service
ratio against two-level AMC-RA) and closes every entry-side knob it tested — trigger spacing,
level count `k`, and shed ordering all moved the objective by under 1.6%, an order of magnitude
under this study's 5% practical-significance floor (`contract.EFFECT_FLOOR`).

**Exit** had never been tested. `multilevel.py`'s exit rule was hard-coded direct-to-`L0` on an
idle instant (matching AMC-RA, generalised across levels); cascade exit was deliberately deferred
(`multilevel.py` module docstring), and `task_model.tex` §"Mode Transition Protocol" states the
exit rule as "direct or cascade, per protocol design" — the formal model does not pin it down.
Building or proving a general cascade-exit mechanism is real, nontrivial work. This work measured
whether there was anything to win **before** paying for either, the same way Stage 3 bounded the
drop-strategy question with an exhaustive search before deciding a metaheuristic comparison wasn't
worth running — and, having found a real effect for one specific case (full exit under
`shed_early`), built and proved that specific case rather than the general one.

## Method, and a confound found before trusting the headline number

`amc_tasksim.simulation.multilevel` gained an opt-in diagnostic
(`simulate_multilevel(measure_cascade_opportunity=True)`): at every event the engine already
visits, it computes `_natural_level` — the deepest level currently justified by *live* evidence
(an active, eligible job whose own busy-period-relative threshold has been reached), by the exact
same rule `escalate_if_triggered` already uses for entry, just applied downward instead of upward.
`state.level` is a ratchet (it only rises within a degraded excursion, resetting only on a full
idle instant), so it can lag behind what current evidence alone would justify once the job that
caused an escalation completes while the system stays at that level.

The first cut of this diagnostic (`overdegraded_pct`: % of degraded time spent deeper than live
evidence justifies) came back at **62–91%** across the grid — far larger than any other knob this
project has measured. Before trusting it, a concrete trace was inspected:

```
(542, 'enter_degraded', -1)
(542, 'level_1', -1)
  ...
(658, 'complete', 3)              # the HI job whose overrun triggered entry completes
(658, 'overdegraded_start', 1)    # natural_level drops to 0 right here -- 1 LO job still active
(670, 'drop', 5)                  # new LO releases keep getting dropped ...
(750, 'drop', 9)                  # ... for another ~1000 ticks, while the backlog drains
(875, 'drop', 9)
(916, 'drop', 6)
```

The pattern is structural, not a bug: the job that triggered entry typically completes quickly,
but the run-queue backlog it left behind (jobs already queued before or during the overrun) takes
much longer to drain, and the system correctly stays degraded until the queue empties. A busy
queue draining after its triggering job left is not itself a cost — `overdegraded_pct` conflates
"legitimately still busy" with "unjustified," and reporting it alone would have overstated the
opportunity by roughly an order of magnitude. The metric used from here on instead is
`overdegraded_jne`: of `jne` (LO jobs abandoned on release), the count abandoned specifically
**while** `state.level > natural_level`. A job dropped during the unjustified tail is a real,
countable instance of service loss that a smarter exit rule could plausibly have avoided by
admitting it instead.

## The reframe that made this tractable

`shed_early` constructs a single drop set, used identically at every level ≥ 1 — there is no
intermediate level to cascade *through* for this policy, so the only lever an exit rule has is
**when** the system returns to `L0`, not **which** drop set applies along the way. That is exactly
the question two-level AMC-RH already answers, with a rule that already exists, is already tested,
and already has a published safety argument: `engine.py`'s `AMC_RH.should_exit` — "no active
HI-criticality job has reached `R_k(LO)` since its busy period started" — is the same rule
`_natural_level` computes with `x_lo=0`. `simulation.multilevel` now implements it as
`exit_policy="amc_rh"`, verified bit-for-bit against `engine.py`'s `AMC_RH` at k=2
(`test_k2_full_drop_amc_rh_exit_reproduces_amc_rh_exactly` — not merely similar, the identical
predicate on the identical population and seeds).

## Result 1: service ratio — real, resolved, the largest exit-related effect measured

Paired comparison, `exit_policy="amc_rh"` (candidate) against `"idle"` (baseline, today's shipped
behaviour), `shed_early` only, both operating points, both deadline regimes. `n=24` task sets
(`n=8` at U=0.90/tight — population scarcity, already documented in Stage 1), 5 seeds, 200,000
ticks, `fp=0.2` — the same scale as Stage 2's regime map, `required_pairs()` confirms 17 task sets
would suffice at the weakest resolved cell, so this is comfortably powered throughout.

| U | regime | op | n | idle | amc_rh | diff | rel% | verdict |
|---|--------|----|---|-----:|-------:|-----:|-----:|:---|
| 0.6 | implicit | A | 24 | 0.9747 | 0.9942 | +0.0195 | +2.0% | sig. |
| 0.6 | implicit | B | 24 | 0.9837 | 0.9957 | +0.0120 | +1.2% | sig. |
| **0.6** | **tight** | **A** | 24 | 0.9259 | 0.9802 | +0.0543 | **+5.9%** | **practsig** |
| 0.6 | tight | B | 24 | 0.9373 | 0.9825 | +0.0451 | +4.8% | sig. |
| 0.7 | implicit | A | 24 | 0.9420 | 0.9832 | +0.0411 | +4.4% | sig. |
| 0.7 | implicit | B | 24 | 0.9601 | 0.9868 | +0.0267 | +2.8% | sig. |
| **0.7** | **tight** | **A** | 24 | 0.8566 | 0.9654 | +0.1088 | **+12.7%** | **practsig** |
| **0.7** | **tight** | **B** | 24 | 0.8711 | 0.9688 | +0.0976 | **+11.2%** | **practsig** |
| **0.8** | **implicit** | **A** | 24 | 0.9057 | 0.9779 | +0.0723 | **+8.0%** | **practsig** |
| **0.8** | **implicit** | **B** | 24 | 0.9289 | 0.9813 | +0.0524 | **+5.6%** | **practsig** |
| **0.8** | **tight** | **A** | 24 | 0.7772 | 0.9502 | +0.1731 | **+22.3%** | **practsig** |
| **0.8** | **tight** | **B** | 24 | 0.8070 | 0.9543 | +0.1473 | **+18.2%** | **practsig** |
| **0.9** | **implicit** | **A** | 24 | 0.8118 | 0.9610 | +0.1492 | **+18.4%** | **practsig** |
| **0.9** | **implicit** | **B** | 24 | 0.8471 | 0.9648 | +0.1177 | **+13.9%** | **practsig** |
| **0.9** | **tight** | **A** | 8 | 0.7355 | 0.9332 | +0.1977 | **+26.9%** | **practsig** *(n small)* |
| **0.9** | **tight** | **B** | 8 | 0.7948 | 0.9418 | +0.1469 | **+18.5%** | **practsig** *(n small)* |

All 16 cells positive; 11 practically significant, growing with utilisation and deadline
tightness. This is an order of magnitude above every entry-side knob already closed (trigger
spacing, level count, shed ordering: all ≤1.6%) and above Stage 3's exhaustive bound (≤0.79%).

## Result 2: oscillation cost — equally real, tracked deliberately

Same cells, same pairing, `level_trans` (mode changes per run) instead of service ratio.

| U | regime | op | idle | amc_rh | diff | rel% |
|---|--------|----|-----:|-------:|-----:|-----:|
| 0.6 | implicit | A | 129.1 | 161.9 | +32.8 | +25.4% |
| 0.6 | implicit | B | 125.6 | 161.9 | +36.3 | +28.9% |
| 0.6 | tight | A | 323.8 | 390.1 | +66.3 | +20.5% |
| 0.6 | tight | B | 322.6 | 390.0 | +67.5 | +20.9% |
| 0.7 | implicit | A | 166.3 | 222.3 | +55.9 | +33.6% |
| 0.7 | implicit | B | 156.5 | 222.0 | +65.5 | +41.9% |
| 0.7 | tight | A | 402.2 | 522.4 | +120.2 | +29.9% |
| 0.7 | tight | B | 398.4 | 522.6 | +124.2 | +31.2% |
| 0.8 | implicit | A | 147.8 | 209.6 | +61.8 | +41.8% |
| 0.8 | implicit | B | 140.8 | 209.7 | +68.8 | +48.9% |
| 0.8 | tight | A | 396.7 | 568.3 | +171.6 | +43.3% |
| 0.8 | tight | B | 380.8 | 568.1 | +187.3 | +49.2% |
| 0.9 | implicit | A | 185.6 | 300.4 | +114.7 | +61.8% |
| 0.9 | implicit | B | 160.7 | 300.4 | +139.7 | +86.9% |
| 0.9 | tight | A | 411.4 | 631.6 | +220.2 | +53.5% |
| 0.9 | tight | B | 356.4 | 631.6 | +275.2 | +77.2% |

Every cell is a practically significant *increase* — 20–87% relative, and substantial in absolute
count too (+33 to +275 transitions on baselines already in the hundreds, not a large percentage of
a small number). It grows with utilisation in lockstep with the service-ratio benefit: the cells
that gain the most also oscillate the most. **This answers the question directly: does
evidence-cleared exit trade service for churn? Yes, and the trade is not free in either direction.**

**Is the churn a problem?** `metrics_objective.md`'s `δ`, the weight on `LevelTrans` in `Φ`, is
deliberately small relative to `α(U)`/`β(U)` — this project's own prior that churn matters less
than service loss or time degraded. That prior is a reasonable starting point, but a 20–87%
relative increase is large enough that it should be checked against `Φ` explicitly, not waved
through on the weight alone; no implementation of `Φ` currently exists in the codebase to check it
against (Stages 1–4 reported service ratio directly rather than a combined score), so this is
reported as two separate, both-resolved numbers rather than a synthetic net figure that would
embed an unvalidated weighting.

## Result 3: progressive / cascade — closed, with evidence, at almost no headroom

The question raised when this diagnostic was first run: might a genuine intermediate-level cascade
recover meaningfully more than simple full exit? Decomposing the same JNE-during-tail measure by
what kind of exit each drop would have needed (`overdegraded_jne_full_exit` vs. the remainder)
answers it directly, same grid, `progressive` policy (severities `(0.0, 0.25)`):

| U | regime | op | n | JNE-tail% | full-exit% | cascade-headroom% |
|---|--------|----|---|----------:|-----------:|-------------------:|
| 0.6 | implicit | A | 22 | 1.614 | 1.560 | 0.054 |
| 0.6 | implicit | B | 22 | 0.993 | 0.949 | 0.043 |
| 0.6 | tight | A | 19 | 5.106 | 4.847 | 0.258 |
| 0.6 | tight | B | 17 | 3.954 | 3.773 | 0.181 |
| 0.7 | implicit | A | 16 | 4.306 | 4.123 | 0.183 |
| 0.7 | implicit | B | 14 | 1.692 | 1.609 | 0.083 |
| 0.7 | tight | A | 15 | 9.265 | 8.890 | 0.376 |
| 0.7 | tight | B | 13 | 8.057 | 7.719 | 0.339 |
| 0.8 | implicit | A | 16 | 6.017 | 5.833 | 0.184 |
| 0.8 | implicit | B | 15 | 3.937 | 3.811 | 0.126 |
| 0.8 | tight | A | 14 | 18.389 | 17.531 | 0.858 |
| 0.8 | tight | B | 12 | 11.977 | 11.359 | 0.618 |
| 0.9 | implicit | A | 17 | 12.433 | 11.964 | 0.469 |
| 0.9 | implicit | B | 16 | 8.488 | 8.134 | 0.354 |
| 0.9 | tight | A | 6 | 17.607 | 16.850 | 0.757 |
| 0.9 | tight | B | 5 | 8.166 | 7.823 | 0.342 |

92–97% of progressive's apparent opportunity is recoverable by full exit alone in every cell; the
cascade-specific remainder is 0.04–0.86 percentage points — below this study's 5% floor everywhere,
an order of magnitude below it almost everywhere. Intuitively: a shallow rung's evidence typically
has a head start on clearing, so by the time the deepest rung's evidence has gone stale, the
shallow rung's usually has too — the system needs a *partial* demotion, rather than a full exit,
only rarely.

**This closes the cascade question with evidence, the same way Stage 3 closed drop-strategy
search.** A general cascade mechanism would need a new abstraction in `multilevel.py` (direct exit
is currently the only exit rule it supports at all) and a new safety argument (an intermediate
level's admissibility was certified for a system that *escalated* to it, not one that *demoted* to
it — see Corollary 2's scope note below). The table shows that investment has almost nothing left
to buy. **Not recommended.**

## Safety

`safety_proof.md`'s "Corollary 2: Evidence-Cleared Exit Is Safe" establishes that swapping idle-only
exit for evidence-cleared exit changes none of Theorem 1 (HI meets deadlines), the LO-deadline
corollary, Lemma 1, or Corollary 1' — every safety property already proven for the shipped scheme
holds unconditionally, because the argument only needs that `L0` is unrestricted normal mode under
both schemes, identical to AMC-RH's own. It is **explicitly scoped to full exit only** — the same
proof does not cover demoting to an intermediate level, which is exactly why the general cascade
mechanism above is not recommended without its own, separate safety argument.

One overclaim was caught and corrected while writing this proof, worth recording rather than
silently fixing: an early draft claimed total time-degraded (TiD) can only *decrease* under
evidence-cleared exit, over a whole run. That is not proven and is not true in general — admitting
a LO job idle-exit would have dropped can, via priority interference, pull a later HI-criticality
task's inherited busy-period start earlier and trigger an escalation the idle-exit run would not
have had, so the two runs' trajectories can diverge after the first differential admission
decision. Safety does not depend on this either way (Theorem 1 protects HI deadlines at whatever
levels either run visits); net TiD, service ratio, and oscillation are the *measured* quantities in
Results 1–2 above, not derived from the proof.

## Result 4: tempered exit (hysteresis) — a real sweet spot, and it moves with U

`exit_policy="hysteresis"` implements the tempered variant: exit to `L0` once evidence has been
continuously clear for `hold_off` ticks, rather than immediately. It needed genuinely new
machinery — a hold-off deadline is a *timed* condition, unlike evidence-cleared exit's
*event*-triggered one, so `multilevel.py` gained `_next_evidence_reappearance` (an extension of
the existing next-event lookahead) so the loop never skips past a reappearance between otherwise-
scheduled events. Two limits make it exactly, not approximately, an interpolation between the two
policies already measured: `hold_off=0` reproduces `exit_policy="amc_rh"` bit-for-bit, and a
`hold_off` longer than any realistic gap reproduces `exit_policy="idle"` bit-for-bit — both
verified in `tests/simulation/test_multilevel.py`, not just argued.

**Safety** (`safety_proof.md`, "Corollary 3: Tempered Exit Is Safe") does not depend on `hold_off`
or on the lookahead's precision. The reason is a boundary fact worth stating plainly:
$R_i(\chi)$ is a worst-case bound computed under *full* interference, so a task that has not yet
crossed it is guaranteed safe under full interference from here on, regardless of what happens
next — $\mathrm{natural}(t) = 0$ isn't a reasonable proxy for the safe-to-exit boundary, it *is*
that boundary, exactly. Every actual exit under `"hysteresis"` is gated on a **fresh** check of
this condition at the instant it fires, not on the hold-off bookkeeping — so a missed or imprecise
wake-up can only delay an exit (still individually safe), never bring one forward into unsafe
territory.

**A real (not retrospective) paired sweep** — `exit_opportunity.hysteresis_sweep` — measured
`hold_off` ∈ {0, 25, 50, 100, 200, 400, 800, 1600, 3200} against `"idle"`, shed_early, operating
point B, across the full U × regime grid (n=16 task sets, n=7 at U=0.9/tight, 5 seeds, 200,000
ticks). **[Interactive chart](https://claude.ai/code/artifact/9b50c611-c9e4-4419-8d0f-ea464180b662)**
— both curves (service-ratio gain, level-transition increase) by `hold_off`, faceted by regime,
one line per U.

The shape is consistent everywhere: level-transition increase falls faster than service-ratio gain
does as `hold_off` grows, which is what makes a middle value worth choosing rather than either
extreme. Anchoring on "smallest `hold_off` that caps the oscillation increase at ≤10%":

| U | regime | `hold_off` for LT≤10% | SR gain retained |
|---|--------|----------------------:|------------------:|
| 0.6 | implicit | 400 | 28.1% |
| 0.6 | tight | 200 | 39.2% |
| 0.7 | implicit | 400 | 31.5% |
| 0.7 | tight | 200 | 43.4% |
| 0.8 | implicit | 400 | 35.8% |
| 0.8 | tight | 400 | 19.6% |
| **0.9** | **implicit** | **1600** | **10.5%** |
| **0.9** | **tight** | **800** | **10.5%** |

**The sweet spot moves with utilisation, and not in the forgiving direction.** At U=0.6–0.7, a
modest `hold_off` (200–400) caps oscillation cheaply, keeping 28–43% of the gain. At U=0.9, the
same 10% cap needs a `hold_off` 2–8× larger and keeps only ~10.5% — churn is denser at high
utilisation (baseline `level_trans` roughly doubles from U=0.6 to U=0.9 in the same regime), so a
fixed `hold_off` suppresses proportionally less of it. **There is no single `hold_off` that is
simultaneously cheap and effective across the whole grid** — any deployed value is a choice about
which utilisation regime to favour, not a free parameter to set once.

## What remains open

1. **Net verdict on the service/oscillation trade** — reported as resolved numbers, not combined
   into one score, since no validated `Φ` implementation exists to combine them against (Result 2).
   The hysteresis sweep sharpens this rather than resolving it: it shows *how much* of each you can
   have for a given `hold_off`, not which point on that curve is "worth it" — that judgement still
   needs a weighting this study hasn't validated.
2. **`hold_off` is not adaptive.** The sweep uses one fixed value per run; whether `hold_off` should
   itself scale with a runtime-observable proxy for utilisation (rather than being fixed at deploy
   time) is unexplored.
3. **U=0.9/tight cells run on n=7, not n=16** (population scarcity, already documented in Stage 1) —
   directionally consistent with the rest of the grid, not at the same power.
4. **Progressive's own regime-map scope limit still applies**: its `n` shrinks at exactly the
   hardest cells (as in Stage 2's regime map), so the cascade-headroom table above, while
   consistent throughout, is thinnest exactly where the numbers are largest.
