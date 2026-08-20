"""Cascade-exit opportunity: is direct-exit's ratchet actually costing anything?

Phase 4's Task 4.2 (exit strategy) has been blocked on two separate things:
building a cascade-exit *mechanism* for the k-level engine (`multilevel.py`
currently only supports direct-to-L0-on-idle, deliberately deferred -- see its
module docstring), and proving that mechanism safe (a new correctness argument,
since there is no existing one for cascade to inherit, unlike hysteresis on top
of the existing direct-exit rule). Both are real costs. Before paying either,
this module measures whether there is anything to win, using the diagnostic in
`amc_tasksim.simulation.multilevel` (`measure_cascade_opportunity=True`,
`_natural_level`): how much of the scheme's *actual* degraded time (under
today's direct-exit rule) is spent deeper than the currently active evidence
alone would justify.

This is a single-arm measurement, not a paired comparison -- there is no
baseline configuration to pair against, only the scheme measured against
itself. It reuses `multilevel_protocol`'s population-building and grid so the
result sits on the same task sets already characterised by the regime map
(`research/mode_optimization.tex`, Stage 2), rather than restating it.

UPPER BOUND, not an achievable or safety-checked number. It assumes demoting
the instant justifying evidence disappears is free and safe -- exactly the
question a real cascade-exit design would still have to settle. If this bound
is small (in the same "under the 5% floor" territory as trigger spacing, level
count, and shed ordering already turned out to be -- research/mode_optimization.tex,
"What This Protocol Does and Does Not Yet Establish"), that closes Task 4.2's
cascade question with evidence, the same way Stage 3 closed the drop-strategy
question, without needing to build or prove the mechanism. If it is not small,
that is the trigger to scope the cascade safety proof properly.

Update: for the adopted `shed_early` policy specifically, the diagnostic above
turned out to measure something with an existing, already-proven mechanism --
not a cascade question at all, since shed_early has only one drop set (see
`docs/exit_strategy_analysis.md`, "The reframe"). `simulation.multilevel` now
implements it (`exit_policy="amc_rh"`, verified bit-identical to `engine.py`'s
`AMC_RH` at k=2), and `safety_proof.md`'s "Corollary 2: Evidence-Cleared Exit
Is Safe" proves it safe, scoped explicitly to full exit only. `early_exit_trial`
below is the real, paired measurement of that mechanism's gain -- and its
oscillation cost (`level_trans`) -- replacing the diagnostic upper bound for
`shed_early`. The diagnostic upper bound above remains the only measurement
available for `progressive`'s harder, still-unimplemented cascade case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from amc_tasksim.experiments.contract import CANONICAL, PairedResult, paired_compare
from amc_tasksim.experiments.multilevel_protocol import (
    TIGHT_ALPHA,
    build_tight_population,
    progressive_ladder,
    shed_early_ladder,
)
from amc_tasksim.experiments.sweep import build_population
from amc_tasksim.simulation.multilevel import simulate_multilevel


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and standard error -- same formula as contract.paired_compare
    (sample variance, ddof=1), so results are reported on a consistent basis
    even though this is a single-arm measurement, not a paired comparison.
    """
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


@dataclass(frozen=True)
class OverdegradedCell:
    """One (U, deadline regime, operating point, policy) cell's measurement.

    Attributes:
        n_tasksets: Task sets contributing (ladder-feasible ones only).
        overdegraded_pct_mean/se: Mean and standard error, across task sets, of
            the % of degraded time spent deeper than live evidence justifies
            (`MultiLevelResult.overdegraded_pct`, averaged over seeds first
            within each task set -- the pairing unit this project's contract
            uses throughout, even though there is no second arm to pair here).
        overdegraded_level_pct_mean/se: Depth-weighted version
            (`overdegraded_level_pct`) -- distinguishes "a little, often" from
            "a lot, rarely".
        overdegraded_jne_pct_mean/se: Mean and standard error of
            `overdegraded_jne_pct` -- LO jobs actually abandoned on release
            while over-degraded, as a % of `lo_expected`. Unlike the two
            *_pct fields above, this is not confounded by legitimate
            backlog-drain time (a busy queue draining after its triggering
            job left is not itself a cost); it is the direct,
            paper-vocabulary (JNE/service-ratio) quantity a smarter exit rule
            could plausibly have saved.
        full_exit_pct_mean/se: The part of overdegraded_jne_pct that a *full*
            exit to L0 recovers (`overdegraded_jne_full_exit_pct`) -- for
            shed_early this is implemented and measured for real by
            `early_exit_trial`; here it is reported for comparability with
            `progressive`, where the remainder (`overdegraded_jne_pct -
            full_exit_pct`) is what a real *cascade* mechanism could
            additionally recover, at most, on top of full exit alone -- the
            drill-down this module exists to support.
        mean_events_per_run: Mean number of distinct over-degraded episodes.
    """

    U: float
    regime: str  # "implicit" or "tight"
    operating_point: str  # "A" (conservative) or "B" (termination)
    policy: str  # "shed_early" or "progressive"
    n_tasksets: int
    overdegraded_pct_mean: float
    overdegraded_pct_se: float
    overdegraded_level_pct_mean: float
    overdegraded_level_pct_se: float
    overdegraded_jne_pct_mean: float
    overdegraded_jne_pct_se: float
    full_exit_pct_mean: float
    full_exit_pct_se: float
    mean_events_per_run: float

    @property
    def cascade_headroom_pct_mean(self) -> float:
        """What a real cascade mechanism could add beyond full exit alone."""
        return self.overdegraded_jne_pct_mean - self.full_exit_pct_mean

    def summary(self) -> str:
        return (
            f"U={self.U} {self.regime}/{self.operating_point}/{self.policy}  "
            f"n={self.n_tasksets}  "
            f"overdegraded={self.overdegraded_pct_mean:.2f}%+/-{self.overdegraded_pct_se:.2f}  "
            f"depth-weighted={self.overdegraded_level_pct_mean:.2f}%+/-{self.overdegraded_level_pct_se:.2f}  "
            f"JNE-during-tail={self.overdegraded_jne_pct_mean:.3f}%+/-{self.overdegraded_jne_pct_se:.3f}  "
            f"(full-exit={self.full_exit_pct_mean:.3f}%, cascade-headroom={self.cascade_headroom_pct_mean:.3f}%)  "
            f"events/run={self.mean_events_per_run:.2f}"
        )


def overdegraded_opportunity(
    U_values: Sequence[float] = CANONICAL.U_levels,
    n_tasksets: int = 24,
    seeds: Sequence[int] = tuple(range(5)),
    duration: int = 200_000,
    fp: float = 0.2,
    n_tasks: int = CANONICAL.n_tasks,
    CP: float = CANONICAL.CP,
    CF: float = CANONICAL.CF,
    progressive_severities: Sequence[float] = (0.0, 0.25),
    seed: int = 0,
) -> list[OverdegradedCell]:
    """How much of the scheme's degraded time is unjustified by live evidence.

    Grid and population-building mirror `multilevel_protocol.regime_map`
    exactly, so a run at the same arguments lands on the same task sets that
    stage already measured -- but this stage runs each configuration once
    (against itself), not paired against AMC-RA, since there is nothing here
    to pair: the question is internal to the scheme's own exit timing.
    """
    cells: list[OverdegradedCell] = []
    for U in U_values:
        for regime, alpha in (("implicit", 1.0), ("tight", TIGHT_ALPHA)):
            if alpha == 1.0:
                tasksets, _ = build_population(n_tasksets, U, seed, n=n_tasks, CP=CP, CF=CF)
            else:
                tasksets = build_tight_population(
                    n_tasksets, U, seed, alpha=alpha, n=n_tasks, CP=CP, CF=CF
                )
            if not tasksets:
                continue

            for op_name, strict in (("A", True), ("B", False)):
                for policy, builder, extra in (
                    ("shed_early", shed_early_ladder, {}),
                    ("progressive", progressive_ladder, {"severities": progressive_severities}),
                ):
                    pct_vals: list[float] = []
                    level_pct_vals: list[float] = []
                    jne_pct_vals: list[float] = []
                    full_exit_pct_vals: list[float] = []
                    events_vals: list[float] = []
                    for ts in tasksets:
                        kwargs = dict(extra)
                        if policy == "progressive":
                            severities = kwargs.pop("severities")
                            lad = builder(ts, severities, require_lo_deadlines=strict)
                        else:
                            lad = builder(ts, require_lo_deadlines=strict)
                        if lad is None:
                            continue
                        run_pcts, run_level_pcts, run_jne_pcts, run_events = [], [], [], []
                        run_full_exit_pcts = []
                        for sd in seeds:
                            r = simulate_multilevel(
                                ts, lad, duration=duration, seed=sd, fp=fp,
                                measure_cascade_opportunity=True,
                            )
                            run_pcts.append(r.overdegraded_pct)
                            run_level_pcts.append(r.overdegraded_level_pct)
                            run_jne_pcts.append(r.overdegraded_jne_pct)
                            run_full_exit_pcts.append(r.overdegraded_jne_full_exit_pct)
                            run_events.append(r.overdegraded_events)
                        pct_vals.append(sum(run_pcts) / len(run_pcts))
                        level_pct_vals.append(sum(run_level_pcts) / len(run_level_pcts))
                        jne_pct_vals.append(sum(run_jne_pcts) / len(run_jne_pcts))
                        full_exit_pct_vals.append(sum(run_full_exit_pcts) / len(run_full_exit_pcts))
                        events_vals.append(sum(run_events) / len(run_events))
                    if len(pct_vals) < 2:
                        continue
                    pct_mean, pct_se = _mean_se(pct_vals)
                    level_mean, level_se = _mean_se(level_pct_vals)
                    jne_mean, jne_se = _mean_se(jne_pct_vals)
                    full_exit_mean, full_exit_se = _mean_se(full_exit_pct_vals)
                    cells.append(OverdegradedCell(
                        U=U, regime=regime, operating_point=op_name, policy=policy,
                        n_tasksets=len(pct_vals),
                        overdegraded_pct_mean=pct_mean, overdegraded_pct_se=pct_se,
                        overdegraded_level_pct_mean=level_mean, overdegraded_level_pct_se=level_se,
                        full_exit_pct_mean=full_exit_mean, full_exit_pct_se=full_exit_se,
                        overdegraded_jne_pct_mean=jne_mean, overdegraded_jne_pct_se=jne_se,
                        mean_events_per_run=sum(events_vals) / len(events_vals),
                    ))
    return cells


# ---------------------------------------------------------------------------
# The real, paired measurement for shed_early: exit_policy="amc_rh" vs "idle"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyExitCell:
    """One (U, deadline regime, operating point) cell's paired comparison of
    exit_policy="amc_rh" (candidate) against "idle" (baseline), shed_early only.

    Attributes:
        service_ratio: Paired comparison on service ratio -- the headline benefit.
            `mean_diff > 0` means "amc_rh" completed more LO work.
        tid: Paired comparison on fraction of duration spent degraded -- a
            second, independent benefit metric: `mean_diff < 0` means "amc_rh"
            spent less time degraded, consistent with exiting sooner.
        wasted_cpu_pct: Paired comparison on `wasted_cpu_pct` (CPU spent on
            work later thrown away, e.g. terminated LO jobs). Included for
            completeness against metrics_objective.md's Phi terms; expected to
            move little either way, since exit timing does not change which
            already-admitted jobs get terminated.
        level_trans: Paired comparison on level transitions per run -- the
            oscillation cost being tracked deliberately, not assumed away:
            evidence-cleared exit can re-enter degraded mode more readily
            than idle-only exit, so a real gain in service_ratio could still
            be a net loss if it comes with materially more mode changes.
    """

    U: float
    regime: str  # "implicit" or "tight"
    operating_point: str  # "A" (conservative) or "B" (termination)
    n_tasksets: int
    service_ratio: PairedResult
    level_trans: PairedResult
    tid: PairedResult
    wasted_cpu_pct: PairedResult

    def summary(self) -> str:
        return (
            f"U={self.U} {self.regime}/{self.operating_point}  n={self.n_tasksets}  "
            f"service_ratio: {self.service_ratio.summary()}  |  "
            f"tid: {self.tid.summary()}  |  "
            f"wasted_cpu_pct: {self.wasted_cpu_pct.summary()}  |  "
            f"level_trans: {self.level_trans.summary()}"
        )


def early_exit_trial(
    U_values: Sequence[float] = CANONICAL.U_levels,
    n_tasksets: int = 24,
    seeds: Sequence[int] = tuple(range(5)),
    duration: int = 200_000,
    fp: float = 0.2,
    n_tasks: int = CANONICAL.n_tasks,
    CP: float = CANONICAL.CP,
    CF: float = CANONICAL.CF,
    seed: int = 0,
) -> list[EarlyExitCell]:
    """Paired trial: does exit_policy="amc_rh" beat "idle" for shed_early, and
    at what oscillation cost?

    Scoped to shed_early only -- deliberately one factor at a time, per the
    reframe in `docs/exit_strategy_analysis.md`: shed_early's exit rule is
    proven safe (safety_proof.md, Corollary 2); progressive's is not, so it is
    not included here (see `overdegraded_opportunity` for its diagnostic
    estimate instead). Grid and population-building mirror `regime_map`'s
    exactly, so this sits on the same task sets that stage already used.
    """
    cells: list[EarlyExitCell] = []
    for U in U_values:
        for regime, alpha in (("implicit", 1.0), ("tight", TIGHT_ALPHA)):
            if alpha == 1.0:
                tasksets, _ = build_population(n_tasksets, U, seed, n=n_tasks, CP=CP, CF=CF)
            else:
                tasksets = build_tight_population(
                    n_tasksets, U, seed, alpha=alpha, n=n_tasks, CP=CP, CF=CF
                )
            if not tasksets:
                continue

            for op_name, strict in (("A", True), ("B", False)):
                idle_sr, amc_sr = [], []
                idle_lt, amc_lt = [], []
                idle_tid, amc_tid = [], []
                idle_wcpu, amc_wcpu = [], []
                for ts in tasksets:
                    lad = shed_early_ladder(ts, require_lo_deadlines=strict)
                    if lad is None:
                        continue
                    idle_runs = [
                        simulate_multilevel(ts, lad, duration=duration, seed=sd, fp=fp,
                                             exit_policy="idle")
                        for sd in seeds
                    ]
                    amc_runs = [
                        simulate_multilevel(ts, lad, duration=duration, seed=sd, fp=fp,
                                             exit_policy="amc_rh")
                        for sd in seeds
                    ]
                    idle_sr.append(sum(r.service_ratio for r in idle_runs) / len(idle_runs))
                    amc_sr.append(sum(r.service_ratio for r in amc_runs) / len(amc_runs))
                    idle_lt.append(sum(r.level_trans for r in idle_runs) / len(idle_runs))
                    amc_lt.append(sum(r.level_trans for r in amc_runs) / len(amc_runs))
                    idle_tid.append(sum(r.tid for r in idle_runs) / len(idle_runs))
                    amc_tid.append(sum(r.tid for r in amc_runs) / len(amc_runs))
                    idle_wcpu.append(sum(r.wasted_cpu_pct for r in idle_runs) / len(idle_runs))
                    amc_wcpu.append(sum(r.wasted_cpu_pct for r in amc_runs) / len(amc_runs))
                if len(idle_sr) < 2:
                    continue
                cells.append(EarlyExitCell(
                    U=U, regime=regime, operating_point=op_name,
                    n_tasksets=len(idle_sr),
                    service_ratio=paired_compare(idle_sr, amc_sr),
                    level_trans=paired_compare(idle_lt, amc_lt),
                    tid=paired_compare(idle_tid, amc_tid),
                    wasted_cpu_pct=paired_compare(idle_wcpu, amc_wcpu),
                ))
    return cells


# ---------------------------------------------------------------------------
# Retrospective hold-off sweep: is evidence-cleared exit's churn dominated by
# quick flicker a hysteresis hold-off would cheaply suppress, or spread out?
# ---------------------------------------------------------------------------


def _exit_reentry_gaps(
    ts, ladder, seed: int, duration: int, fp: float,
) -> list[tuple[int, int]]:
    """One run's (gap_length, lo_admitted_in_gap) pairs under exit_policy="amc_rh".

    A "gap" is the interval between one exit-to-L0 and the next re-entry into
    a degraded level -- exactly the interval a hold-off of length >= gap would
    merge into one continuous degraded excursion, suppressing both the exit
    and the following re-entry (2 of the level_trans events amc_rh adds over
    idle-exit). `lo_admitted_in_gap` is how many LO-criticality jobs were
    released (and, since the system is at L0 throughout a gap, unconditionally
    admitted) during that specific interval -- exactly what a hold-off long
    enough to suppress the gap would instead have dropped, since the system
    would still be degraded throughout. A run's trailing gap (after its last
    exit, if the run ends before any re-entry) is excluded: it never closes,
    so it is not a transition a hold-off could suppress.
    """
    trace: list[tuple] = []
    r = simulate_multilevel(ts, ladder, duration=duration, seed=seed, fp=fp,
                             exit_policy="amc_rh", trace=trace)
    gaps: list[tuple[int, int]] = []
    exit_time: Optional[int] = None
    lo_in_gap = 0
    for now, event, tid in trace:
        if event == "exit_degraded":
            exit_time = now
            lo_in_gap = 0
        elif event == "release" and exit_time is not None and ts.criticality[tid] == "LO":
            lo_in_gap += 1
        elif event == "enter_degraded" and exit_time is not None:
            gaps.append((now - exit_time, lo_in_gap))
            exit_time = None
    return gaps


@dataclass(frozen=True)
class HoldoffPoint:
    """What a candidate hold-off H would do, retrospectively, to one cell's
    exit-to-reentry gaps under exit_policy="amc_rh".

    Attributes:
        hold_off: Candidate minimum time at L0 before exit is allowed to
            complete (a hysteresis rule enforcing this is not built -- see
            docs/exit_strategy_analysis.md; this is a what-if computed from
            traces of the mechanism that *is* built).
        n_gaps: Total exit-to-reentry gaps observed (the denominator).
        frac_suppressed: Fraction of gaps <= hold_off -- these transitions
            (both the exit and the following re-entry) would not have
            happened, so 2 * frac_suppressed * n_gaps is roughly how many of
            amc_rh's *extra* level_trans (over idle) this hold_off removes.
        lo_given_back: LO jobs admitted during a suppressed gap -- these would
            instead have been dropped, since a long-enough hold-off keeps the
            system degraded (and shed_early's drop set applies) for the whole
            gap. This is the service-ratio cost of choosing this hold_off.
        lo_admitted_in_any_gap: Total LO jobs admitted across ALL gaps
            (suppressed or not) -- the denominator for `lo_given_back`, and a
            lower bound on evidence-cleared exit's total benefit over idle (it
            excludes jobs admitted after a run's last exit, which never closes).
    """

    hold_off: int
    n_gaps: int
    frac_suppressed: float
    lo_given_back: int
    lo_admitted_in_any_gap: int

    @property
    def lo_given_back_frac(self) -> float:
        """`lo_given_back` as a fraction of the benefit this hold_off puts at risk."""
        return self.lo_given_back / self.lo_admitted_in_any_gap if self.lo_admitted_in_any_gap else 0.0


def hold_off_sweep(
    hold_offs: Sequence[int],
    U: float,
    regime: str = "implicit",
    require_lo_deadlines: bool = False,
    n_tasksets: int = 16,
    seeds: Sequence[int] = tuple(range(5)),
    duration: int = 200_000,
    fp: float = 0.2,
    n_tasks: int = CANONICAL.n_tasks,
    CP: float = CANONICAL.CP,
    CF: float = CANONICAL.CF,
    seed: int = 0,
) -> list[HoldoffPoint]:
    """For each candidate hold-off, what fraction of amc_rh's extra churn would
    it suppress, and what fraction of amc_rh's benefit would it give back?

    Answers whether a tempered (hysteresis) variant is worth building, without
    building it: pools exit-to-reentry gaps (`_exit_reentry_gaps`) across a
    population, then for each candidate `H` in `hold_offs`, reports what
    fraction of gaps are short enough to be suppressed and what fraction of
    the LO work admitted in gaps would be given back. A hold_off is only
    promising if `frac_suppressed` is materially larger than
    `lo_given_back_frac` -- if the two move together, there is no hold_off
    that trades much churn for little service, and hysteresis is not worth
    building for this regime.
    """
    if regime == "implicit":
        tasksets, _ = build_population(n_tasksets, U, seed, n=n_tasks, CP=CP, CF=CF)
    elif regime == "tight":
        tasksets = build_tight_population(
            n_tasksets, U, seed, alpha=TIGHT_ALPHA, n=n_tasks, CP=CP, CF=CF
        )
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    all_gaps: list[tuple[int, int]] = []
    for ts in tasksets:
        lad = shed_early_ladder(ts, require_lo_deadlines=require_lo_deadlines)
        if lad is None:
            continue
        for sd in seeds:
            all_gaps.extend(_exit_reentry_gaps(ts, lad, sd, duration, fp))

    n_gaps = len(all_gaps)
    total_lo = sum(lo for _gap, lo in all_gaps)
    points = []
    for H in hold_offs:
        suppressed = [(g, lo) for g, lo in all_gaps if g <= H]
        points.append(HoldoffPoint(
            hold_off=H,
            n_gaps=n_gaps,
            frac_suppressed=len(suppressed) / n_gaps if n_gaps else 0.0,
            lo_given_back=sum(lo for _g, lo in suppressed),
            lo_admitted_in_any_gap=total_lo,
        ))
    return points


# ---------------------------------------------------------------------------
# The real (non-retrospective) hysteresis sweep: exit_policy="hysteresis" at
# each hold_off, paired against "idle", across U -- is there a sweet spot,
# and does it move with utilisation?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HysteresisCell:
    """One (hold_off, U, regime) cell's paired comparison of
    exit_policy="hysteresis" (candidate) against "idle" (baseline), shed_early,
    operating point B (termination).

    `hold_off=0` is exactly `exit_policy="amc_rh"` (safety_proof.md Corollary
    3, verified bit-identical in tests/simulation/test_multilevel.py) -- it is
    included in every sweep as a built-in cross-check against `early_exit_trial`,
    not a special case needing separate code.
    """

    hold_off: int
    U: float
    regime: str  # "implicit" or "tight"
    n_tasksets: int
    service_ratio: PairedResult
    level_trans: PairedResult

    def summary(self) -> str:
        return (
            f"H={self.hold_off} U={self.U} {self.regime}  n={self.n_tasksets}  "
            f"service_ratio: {self.service_ratio.summary()}  |  "
            f"level_trans: {self.level_trans.summary()}"
        )


def hysteresis_sweep(
    hold_offs: Sequence[int],
    U_values: Sequence[float] = CANONICAL.U_levels,
    regimes: Sequence[str] = ("implicit", "tight"),
    require_lo_deadlines: bool = False,
    n_tasksets: int = 16,
    seeds: Sequence[int] = tuple(range(5)),
    duration: int = 200_000,
    fp: float = 0.2,
    n_tasks: int = CANONICAL.n_tasks,
    CP: float = CANONICAL.CP,
    CF: float = CANONICAL.CF,
    seed: int = 0,
) -> list[HysteresisCell]:
    """Real, paired measurement of exit_policy="hysteresis" across a grid of
    (hold_off, U, regime) -- graph-ready data for how the service/oscillation
    trade moves as hold_off increases, and whether that shape depends on U.

    The "idle" baseline is simulated once per task set and reused across every
    hold_off in the sweep (it does not depend on hold_off), so the cost of
    adding more hold_off values to `hold_offs` is one extra simulation per
    task set per seed, not two.
    """
    cells: list[HysteresisCell] = []
    for U in U_values:
        for regime in regimes:
            if regime == "implicit":
                tasksets, _ = build_population(n_tasksets, U, seed, n=n_tasks, CP=CP, CF=CF)
            elif regime == "tight":
                tasksets = build_tight_population(
                    n_tasksets, U, seed, alpha=TIGHT_ALPHA, n=n_tasks, CP=CP, CF=CF
                )
            else:
                raise ValueError(f"Unknown regime: {regime!r}")
            if not tasksets:
                continue

            ladders: list = []
            idle_sr_by_ts: list[list[float]] = []
            idle_lt_by_ts: list[list[float]] = []
            for ts in tasksets:
                lad = shed_early_ladder(ts, require_lo_deadlines=require_lo_deadlines)
                if lad is None:
                    continue
                ladders.append((ts, lad))
                idle_runs = [
                    simulate_multilevel(ts, lad, duration=duration, seed=sd, fp=fp,
                                         exit_policy="idle")
                    for sd in seeds
                ]
                idle_sr_by_ts.append([r.service_ratio for r in idle_runs])
                idle_lt_by_ts.append([r.level_trans for r in idle_runs])
            if len(ladders) < 2:
                continue

            for H in hold_offs:
                hyst_sr, hyst_lt = [], []
                for (ts, lad), idle_sr_seeds, idle_lt_seeds in zip(
                    ladders, idle_sr_by_ts, idle_lt_by_ts
                ):
                    hyst_runs = [
                        simulate_multilevel(ts, lad, duration=duration, seed=sd, fp=fp,
                                             exit_policy="hysteresis", hold_off=H)
                        for sd in seeds
                    ]
                    hyst_sr.append(sum(r.service_ratio for r in hyst_runs) / len(hyst_runs))
                    hyst_lt.append(sum(r.level_trans for r in hyst_runs) / len(hyst_runs))
                idle_sr_means = [sum(v) / len(v) for v in idle_sr_by_ts]
                idle_lt_means = [sum(v) / len(v) for v in idle_lt_by_ts]
                cells.append(HysteresisCell(
                    hold_off=H, U=U, regime=regime, n_tasksets=len(ladders),
                    service_ratio=paired_compare(idle_sr_means, hyst_sr),
                    level_trans=paired_compare(idle_lt_means, hyst_lt),
                ))
    return cells
