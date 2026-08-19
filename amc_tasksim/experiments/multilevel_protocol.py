"""The revised multi-level protocol (research/mode_optimization.tex, "Revised
Protocol"): feasibility characterisation, the scheme-vs-AMC-RH regime map, and
bounded-gain configuration selection.

Superseded question, and why this module looks the way it does
----------------------------------------------------------------
The protocol this replaces (``sec:phase-a`` / ``sec:phase-b`` in the tex) was
designed to optimise *trigger severity spacing*. A structural pilot on the
built k-level engine found that spacing has **no effect at all** under
``drop_set_shed_early``: that policy requires one drop set feasible at every
operating severity including the pinned deepest one, so by monotonicity the
intermediate severities never bind and the drop set -- the only thing that
distinguishes one level from another, since operating budgets are an analysis
quantity and never a run-time throttle -- is identical regardless of spacing.
Confirmed on 16/16 task sets; service ratio agreed to six decimal places
across k in {2,3,4,5}.

So there is no severity-spacing search left to run. What this module measures
instead:

1. **Feasibility** (design-time, no simulation): what fraction of task sets
   admit a shed-early ladder versus a genuinely progressive one, per operating
   point. This is the primary applicability result, because progressive
   ladders are where grading actually shows up (shed-early collapses every
   level to the same drop set) but they do not always exist.
2. **The regime map**: service ratio against two-level AMC-RA, paired, across
   utilisation and deadline tightness, for both operating points. This is the
   one comparison whose effect exceeds this study's 5% practical-significance
   threshold (contract.EFFECT_FLOOR); everything else measured in the pilot
   (ordering choice, greedy vs exhaustive, x_LO) does not.
3. **Bounded-gain configuration selection**: on small task sets, how much
   could any search possibly win over the default (execution-time-ordered
   shed-early) configuration? Answered by exhaustive enumeration over the
   LO-criticality subsets, which is only tractable at small n.

Every function reuses ``amc_tasksim.experiments.contract`` for pairing and
``amc_tasksim.experiments.sweep.build_population`` for population generation,
so results compose with the rest of the measurement contract rather than
restating it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

from amc_tasksim.experiments.contract import (
    CANONICAL,
    PairedResult,
    aggregate_by_taskset,
    paired_compare,
)
from amc_tasksim.experiments.sweep import build_population
from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import (
    amc_rtb,
    is_nontrivial_amc_taskset,
    severity_trigger,
)
from amc_tasksim.scheduling.drop_sets import (
    Ordering,
    by_execution_time,
    drop_ladder,
    drop_set_shed_early,
)
from amc_tasksim.scheduling.priority import assign_audsley_opa, assign_deadline_monotonic
from amc_tasksim.simulation.engine import AMC_RA, simulate
from amc_tasksim.simulation.multilevel import (
    SeverityLadder,
    _operating_severities,
    simulate_multilevel,
)

#: The ordering the structural pilot found best under the default policy
#: (drop_set_shed_early, U=0.7, CF=2): +0.96% to +1.54% service ratio over
#: by_utilisation, resolved under the conservative operating point at a 24
#: task set x 5 seed pilot. by_utilisation minimises the number of tasks
#: shed, which is not the same as minimising the number of jobs lost; see
#: drop_sets.by_execution_time's own docstring.
DEFAULT_ORDERING: Ordering = by_execution_time

#: D_i pulled down to this fraction of the way from R_i(LO) to T_i for the
#: "tight" deadline regime. Preserves normal-mode schedulability exactly
#: while removing most of a LO-criticality task's slack, which is where
#: deadline termination and the two operating points actually diverge --
#: at implicit deadlines (alpha=1.0) terminations are near zero regardless
#: of operating point.
TIGHT_ALPHA: float = 0.3


# ---------------------------------------------------------------------------
# Population generation
# ---------------------------------------------------------------------------


def build_tight_population(
    n_replicates: int,
    U: float,
    seed: int,
    alpha: float = TIGHT_ALPHA,
    max_attempts_factor: int = 800,
    **kwargs,
) -> list[TaskSet]:
    """Qualifying task sets with deadlines tightened toward R_i(LO).

    Mirrors ``sweep.build_population``'s qualification (fails single-criticality
    FPPS, passes AMC-rtb under OPA), but computes R_i(LO) under
    deadline-monotonic priorities first so ``D_i`` can be set to
    ``R_i(LO) + alpha * (T_i - R_i(LO))`` before OPA is assigned -- OPA itself
    depends on D, so the tightening has to happen first, not after.
    """
    base = seed + int(round(U * 1000)) * 1_000_003 + int(round(alpha * 100)) * 7
    accepted: list[TaskSet] = []
    attempts = 0
    limit = n_replicates * max_attempts_factor

    while len(accepted) < n_replicates and attempts < limit:
        ts = generate_taskset(U=U, rng_seed=base + attempts, **kwargs)
        attempts += 1
        assign_deadline_monotonic(ts)
        r_lo = amc_rtb(ts).r_lo
        if any(r_lo[i] > ts.T[i] for i in range(ts.n)):
            continue  # R_i(LO) already exceeds T_i; tightening would be vacuous
        if alpha < 1.0:
            ts.D = [int(r_lo[i] + alpha * (ts.T[i] - r_lo[i])) for i in range(ts.n)]
        if not assign_audsley_opa(ts):
            continue
        if not is_nontrivial_amc_taskset(ts, use_opa=False):
            continue
        accepted.append(ts)

    return accepted


# ---------------------------------------------------------------------------
# Ladder construction for the two policies
# ---------------------------------------------------------------------------


def shed_early_ladder(
    ts: TaskSet,
    ordering: Ordering = DEFAULT_ORDERING,
    require_lo_deadlines: bool = False,
) -> Optional[SeverityLadder]:
    """The k=2-equivalent ladder: one drop set, shed at the shallowest rung.

    ``require_lo_deadlines=True`` is operating point A (conservative, LDM=0 by
    analysis); ``False`` is operating point B (termination). See
    task_model.tex "Two Operating Points".
    """
    S = drop_set_shed_early(ts, [1.0], 0.0, ordering, require_lo_deadlines=require_lo_deadlines)
    if S is None:
        return None
    return SeverityLadder(
        severities=[0.0],
        operating_severities=[1.0],
        thresholds=[severity_trigger(ts, 0.0)],
        drop_sets=[S],
        x_lo=0,
    )


def progressive_ladder(
    ts: TaskSet,
    severities: Sequence[float],
    ordering: Ordering = DEFAULT_ORDERING,
    require_lo_deadlines: bool = False,
) -> Optional[SeverityLadder]:
    """A genuinely graded ladder: S_1 subsetneq S_{k-1} where feasible.

    Unlike shed_early_ladder, this is not always feasible: a task first shed at
    a deep rung has an infinite shed instant for a third of HI-criticality
    tasks at severity 1 (task_model.tex "Shed-Aware Response Time"), so the
    single-phase carry-in bound cannot always certify it. See
    :func:`feasibility_fraction`.
    """
    oper = _operating_severities(list(severities))
    ds = drop_ladder(
        ts, oper, ordering,
        trigger_severities=list(severities),
        charge_carry_in=True,
        require_lo_deadlines=require_lo_deadlines,
    )
    if ds is None:
        return None
    return SeverityLadder(
        severities=list(severities),
        operating_severities=oper,
        thresholds=[severity_trigger(ts, x) for x in severities],
        drop_sets=[set(d) for d in ds],
        x_lo=0,
    )


# ---------------------------------------------------------------------------
# Stage 1: feasibility characterisation (design-time, no simulation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityResult:
    """Fraction of a population admitting each ladder construction.

    Attributes:
        n: Task sets tested.
        shed_early: Fraction admitting the shed-early ladder.
        progressive: Fraction admitting the progressive ladder, or None if
            ``severities`` was not supplied.
        mean_shed_pct_shed_early: Mean % of LO-criticality tasks shed, over
            task sets where shed_early succeeded.
        mean_shed_pct_progressive_l1: Mean % shed at the shallowest rung under
            the progressive ladder, over task sets where it succeeded -- the
            number that shows grading is doing something (it is markedly
            smaller than the shed-early figure whenever both exist).
    """

    n: int
    shed_early: float
    progressive: Optional[float]
    mean_shed_pct_shed_early: Optional[float]
    mean_shed_pct_progressive_l1: Optional[float]


def feasibility_fraction(
    tasksets: Sequence[TaskSet],
    require_lo_deadlines: bool,
    severities: Optional[Sequence[float]] = None,
    ordering: Ordering = DEFAULT_ORDERING,
) -> FeasibilityResult:
    """What fraction of ``tasksets`` admit each ladder construction.

    No simulation is run -- admissibility is decided by response-time analysis
    alone, which is what makes this stage cheap enough to run over a wide grid
    before committing to Stage 2's simulation budget.
    """
    n = len(tasksets)
    se_ok = pg_ok = 0
    se_shed: list[float] = []
    pg_shed: list[float] = []
    for ts in tasksets:
        lo_n = sum(1 for i in range(ts.n) if ts.criticality[i] == "LO")
        se = drop_set_shed_early(ts, [1.0], 0.0, ordering, require_lo_deadlines=require_lo_deadlines)
        if se is not None:
            se_ok += 1
            if lo_n:
                se_shed.append(100.0 * len(se) / lo_n)
        if severities is not None:
            oper = _operating_severities(list(severities))
            pg = drop_ladder(
                ts, oper, ordering,
                trigger_severities=list(severities),
                charge_carry_in=True,
                require_lo_deadlines=require_lo_deadlines,
            )
            if pg is not None:
                pg_ok += 1
                if lo_n:
                    pg_shed.append(100.0 * len(pg[0]) / lo_n)

    return FeasibilityResult(
        n=n,
        shed_early=se_ok / n if n else 0.0,
        progressive=(pg_ok / n if n else 0.0) if severities is not None else None,
        mean_shed_pct_shed_early=(sum(se_shed) / len(se_shed)) if se_shed else None,
        mean_shed_pct_progressive_l1=(sum(pg_shed) / len(pg_shed)) if pg_shed else None,
    )


# ---------------------------------------------------------------------------
# Stage 2: the regime map
# ---------------------------------------------------------------------------


def service_ratios(
    ts: TaskSet,
    ladder: SeverityLadder,
    seeds: Sequence[int],
    duration: int,
    fp: float,
) -> list[float]:
    """Per-seed service ratio for one task set under the k-level engine."""
    return [
        simulate_multilevel(ts, ladder, duration=duration, seed=sd, fp=fp).service_ratio
        for sd in seeds
    ]


def amc_ra_service_ratios(
    ts: TaskSet,
    seeds: Sequence[int],
    duration: int,
    fp: float,
) -> list[float]:
    """Per-seed service ratio for one task set under two-level AMC-RA.

    Computed the same way as MultiLevelResult.service_ratio (Comp / Exp), from
    the two-level engine's own release/jne/lo_terminated counts, so the two are
    comparable term for term.
    """
    r_lo = amc_rtb(ts).r_lo
    out = []
    for sd in seeds:
        r = simulate(
            ts, duration=duration, seed=sd, fp=fp,
            mode_protocol=AMC_RA(r_lo), skip_quiet=False,
        )
        n = sum(r.lo_releases_per_task)
        out.append((n - r.jne - r.lo_terminated) / n if n else 1.0)
    return out


@dataclass(frozen=True)
class RegimeCell:
    """One (U, deadline regime, operating point, policy) cell of the map."""

    U: float
    regime: str  # "implicit" or "tight"
    operating_point: str  # "A" (conservative) or "B" (termination)
    policy: str  # "shed_early" or "progressive"
    n_tasksets: int
    vs_amc_ra: PairedResult


def regime_map(
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
) -> list[RegimeCell]:
    """The main experiment: scheme vs two-level AMC-RA across the grid.

    For each utilisation level, both deadline regimes (implicit and tight,
    TIGHT_ALPHA), both operating points, and both policies (shed_early always
    attempted; progressive attempted only where feasible), builds one
    population and runs every configuration on the *same* task sets and seeds
    -- the pairing the measurement contract requires -- then compares against
    two-level AMC-RA computed on that same population.
    """
    cells: list[RegimeCell] = []
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

            # AMC-RA depends only on the task set, not on the operating point or
            # policy, so it is computed once per task set and re-paired against
            # whichever feasible subset each cell below turns out to use.
            baseline = {id(ts): amc_ra_service_ratios(ts, seeds, duration, fp) for ts in tasksets}

            for op_name, strict in (("A", True), ("B", False)):
                for policy, builder, extra in (
                    ("shed_early", shed_early_ladder, {}),
                    ("progressive", progressive_ladder, {"severities": progressive_severities}),
                ):
                    scheme_raw, base_raw = [], []
                    for ts in tasksets:
                        kwargs = dict(extra)
                        if policy == "progressive":
                            severities = kwargs.pop("severities")
                            lad = builder(ts, severities, require_lo_deadlines=strict)
                        else:
                            lad = builder(ts, require_lo_deadlines=strict)
                        if lad is None:
                            continue
                        scheme_raw.append(service_ratios(ts, lad, seeds, duration, fp))
                        base_raw.append(baseline[id(ts)])
                    if len(scheme_raw) < 2:
                        continue
                    scheme_agg = aggregate_by_taskset(
                        [v for run in scheme_raw for v in run], len(seeds)
                    )
                    base_agg = aggregate_by_taskset(
                        [v for run in base_raw for v in run], len(seeds)
                    )
                    cells.append(RegimeCell(
                        U=U, regime=regime, operating_point=op_name, policy=policy,
                        n_tasksets=len(scheme_raw),
                        vs_amc_ra=paired_compare(base_agg, scheme_agg),
                    ))
    return cells


# ---------------------------------------------------------------------------
# Stage 3: bounded-gain configuration selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundedGainResult:
    """How much any search could win over the default configuration.

    Only tractable at small n, since it enumerates every LO-criticality
    subset. The default is shed_early with DEFAULT_ORDERING; "best" is the
    exhaustive-search optimum among *feasible* subsets, evaluated by simulated
    service ratio (not by the design-time shed count, which is a proxy).
    """

    n_tasksets: int
    default_vs_best: PairedResult
    mean_default_shed_size: float
    mean_best_shed_size: float


def bounded_gain(
    require_lo_deadlines: bool,
    n_tasksets: int = 16,
    n_tasks: int = 10,
    U: float = 0.7,
    CP: float = 0.5,
    CF: float = 2.0,
    seeds: Sequence[int] = tuple(range(5)),
    duration: int = 200_000,
    fp: float = 0.2,
    seed: int = 0,
) -> Optional[BoundedGainResult]:
    """Exhaustive optimum against the default shed-early configuration.

    Uses simulated service ratio (not R_i(chi) analysis) to pick the best
    feasible subset, so it also captures anything a smarter *drop policy*
    (not just a smarter ordering) could win -- the widest bound this stage can
    give, at n small enough to make the enumeration tractable.
    """
    tasksets, _ = build_population(n_tasksets, U, seed, n=n_tasks, CP=CP, CF=CF)
    if not tasksets:
        return None

    default_raw: list[list[float]] = []
    best_raw: list[list[float]] = []
    default_sizes: list[int] = []
    best_sizes: list[int] = []

    for ts in tasksets:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        default_lad = shed_early_ladder(ts, require_lo_deadlines=require_lo_deadlines)
        if default_lad is None:
            continue

        from amc_tasksim.scheduling.amc_rtb import severity_budgets
        from amc_tasksim.scheduling.drop_sets import is_feasible

        th = severity_trigger(ts, 0.0)
        best_mean = -1.0
        best_vals: Optional[list[float]] = None
        best_size = None
        for r in range(len(lo) + 1):
            for combo in itertools.combinations(lo, r):
                S = set(combo)
                cif = lambda i, _S=S: dict.fromkeys(_S, th[i])
                if not is_feasible(
                    ts, severity_budgets(ts, 1.0), S, cif, require_lo_deadlines
                ):
                    continue
                lad = SeverityLadder(
                    severities=[0.0], operating_severities=[1.0],
                    thresholds=[th], drop_sets=[S], x_lo=0,
                )
                vals = service_ratios(ts, lad, seeds, duration, fp)
                m = sum(vals) / len(vals)
                if m > best_mean:
                    best_mean, best_vals, best_size = m, vals, len(S)

        if best_vals is None:
            continue
        default_raw.append(service_ratios(ts, default_lad, seeds, duration, fp))
        default_sizes.append(len(default_lad.drop_sets[0]))
        best_raw.append(best_vals)
        best_sizes.append(best_size)

    if not default_raw:
        return None

    default_agg = aggregate_by_taskset([v for run in default_raw for v in run], len(seeds))
    best_agg = aggregate_by_taskset([v for run in best_raw for v in run], len(seeds))
    return BoundedGainResult(
        n_tasksets=len(default_raw),
        default_vs_best=paired_compare(default_agg, best_agg),
        mean_default_shed_size=sum(default_sizes) / len(default_sizes),
        mean_best_shed_size=sum(best_sizes) / len(best_sizes),
    )
