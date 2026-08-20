"""Event-driven simulator for the k-level severity-ladder scheme.

Generalises ``amc_tasksim.simulation.engine`` from two modes (normal,
degraded) to k levels $L_0, \\ldots, L_{k-1}$, each with its own severity
$\\chi_x$ (task_model.tex, "Multi-Level Degradation Model"). Deliberately a
separate module rather than a modification of engine.py: the two-level
engine is exact, heavily tested, and everything else in this project is
built on it, so changing it in place risks that foundation for no benefit --
the two coexist, and this module's own tests hold it to reproducing the
two-level engine exactly at k=2 rather than trusting the two are equivalent.

Scope of this first implementation, stated rather than discovered later:

- **Exit is direct, not cascade.** From any level $L_x > 0$, an idle instant
  (empty run-queue) returns straight to $L_0$, matching AMC-RA's exit rule
  generalised across levels. Per-level cascade exit (returning to whichever
  level's own threshold is no longer exceeded) is a reasonable extension,
  deferred rather than guessed at -- see the "Mode Transition Protocol"
  section of task_model.tex, which leaves the choice open explicitly.
- **Abandonment is on release**, matching engine.py and the AMC-RH paper's
  model: a task entering a level's drop set has its *future* releases
  abandoned; a job already running continues.

- **A LO-criticality job is terminated at its deadline**, never allowed to
  complete late -- a late result may be worse than no result. This is the
  same action two-level AMC already takes when it abandons LO jobs on a mode
  change; only the moment differs. Two consequences: the objective is JNC
  (jobs that delivered no result, however they failed to) rather than JNE
  alone; and WastedCPU is live rather than zero, because terminating at the
  deadline *does* abandon a job mid-execution, which abandon-on-release never
  does. An earlier version of this docstring claimed WastedCPU was always
  zero; that was true only while termination was miscounted as a deadline
  miss.

- **Metrics are normalised.** A raw count of abandoned jobs scales with the
  run duration and with the tasks' periods, so it is not comparable across
  configurations. Every rate is expressed against ``lo_expected``, the jobs a
  perfect scheduler would have completed, which depends only on the duration
  and the periods -- see :attr:`MultiLevelResult.service_ratio`.
- **No skip_quiet.** Exact simulation only. The two-level engine's
  fast-forward has documented preconditions (severity_trigger's own
  soundness argument); extending it to a nested drop-set model is future
  work, not assumed here.
- **Admissibility is not enforced by the engine.** The engine executes
  whatever ladder (severities, drop sets) it is given; it is the caller's
  responsibility to build one via :func:`build_ladder`, which uses the
  admissibility criterion in ``amc_tasksim.scheduling.drop_sets``. An
  inadmissible ladder will simulate -- possibly with HI-criticality deadline
  misses -- rather than being rejected, so that a test can deliberately
  construct one and confirm HDM becomes nonzero (see test_multilevel.py).
"""

from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import severity_trigger
from amc_tasksim.scheduling.drop_sets import (
    Ordering,
    by_utilisation,
    drop_ladder,
    drop_set_shed_early,
)
from amc_tasksim.simulation.engine import _TriggerSchedule


@dataclass
class SeverityLadder:
    """A concrete k-level scheme.

    Two different severities are tracked per level, and conflating them is
    the mistake this split exists to prevent (found by trying to make k=2
    reproduce the classic scheme exactly, and failing until they were
    separated):

    - **Trigger severity** ``severities[x]`` decides *when* level x+1 fires:
      R_i(severities[x]) is the threshold (task_model.tex,
      "Severity-Indexed Response Time"). ``severities[0] == 0`` is pinned --
      level 1 must fire at exactly the classic AMC trigger point.
    - **Operating severity** ``operating_severities[x]`` decides what level
      x+1 actually *does* once entered: the enforced execution budget
      C_i(operating_severities[x]), and the severity charged when computing
      that level's admissible drop set. It is derived, not chosen
      independently: ``operating_severities[x] = severities[x+1]`` for every
      level except the deepest, whose operating severity is pinned to 1 --
      the deepest level must always grant full C_i(HI) protection, which is
      what Theorem 1's HI-criticality guarantee needs, independent of
      whatever its own trigger severity happens to be.

    Without this split, k=2 cannot reproduce the classic scheme at all:
    trigger severity 0 (required) and operating severity 1 (required for
    full HI protection) would otherwise have to be the same number.

    Attributes:
        severities: Length k-1, ascending, ``severities[0] == 0.0``.
        operating_severities: Length k-1, derived as described above.
        thresholds: Length k-1; ``thresholds[x][i]`` is R_i(severities[x]).
            May be ``math.inf`` for an unreachable level.
        drop_sets: Length k-1, nested (``drop_sets[x] <= drop_sets[x+1]``);
            ``drop_sets[x]`` is the set of LO-criticality task indices
            abandoned at level x+1 and every deeper level.
        x_lo: Deepest rung a LO-criticality task may fire. 0 restricts
            triggering to HI-criticality tasks, which is the classic rule and
            the only setting under which total JNE is bounded by AMC-RH's.
    """

    severities: list[float]
    operating_severities: list[float]
    thresholds: list[list[float]]
    drop_sets: list[set[int]]
    x_lo: int = 0

    @property
    def k(self) -> int:
        return len(self.severities) + 1

    def may_trigger(self, taskset: TaskSet, task_id: int, level: int) -> bool:
        """Whether ``task_id`` is permitted to fire the rung entering ``level``.

        Two conditions, both from task_model.tex's "Mode Transition Protocol":

        1. The rung must not abandon the task -- a task never demands a level
           that sheds it, which is what keeps the abandoned set a strict prefix
           of the criticality order (Corollary 1'(b)).
        2. A LO-criticality task may only reach rungs at or below ``x_lo``.
           At ``x_lo = 0`` this is the classic rule: HI-criticality tasks only,
           and the original JNE containment bound holds.
        """
        if task_id in self.drop_sets[level - 1]:
            return False
        return taskset.criticality[task_id] == "HI" or level <= self.x_lo


def _operating_severities(severities: list[float]) -> list[float]:
    """Derive operating severities from trigger severities (see SeverityLadder)."""
    return [severities[x + 1] for x in range(len(severities) - 1)] + [1.0]


def build_ladder(
    taskset: TaskSet,
    severities: list[float],
    ordering: Ordering = by_utilisation,
    drop_policy: str = "admissible",
    charge_carry_in: bool = False,
    x_lo: int = 0,
    require_lo_deadlines: bool = False,
) -> Optional[SeverityLadder]:
    """Build a ladder from trigger severities.

    Args:
        taskset: Task set with priorities assigned.
        severities: Ascending trigger severities, ``severities[0] == 0.0``.
        ordering: Which LO-criticality task to shed next when a level is
            inadmissible without shedding; see ``scheduling.drop_sets``.
            Only used when ``drop_policy == "admissible"``.
        drop_policy: ``"admissible"`` (default) computes the minimal drop set
            at each level's operating severity (the actual scheme this
            project studies). ``"full"`` drops every LO-criticality task at
            every level regardless of severity, matching the classic AMC
            scheme's response exactly -- used to isolate whether a
            discrepancy against the two-level engine comes from this
            engine's mechanics or from the admissible-drop-set optimisation
            itself (see test_multilevel.py's exact-reproduction tests).
            ``"shed_early"`` computes one set and sheds it at the shallowest
            rung, which is the only progressive-shedding alternative the
            carry-in analysis can certify (drop_sets.drop_set_shed_early).
        charge_carry_in: Charge a shed task for jobs released before its rung
            was entered. Only meaningful with ``drop_policy="admissible"``;
            ``"shed_early"`` always charges it.
        x_lo: Deepest rung a retained LO-criticality task may fire.
        require_lo_deadlines: Selects between the scheme's two operating points.
            False (default) is the *termination* point: retained LO-criticality
            tasks are best-effort and are terminated at their deadline if they
            cannot finish. True is the *conservative* point: every retained
            LO-criticality task is certified to complete, so LDM is zero by
            analysis and no new execution model is needed -- it is sound under
            the original two-level semantics. The conservative point sheds more
            and delivers less; both beat two-level AMC. See the paper's
            "Two Operating Points".

    Returns:
        A :class:`SeverityLadder`, or ``None`` if some level is infeasible
        even with every LO-criticality task shed.

    Raises:
        ValueError: If ``severities`` is empty, not ascending, does not start
            at exactly 0.0, or ``drop_policy`` is not recognised.
    """
    if not severities:
        raise ValueError("severities must have at least one entry (k >= 2)")
    if severities[0] != 0.0:
        raise ValueError(
            f"severities[0] must be exactly 0.0 (level 1 = classic AMC "
            f"trigger), got {severities[0]}"
        )
    if any(a > b for a, b in zip(severities, severities[1:])):
        raise ValueError(f"severities must be ascending, got {severities}")

    operating = _operating_severities(severities)

    if drop_policy == "admissible":
        drop_sets = drop_ladder(
            taskset, operating, ordering,
            trigger_severities=severities,
            charge_carry_in=charge_carry_in,
            x_lo=x_lo,
        )
        if drop_sets is None:
            return None
    elif drop_policy == "shed_early":
        # One set, shed in full at the shallowest rung. The policy the carry-in
        # analysis endorses: progressive shedding cannot be certified at deep
        # rungs because R_i(chi) is infinite there for a third of HI tasks, so
        # no carry-in bound exists. See drop_sets.drop_set_shed_early.
        early = drop_set_shed_early(taskset, operating, severities[0], ordering,
                                    require_lo_deadlines=require_lo_deadlines)
        if early is None:
            return None
        drop_sets = [set(early) for _ in severities]
    elif drop_policy == "full":
        all_lo = {i for i in range(taskset.n) if taskset.criticality[i] == "LO"}
        drop_sets = [set(all_lo) for _ in severities]
    else:
        raise ValueError(f"Unknown drop_policy: {drop_policy!r}")

    thresholds = [severity_trigger(taskset, chi) for chi in severities]
    return SeverityLadder(
        severities=list(severities),
        operating_severities=operating,
        thresholds=thresholds,
        drop_sets=drop_sets,
        x_lo=x_lo,
    )


@dataclass
class MultiLevelJob:
    """A single job release under the k-level scheme.

    Unlike engine.Job, ``budget`` is mutable: a HI-criticality job's enforced
    budget is ``C_i(severity of the job's task at the CURRENT level)`` and
    increases if the system escalates while the job is still active
    (task_model.tex, "Execution Budget Enforcement"). A LO-criticality job's
    budget is fixed at C_i(LO) for its whole life.
    """

    task_id: int
    seq: int
    release: int
    deadline: int
    c_lo: int
    c_hi: int
    criticality: str
    priority: int
    exec_time: int
    budget: int
    hi_behaviour: bool = False
    busy_start: int = 0
    executed: int = 0

    @property
    def remaining(self) -> int:
        return self.exec_time - self.executed

    def sort_key(self) -> tuple[int, int]:
        return (self.priority, self.seq)


@dataclass
class MultiLevelState:
    """The run-time state the k-level protocol operates over.

    Attributes:
        time: Current simulation time.
        active: Jobs released but not yet completed, priority order.
        running: The job currently executing, or None if idle.
        level: Current level, 0 (normal) to k-1 (deepest).
    """

    time: int = 0
    active: list[MultiLevelJob] = field(default_factory=list)
    running: Optional[MultiLevelJob] = None
    level: int = 0


@dataclass
class MultiLevelResult:
    """Metrics from a single k-level simulation run.

    Attributes:
        nid: Legacy NiD -- entries into any degraded level (>=1) FROM level 0
            only, for direct comparison with the two-level baseline.
        level_trans: Number of level changes, any direction, any depth --
            see metrics_objective.md, "Why a churn term, and why LevelTrans
            rather than NiD".
        tid: Fraction of duration spent at level >= 1.
        jne: LO-criticality jobs abandoned on release at any level >= 1.
        lo_terminated: LO-criticality jobs that started executing but were
            *terminated at their deadline* without completing. Not a deadline
            miss: a LO-criticality job is never permitted to complete late,
            because a late result may be worse than no result. Termination is
            the same action two-level AMC already takes when it abandons a job
            on a mode change -- only the moment differs.
        hdm: HI-criticality jobs that missed their deadline (should be zero
            for an admissible ladder; nonzero is the signal that it was not).
        wasted_cpu: Execution consumed by jobs that were terminated at their
            deadline without completing -- work done and thrown away. Non-zero
            precisely because deadline termination abandons a job
            *mid-execution*, which abandon-on-release never does.
        hi_releases_per_task: HI-criticality releases per task.
        lo_releases_per_task: LO-criticality releases per task, including
            abandoned ones.
        lo_completed: LO-criticality jobs that completed within their deadline.
        hi_trigger_events: Releases that drew HI-criticality behaviour.
        level_ticks: Time spent at each level, index 0..k-1; sums to duration.
        duration: Simulation duration.
        budget_overruns: Defensive check; should always be zero.
    """

    nid: int = 0
    level_trans: int = 0
    tid: float = 0.0
    jne: int = 0
    lo_terminated: int = 0
    lo_completed: int = 0
    hdm: int = 0
    wasted_cpu: int = 0
    hi_releases_per_task: list[int] = field(default_factory=list)
    lo_releases_per_task: list[int] = field(default_factory=list)
    hi_trigger_events: int = 0

    @property
    def lo_expected(self) -> int:
        """Jobs a perfect scheduler would have completed: every LO release.

        The reference the objective is measured against. It depends only on the
        run duration and the tasks' periods, not on anything the scheme did, so
        it is a fixed denominator that makes runs of different lengths and
        different task sets comparable.
        """
        return sum(self.lo_releases_per_task)

    @property
    def jnc(self) -> int:
        """Jobs Not Completed: released in a perfect world, but no result.

        Defined by difference against :attr:`lo_expected` rather than by adding
        up the ways a job can fail, so it cannot silently omit one. A job fails
        to complete by being abandoned on release (``jne``), by being terminated
        at its deadline having run (``lo_terminated``), or by still being in
        flight when the run ends -- the last is a horizon artefact bounded by
        the number of tasks, and is why this is not asserted equal to
        ``jne + lo_terminated``.

        Prefer :attr:`jnc_pct`: a raw count scales with duration and with the
        tasks' periods, so it is not comparable across configurations.
        """
        return self.lo_expected - self.lo_completed

    @property
    def service_ratio(self) -> float:
        """Fraction of LO-criticality jobs that delivered a result. 1.0 is perfect."""
        exp = self.lo_expected
        return self.lo_completed / exp if exp else 1.0

    @property
    def jnc_pct(self) -> float:
        """JNC as a percentage of the jobs a perfect scheduler would have completed."""
        exp = self.lo_expected
        return 100.0 * self.jnc / exp if exp else 0.0

    @property
    def jne_pct(self) -> float:
        """Jobs abandoned on release, as a percentage of all LO releases."""
        exp = self.lo_expected
        return 100.0 * self.jne / exp if exp else 0.0

    @property
    def lo_terminated_pct(self) -> float:
        """Jobs terminated at their deadline, as a percentage of all LO releases."""
        exp = self.lo_expected
        return 100.0 * self.lo_terminated / exp if exp else 0.0

    @property
    def wasted_cpu_pct(self) -> float:
        """Processor time spent on work that was thrown away, as a % of duration."""
        return 100.0 * self.wasted_cpu / self.duration if self.duration else 0.0
    level_ticks: list[int] = field(default_factory=list)
    duration: int = 0
    budget_overruns: int = 0

    #: Populated only when `simulate_multilevel(measure_cascade_opportunity=True)`.
    #: See `_natural_level` for the definition of "justified".
    overdegraded_ticks: int = 0
    overdegraded_level_ticks: int = 0
    overdegraded_events: int = 0
    max_overdegraded_gap: int = 0
    #: Of `jne`, the count abandoned on release while state.level > natural_level
    #: -- i.e. LO jobs dropped after the evidence that justified the current
    #: level had already gone stale. Unlike the *_ticks/pct diagnostics above,
    #: this is not confounded by legitimate backlog-drain time: a busy queue
    #: draining after its triggering job left is not itself a cost, but a job
    #: dropped *during* that tail is a real, countable instance of exactly the
    #: service loss a smarter exit rule could plausibly have avoided.
    overdegraded_jne: int = 0
    #: Split of overdegraded_jne by what kind of exit would have been needed
    #: to avoid it. `overdegraded_jne_full_exit`: natural_level was 0 -- an
    #: AMC-RH-style *full* exit to L0 (exit_policy="amc_rh", safe -- see
    #: safety_proof.md) would have admitted this job instead. The remainder,
    #: `overdegraded_jne - overdegraded_jne_full_exit`, needed a *partial*
    #: demotion to an intermediate level (0 < natural_level < state.level) --
    #: only possible for k > 2, and only safe if a real cascade-exit
    #: mechanism is built and proven; not implemented, so this is a
    #: diagnostic-only estimate of what such a mechanism could additionally
    #: recover on top of full exit alone.
    overdegraded_jne_full_exit: int = 0

    @property
    def total_hi_releases(self) -> int:
        return sum(self.hi_releases_per_task)

    @property
    def total_lo_releases(self) -> int:
        return sum(self.lo_releases_per_task)

    @property
    def overdegraded_pct(self) -> float:
        """% of time at level >= 1 spent deeper than currently active evidence
        justifies -- i.e. `state.level > natural_level` (see `_natural_level`).

        An UPPER bound on what an instant, unconditionally-safe cascade exit
        could recover, not an achievable or safety-checked gain: it assumes
        demoting the moment justifying evidence disappears is free and safe,
        which is exactly the question a cascade-exit design would still have
        to settle. Zero unless `measure_cascade_opportunity=True` was passed
        to `simulate_multilevel`.
        """
        degraded = sum(self.level_ticks[1:]) if self.level_ticks else 0
        return 100.0 * self.overdegraded_ticks / degraded if degraded else 0.0

    @property
    def overdegraded_level_pct(self) -> float:
        """Depth-weighted `overdegraded_pct`: unjustified level-ticks as a % of
        total accumulated level-depth-time (sum of level x level_ticks[level]).

        Distinguishes "over-degraded a little, often" from "a lot, rarely" --
        `overdegraded_pct` alone cannot, since it does not weight by how many
        levels deep the gap was.
        """
        depth = sum(x * t for x, t in enumerate(self.level_ticks))
        return 100.0 * self.overdegraded_level_ticks / depth if depth else 0.0

    @property
    def overdegraded_jne_pct(self) -> float:
        """`overdegraded_jne` as a % of `lo_expected` -- directly comparable to
        `jne_pct` and `jnc_pct`, since it is a subset of the same `jne` count.
        """
        exp = self.lo_expected
        return 100.0 * self.overdegraded_jne / exp if exp else 0.0

    @property
    def overdegraded_jne_full_exit_pct(self) -> float:
        """`overdegraded_jne_full_exit` as a % of `lo_expected` -- the part of
        overdegraded_jne_pct that a safe, already-implemented full exit alone
        recovers. `overdegraded_jne_pct - overdegraded_jne_full_exit_pct` is
        what a real cascade mechanism could additionally recover, at most.
        """
        exp = self.lo_expected
        return 100.0 * self.overdegraded_jne_full_exit / exp if exp else 0.0


def _natural_level(
    ladder: SeverityLadder,
    taskset: TaskSet,
    active: list[MultiLevelJob],
    level: int,
    now: int,
) -> int:
    """The deepest level currently justified by active, eligible evidence.

    `escalate_if_triggered` asks "is level x+1 justified" while walking up.
    This asks the same question while walking down from the current level: for
    each y <= level, is there still an active, eligible job whose threshold for
    y has been reached? `state.level` is a ratchet -- it only rises within a
    degraded excursion, and resets to 0 only on a full idle instant -- so it
    can lag behind what current evidence alone would justify once the job that
    caused an escalation completes while the system remains at that level.
    This recovers that lag as a number instead of leaving it unmeasured.

    A job whose threshold justifies level y also justifies every shallower
    level y' < y: thresholds are non-decreasing in severity for a fixed task
    (ladder property (B)), and `may_trigger` eligibility at y implies
    eligibility at y' < y (drop sets are nested, and x_lo is a single cutoff).
    So the set of levels justified by the active population is a prefix of
    [0, level], and the search below can stop at the first (deepest-first) hit.
    """
    for y in range(level, 0, -1):
        thresholds = ladder.thresholds[y - 1]
        for job in active:
            if not ladder.may_trigger(taskset, job.task_id, y):
                continue
            th = thresholds[job.task_id]
            if math.isinf(th):
                continue
            if job.busy_start + int(math.ceil(th)) <= now:
                return y
    return 0


def _next_evidence_reappearance(
    ladder: SeverityLadder,
    taskset: TaskSet,
    active: list[MultiLevelJob],
    now: int,
) -> Optional[int]:
    """Earliest instant `_natural_level(..., level=state.level, ...)` could next
    become nonzero, given the currently active population -- i.e. the earliest
    not-yet-reached level-1 threshold among eligible active jobs.

    Only level 1 needs checking, not every level up to state.level: a job
    reaching a deeper threshold has already reached its (non-decreasing, per
    ladder property (B)) level-1 threshold, so level 1 is always the earliest
    possible transition from natural_level=0 to natural_level>0. This is what
    lets a hold-off exit policy (`exit_policy="hysteresis"`) track "evidence
    has been continuously clear since T" without missing a reappearance that
    happens between two otherwise-scheduled events -- the same role
    `escalate_if_triggered`'s own one-level-ahead lookahead plays for entry.

    Not required for safety: whatever policy uses this still re-checks
    `_natural_level` fresh at the instant it actually decides to exit, so a
    missed or delayed wake-up here can only make a hold-off's timing less
    precise, never unsafe. See `simulate_multilevel`'s `exit_policy` docs and
    safety_proof.md's tempered-exit corollary.
    """
    thresholds = ladder.thresholds[0]
    best: Optional[int] = None
    for job in active:
        if not ladder.may_trigger(taskset, job.task_id, 1):
            continue
        th = thresholds[job.task_id]
        if math.isinf(th):
            continue
        expiry = job.busy_start + int(math.ceil(th))
        if expiry > now:
            best = expiry if best is None else min(best, expiry)
    return best


def _draw_exec_time(rng: np.random.Generator, taskset: TaskSet, i: int, hi_behaviour: bool) -> int:
    """Draw a job's execution time -- identical model to engine.py."""
    c_lo = taskset.C_lo[i]
    c_hi = taskset.C_hi[i]
    if hi_behaviour:
        if c_hi <= c_lo:
            return c_hi
        return int(rng.integers(c_lo, c_hi + 1))
    bcet = min(taskset.BCET[i], c_lo)
    if bcet >= c_lo:
        return c_lo
    return int(rng.integers(bcet, c_lo + 1))


def simulate_multilevel(
    taskset: TaskSet,
    ladder: SeverityLadder,
    duration: int = 10**6,
    seed: Optional[int] = None,
    fp: float = 1e-4,
    release_offsets: Optional[list[Optional[int]]] = None,
    trace: Optional[list[tuple]] = None,
    measure_cascade_opportunity: bool = False,
    exit_policy: str = "idle",
    hold_off: int = 0,
) -> MultiLevelResult:
    """Simulate a task set under the k-level severity-ladder scheme.

    Entry escalates one level at a time -- from L_x to L_{x+1} when an active
    HI-criticality job's busy-period-relative time reaches
    ``ladder.thresholds[x][task_id]`` -- but re-checks immediately after each
    escalation, so multiple levels can be crossed at the same instant if
    warranted (this is sound because thresholds are non-decreasing in
    severity: reaching a deeper threshold implies every shallower one was
    already reached). Exit is direct: an idle instant returns to L_0 from any
    level, matching AMC-RA generalised across levels (module docstring).

    Args:
        taskset: Task set. Priorities are assigned if absent.
        ladder: A :class:`SeverityLadder` from :func:`build_ladder`.
        duration: Simulation duration in ticks.
        seed: Random seed.
        fp: Per-job probability a HI-criticality release exhibits HI behaviour.
        release_offsets: Per-task phase of the first release.
        trace: If given, appended with (time, event, task_id) tuples.
        measure_cascade_opportunity: If True, also compute the `overdegraded_*`
            diagnostics on the result -- how much of the run's degraded time
            was spent deeper than currently active evidence justifies, an
            upper bound on what a hypothetical instant cascade-exit could
            recover (see `_natural_level`). Off by default: it adds an
            O(k x n) check per event, on top of the existing escalation
            check of the same order, so runs that do not need it do not pay
            for it.
        exit_policy: ``"idle"`` (default): exit to L0 only on an idle instant,
            matching AMC-RA generalised across levels -- today's behaviour,
            unchanged. ``"amc_rh"``: additionally exit to L0 the instant no
            active, eligible job's threshold remains met (`_natural_level`
            reaches 0), matching AMC-RH's own exit rule -- see
            `docs/exit_strategy_analysis.md` for why this is a large,
            resolved effect, and `safety_proof.md`'s evidence-cleared-exit
            corollary for why it is safe: it only ever exits straight to L0,
            never to an intermediate level, so it inherits AMC-RH's own
            existing safety argument rather than needing a new one.
            ``"hysteresis"``: exit to L0 once evidence has been continuously
            clear for at least `hold_off` ticks, tempering "amc_rh"'s
            immediate exit to trade back some of its oscillation increase for
            some of its service-ratio gain -- see `docs/exit_strategy_analysis.md`
            for the retrospective data motivating this, and `safety_proof.md`'s
            tempered-exit corollary for why it is safe regardless of
            `hold_off` or of any imprecision in *when* the hold-off's own
            deadline is detected: the actual exit is always gated on a fresh
            `_natural_level` check at the instant it fires, not on the
            hold-off bookkeeping, so a missed wake-up can only delay an exit
            (still safe), never cause an early one.
        hold_off: Minimum ticks evidence must have been continuously clear
            before `exit_policy="hysteresis"` will exit. Ignored otherwise.
            `hold_off=0` is exactly `exit_policy="amc_rh"`; a `hold_off`
            larger than any realistic gap is exactly `exit_policy="idle"` --
            both are exact equivalences, not approximations, verified in
            `tests/simulation/test_multilevel.py`.

    Returns:
        A :class:`MultiLevelResult`.
    """
    if exit_policy not in ("idle", "amc_rh", "hysteresis"):
        raise ValueError(f"Unknown exit_policy: {exit_policy!r}")
    if not taskset.priority:
        from amc_tasksim.scheduling.priority import assign_deadline_monotonic

        assign_deadline_monotonic(taskset)

    rng = np.random.default_rng(seed)
    n = taskset.n
    k = ladder.k

    result = MultiLevelResult(duration=duration)
    result.hi_releases_per_task = [0] * n
    result.lo_releases_per_task = [0] * n
    result.level_ticks = [0] * k

    offsets = [0] * n
    for i in range(n):
        if release_offsets is not None and release_offsets[i] is not None:
            offsets[i] = int(release_offsets[i])
    release_heap: list[tuple[int, int]] = [(offsets[i], i) for i in range(n)]
    heapq.heapify(release_heap)

    priority = taskset.priority

    # Which releases exhibit HI-criticality behaviour must be drawn by the SAME
    # process engine.py uses, not merely the same distribution: exact
    # reproduction at k=2 compares traces, and a per-release rng.random() and
    # _TriggerSchedule's geometric jumps consume the stream at different rates,
    # so the two engines disagree about *which* jobs overrun. Measured before
    # this was shared: identical results at fp=0, but 10 vs 9 HI-behaviour
    # releases at fp=1e-3, which no scheduling fix could ever reconcile.
    schedule = _TriggerSchedule(taskset, fp, offsets, rng)

    state = MultiLevelState()
    active: list[MultiLevelJob] = state.active
    seq = 0
    level_entered_at = 0  # when the CURRENT level was entered, for level_ticks accounting
    was_overdegraded = False  # for overdegraded_events: counts 0 -> positive transitions

    def escalate_if_triggered(now: int) -> None:
        """Escalate one level if the next level's threshold is already reached.

        Called before releases at `now` and after each release/escalation, so
        cascading escalations at the same instant are each individually
        applied -- mirrors engine.py's enter_if_triggered.
        """
        nonlocal level_entered_at
        while state.level < k - 1:
            target = state.level + 1
            thresholds = ladder.thresholds[target - 1]
            best: Optional[int] = None
            for job in active:
                if not ladder.may_trigger(taskset, job.task_id, target):
                    continue
                th = thresholds[job.task_id]
                if math.isinf(th):
                    continue
                expiry = job.busy_start + int(math.ceil(th))
                if expiry <= now:
                    best = now
                    break
                best = expiry if best is None else min(best, expiry)
            if best is None or best > now:
                return
            # Escalate: record level_ticks for the level we are LEAVING.
            result.level_ticks[state.level] += now - level_entered_at
            if state.level == 0:
                result.nid += 1
                if trace is not None:
                    trace.append((now, "enter_degraded", -1))
            result.level_trans += 1
            state.level = target
            level_entered_at = now
            # No budget adjustment on escalation: a HI job is entitled to
            # C_i(HI) from release, so there is nothing for a deeper level to
            # grant it. Operating severity survives only as the severity at
            # which the level's drop set is computed.
            if trace is not None:
                trace.append((now, f"level_{target}", -1))

    def exit_if_idle(now: int) -> None:
        """Exit to L0, on idle always, and additionally on cleared evidence
        under `exit_policy="amc_rh"` -- see MultiLevelResult / the module
        docstring's "Exit policy" note. Only ever exits straight to L0: a
        partial demotion to an intermediate level is a different, still-open
        safety question (safety_proof.md, "Corollary: Evidence-Cleared Exit
        is Safe" -- scoped explicitly to full exit, not partial).
        """
        nonlocal level_entered_at
        if state.level == 0:
            return
        exit_now = not active
        if not exit_now and exit_policy == "amc_rh":
            exit_now = _natural_level(ladder, taskset, active, state.level, now) == 0
        if exit_now:
            result.level_ticks[state.level] += now - level_entered_at
            result.level_trans += 1
            state.level = 0
            level_entered_at = now
            if trace is not None:
                trace.append((now, "exit_degraded", -1))

    while state.time < duration:
        now = state.time

        # --- deadline expiries ---
        if active:
            expired = [j for j in active if j.deadline <= now]
            if expired:
                for job in expired:
                    if job.criticality == "HI":
                        # A HI-criticality job passing its deadline is a real
                        # failure of the certification, not a policy action.
                        result.hdm += 1
                        event = "deadline_miss"
                    else:
                        # Deliberate: a LO-criticality job is terminated rather
                        # than allowed to complete late. Whatever it executed
                        # is wasted, which is what makes wasted_cpu non-zero.
                        result.lo_terminated += 1
                        result.wasted_cpu += job.executed
                        event = "lo_terminated"
                    active.remove(job)
                    if trace is not None:
                        trace.append((now, event, job.task_id))

        # --- exit (direct, on idle) before releases at `now` ---
        exit_if_idle(now)

        # --- entry / escalation ---
        state.running = active[0] if active else None
        escalate_if_triggered(now)

        # --- releases at `now`, priority order ---
        due: list[int] = []
        while release_heap and release_heap[0][0] <= now:
            due.append(heapq.heappop(release_heap)[1])
        if len(due) > 1:
            due.sort(key=lambda i: priority[i])

        for i in due:
            crit = taskset.criticality[i]
            if crit == "HI":
                result.hi_releases_per_task[i] += 1
            else:
                result.lo_releases_per_task[i] += 1

            hi_behaviour = False
            if crit == "HI" and schedule.enabled():
                release_index = (now - offsets[i]) // taskset.T[i]
                if schedule.triggers(i, release_index):
                    hi_behaviour = True
                    schedule.consume(i, rng)
            # Drawn for every release, including one about to be abandoned, so
            # the stream depends only on the release sequence -- engine.py does
            # the same, and it is what keeps the two comparable.
            exec_time = _draw_exec_time(rng, taskset, i, hi_behaviour)

            dropped = (
                crit == "LO"
                and state.level > 0
                and i in ladder.drop_sets[state.level - 1]
            )
            if dropped:
                result.jne += 1
                if measure_cascade_opportunity and state.level > 0:
                    # Freshly computed, not reused from the end-of-iteration
                    # `gap` below: that value describes the *forward* interval
                    # starting at `now`, computed after every release this
                    # instant is processed, whereas this release's own drop
                    # decision happens mid-batch, possibly before a later
                    # release in the same instant escalates the level further.
                    nat = _natural_level(ladder, taskset, active, state.level, now)
                    if nat < state.level:
                        result.overdegraded_jne += 1
                        if nat == 0:
                            result.overdegraded_jne_full_exit += 1
                if trace is not None:
                    trace.append((now, "drop", i))
            else:
                if hi_behaviour:
                    result.hi_trigger_events += 1
                if exec_time == 0 and crit == "LO":
                    # No work to do, so it has trivially delivered its result at
                    # its release instant (engine.py treats it the same way).
                    # It must be counted, or the completion accounting would
                    # report a shortfall that never happened.
                    result.lo_completed += 1
                if exec_time > 0:
                    # Enforced budget is the task's own WCET, never the level's
                    # severity budget. C_i(chi_x) is an *analysis* quantity --
                    # what the response-time bound charges -- not a run-time
                    # throttle. Capping a HI job at C_i(chi_0) = C_i(LO) in L_0
                    # would make an overrun physically impossible, so no
                    # threshold could ever be reached and the ladder would never
                    # leave L_0 (measured: 1469 truncated jobs, 0 transitions).
                    # This matches engine.py, which is why its budget_overruns
                    # really is always zero.
                    budget = taskset.C_hi[i] if crit == "HI" else taskset.C_lo[i]
                    job = MultiLevelJob(
                        task_id=i, seq=seq, release=now, deadline=now + taskset.D[i],
                        c_lo=taskset.C_lo[i], c_hi=taskset.C_hi[i], criticality=crit,
                        priority=taskset.priority[i], exec_time=exec_time, budget=budget,
                        hi_behaviour=hi_behaviour,
                    )
                    seq += 1
                    if job.exec_time > job.budget:
                        result.budget_overruns += 1
                        warnings.warn(
                            f"task {i}: execution time {job.exec_time} exceeds its "
                            f"enforced budget {job.budget}; job truncated", stacklevel=2,
                        )
                        job.exec_time = job.budget
                    key = job.sort_key()
                    idx = 0
                    while idx < len(active) and active[idx].sort_key() < key:
                        idx += 1
                    job.busy_start = now if idx == 0 else active[idx - 1].busy_start
                    active.insert(idx, job)
                    if trace is not None:
                        trace.append((now, "release", i))
                    # With x_lo > 0 a LO-criticality release can also become a
                    # trigger, so the check is no longer HI-only.
                    if crit == "HI" or ladder.x_lo > 0:
                        escalate_if_triggered(now)
            heapq.heappush(release_heap, (now + taskset.T[i], i))

        state.running = active[0] if active else None

        # --- cascade-opportunity diagnostic (read-only; see MultiLevelResult) ---
        gap = 0
        if measure_cascade_opportunity and state.level > 0:
            gap = state.level - _natural_level(ladder, taskset, active, state.level, now)

        # --- next event ---
        next_t = duration
        if release_heap and release_heap[0][0] < next_t:
            next_t = release_heap[0][0]
        for job in active:
            if job.deadline < next_t:
                next_t = job.deadline
        if state.running is not None:
            done = now + state.running.remaining
            if done < next_t:
                next_t = done
        if state.level < k - 1:
            thresholds = ladder.thresholds[state.level]
            for job in active:
                if not ladder.may_trigger(taskset, job.task_id, state.level + 1):
                    continue
                th = thresholds[job.task_id]
                if math.isinf(th):
                    continue
                expiry = job.busy_start + int(math.ceil(th))
                if expiry < next_t:
                    next_t = expiry

        if next_t <= now:
            warnings.warn(f"simulation made no progress at t={now}; forcing advance", stacklevel=2)
            next_t = now + 1

        if measure_cascade_opportunity:
            dt = next_t - now
            if gap > 0:
                result.overdegraded_ticks += dt
                result.overdegraded_level_ticks += gap * dt
                if gap > result.max_overdegraded_gap:
                    result.max_overdegraded_gap = gap
                if not was_overdegraded:
                    result.overdegraded_events += 1
                    if trace is not None:
                        n_hi = sum(1 for j in active if j.criticality == "HI")
                        trace.append((now, "overdegraded_start", len(active) - n_hi))
                was_overdegraded = True
            else:
                was_overdegraded = False

        if state.running is not None:
            state.running.executed += next_t - now
        state.time = next_t

        if state.running is not None and state.running.remaining <= 0:
            active.remove(state.running)
            if state.running.criticality == "LO":
                result.lo_completed += 1
            if trace is not None:
                trace.append((state.time, "complete", state.running.task_id))
            state.running = None

    result.level_ticks[state.level] += duration - level_entered_at
    result.tid = sum(result.level_ticks[1:]) / duration if duration > 0 else 0.0
    return result


def _severity_budget(taskset: TaskSet, task_id: int, chi: float) -> int:
    """C_i(chi) for a single task -- inlined copy of severity_budgets' formula
    to avoid recomputing the whole task set's vector on every escalation."""
    c_lo, c_hi = taskset.C_lo[task_id], taskset.C_hi[task_id]
    return max(c_lo, int(round(c_lo + chi * (c_hi - c_lo))))
