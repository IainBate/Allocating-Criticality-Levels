"""Which LO-criticality tasks to shed at a degradation level, and why that set.

Two-level AMC abandons *every* LO-criticality task when a HI-criticality job
overruns, because the analysis has no way to say which ones it could afford to
keep. Response-time analysis at the degraded budget does have that ability: at
severity ``x``, charge every task ``C_i(x)`` and ask which LO-criticality tasks
can be retained while

1. every HI-criticality task still meets its deadline (**clause 1**), and
2. every *retained* LO-criticality task still meets its deadline (**clause 2**).

Clause 1 is the certification constraint and is always enforced. Clause 2 is
off by default (``require_lo_deadlines``), because under deadline termination a
retained LO-criticality job that cannot finish is terminated rather than
delivered late -- so its deadline is not a safety obligation, and requiring it
buys certainty of service rather than safety. It charges heavily for that
certainty: a task that cannot be certified must be shed outright, leaving no
state between "guaranteed to complete" and "never runs".

Measured over 199 non-trivial AMC-rtb-schedulable task sets (n=12, U=0.7,
CF=2.0), shedding at the shallowest rung with carry-in charged:

==================  ===============  =================
Clause 2            LO tasks shed    LO tasks retained
==================  ===============  =================
enforced                     68.3%              31.7%
relaxed (default)            47.6%              52.4%
two-level AMC               100.0%               0.0%
==================  ===============  =================

Carry-in: what a shed task still costs
--------------------------------------
Abandonment is on release, so a job of a shed task already in flight when its
level is entered runs to completion and still interferes. Charging it nothing
-- the ``carry_in=None`` path below, and everything this module did originally
-- is sound only in the limit where shedding takes effect instantaneously.
Charging it honestly is the ``carry_in`` mapping, which generalises the
``min(.,.)`` term of AMC-rtb's equation (2) to a per-task shed instant.

Doing so changes the design conclusion, not merely the numbers. A task shed at
rung r stops releasing at R_i(chi_r) for that rung's *trigger* severity; that
bound is infinite for 33% of HI-criticality tasks at chi = 1, and where there
is no bound the analysis must charge the task in full, which no amount of
further shedding can repair. Measured over 199 non-trivial AMC-rtb-schedulable
task sets (n=12, U=0.7, CF=2.0), trigger severities (0, 0.5, 1):

=========================  ============  ==============
Policy                     Infeasible    LO tasks shed
=========================  ============  ==============
progressive (per rung)       93 (47%)             71.5%
shed-early (all at rung 1)    0 ( 0%)             68.3%
two-level AMC                 0 ( 0%)            100.0%
=========================  ============  ==============

On the 106 sets where both are feasible, shedding early sheds strictly less on
12 and the same on 94 -- never more. So grading belongs in the *budgets*, not
in the drop sets: progressive shedding buys nothing the analysis can certify
and costs feasibility on nearly half the population. See
:func:`drop_set_shed_early`, which is feasible exactly when AMC-rtb passes.

The optimistic figures this module reported before carry-in was charged (0.0%
shed at severity <= 0.25, 10.0% at 0.50, 33.7% at 0.75, 56.3% at 1.00) remain
reachable via ``charge_carry_in=False``, and are retained only as the
sound-in-the-limit comparison.

On the choice of ordering
-------------------------
Shedding the highest-utilisation task first matched the exhaustive minimum in
every case tested (40 task sets x 3 severities, 0% excess), while shedding by
priority -- the conventional choice -- dropped 73-78% more tasks than necessary.
That is not a proof of optimality, and it minimises the *number of tasks* shed,
which is not the same as minimising the number of *jobs* lost: a task's job
count over a run is inversely proportional to its period. Where the objective is
JNE rather than task count, prefer :func:`by_execution_time`, which sheds the
most interference per job forgone.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import _num_jobs, severity_budgets, severity_trigger

#: An ordering picks the next task to shed from the retained candidates.
Ordering = Callable[[TaskSet, list[int]], int]


def _releases_before(task_T: int, instant: float) -> float:
    """Jobs of a task with period ``task_T`` released in [0, instant).

    ``math.inf`` when the instant is unbounded, which makes the ``min`` in
    :func:`response_time` select the full charge.
    """
    if instant == math.inf:
        return math.inf
    return _num_jobs(task_T, int(instant))


def response_time(
    taskset: TaskSet,
    i: int,
    budgets: list[int],
    dropped: set[int],
    carry_in: Optional[dict[int, float]] = None,
    max_iterations: int = 10000,
) -> float:
    r"""Response time of task ``i`` at a level, given what has been shed.

    Higher-priority HI-criticality tasks are charged ``budgets``; retained
    LO-criticality tasks are charged C(LO).

    What a *shed* task contributes depends on ``carry_in``:

    * ``None`` -- nothing at all. Sound only in the limit where shedding takes
      effect instantaneously, which no implementation achieves: abandonment is
      on release, so a job already in flight when the level is entered runs to
      completion and still interferes.
    * a mapping -- ``carry_in[k]`` is the instant, measured from task ``i``'s
      busy-period start, at which task ``k`` stopped releasing. Task ``k`` is
      then charged for the jobs it released before that instant, bounded by the
      window under analysis:

      .. math:: \min(\lceil w/T_k \rceil, \lceil s_k/T_k \rceil) \, C_k^{lo}

      This is exactly the term AMC-rtb's equation (2) applies to the two-level
      case, generalised to a per-task shed instant. At ``carry_in[k] = 0`` it
      recovers the ``None`` behaviour; at ``math.inf`` it charges ``k`` in full,
      as though it had never been shed.

    Because a task shed at a *shallower* rung has a smaller :math:`s_k`, this is
    what makes shedding early pay for itself in the analysis rather than only in
    simulation.

    Returns:
        The response time, or ``math.inf`` once it passes D_i (at which point
        the task is unschedulable and the exact value carries no information).
    """
    own = budgets[i] if taskset.criticality[i] == "HI" else taskset.C_lo[i]
    hp = [j for j in range(taskset.n) if taskset.priority[j] < taskset.priority[i]]

    w = own
    for _ in range(max_iterations):
        interference = 0
        for j in hp:
            if taskset.criticality[j] == "HI":
                interference += _num_jobs(taskset.T[j], w) * budgets[j]
            elif j not in dropped:
                interference += _num_jobs(taskset.T[j], w) * taskset.C_lo[j]
            elif carry_in is not None:
                jobs = min(_num_jobs(taskset.T[j], w),
                           _releases_before(taskset.T[j], carry_in.get(j, math.inf)))
                interference += jobs * taskset.C_lo[j]
        nxt = own + interference
        if nxt == w:
            return float(w)
        w = nxt
        if w > taskset.D[i]:
            return math.inf
    return math.inf


def is_feasible(
    taskset: TaskSet,
    budgets: list[int],
    dropped: set[int],
    carry_in_for: Optional[Callable[[int], dict[int, float]]] = None,
    require_lo_deadlines: bool = False,
) -> bool:
    """Whether the level's deadline obligations hold for this drop set.

    **Clause 1** -- every HI-criticality task meets its deadline -- is the
    certification constraint and is always checked.

    **Clause 2** -- every *retained* LO-criticality task meets its deadline --
    is controlled by ``require_lo_deadlines`` and defaults to **off**, because
    under deadline termination it is not a safety obligation. A LO-criticality
    job that cannot finish in time is terminated at its deadline rather than
    allowed to complete late, so it never delivers a late result; it delivers
    none, which is the same outcome as being abandoned on release and is
    counted the same way (JNC). Two-level AMC already takes exactly this action
    when it abandons LO jobs on a mode change -- only the moment differs.

    Requiring clause 2 therefore buys no safety. What it buys is certainty of
    *service*, and it charges heavily: a task that cannot be certified must be
    shed outright, so every LO task is either "guaranteed to complete" or
    "never runs", with nothing between. Leaving it off admits the third state --
    retained, allowed to run, terminated if it cannot finish -- which is where
    graded criticality levels actually live, and without which the intermediate
    tiers collapse.

    Args:
        carry_in_for: Given the index of the task under analysis, the shed
            instants to charge (see :func:`response_time`). ``None`` keeps the
            optimistic instantaneous-shedding assumption.
        require_lo_deadlines: Enforce clause 2 as well, reproducing the earlier
            and stricter criterion.
    """
    for i in range(taskset.n):
        if taskset.criticality[i] == "LO":
            if i in dropped or not require_lo_deadlines:
                continue
        carry_in = carry_in_for(i) if carry_in_for is not None else None
        if response_time(taskset, i, budgets, dropped, carry_in) > taskset.D[i]:
            return False
    return True


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------


def by_utilisation(taskset: TaskSet, candidates: list[int]) -> int:
    """Highest C_i(LO)/T_i first -- sheds the most interference per task."""
    return max(candidates, key=lambda i: (taskset.C_lo[i] / taskset.T[i], -i))


def by_execution_time(taskset: TaskSet, candidates: list[int]) -> int:
    """Highest C_i(LO) first -- sheds the most interference per *job* forgone.

    A task contributes jobs at rate 1/T_i and interference at rate C_i/T_i, so
    interference shed per job lost is C_i. Prefer this when the objective counts
    jobs (JNE) rather than tasks.
    """
    return max(candidates, key=lambda i: (taskset.C_lo[i], -i))


def by_priority(taskset: TaskSet, candidates: list[int]) -> int:
    """Lowest priority first -- the conventional choice, and a poor one here.

    Retained for comparison: it sheds substantially more than necessary,
    because the least important task is rarely the one causing the interference.
    """
    return max(candidates, key=lambda i: (taskset.priority[i], -i))


# ---------------------------------------------------------------------------
# Drop-set construction
# ---------------------------------------------------------------------------


def drop_set_at_severity(
    taskset: TaskSet,
    x: float,
    ordering: Ordering = by_utilisation,
    charge_carry_in: bool = False,
    lo_may_trigger: bool = False,
    require_lo_deadlines: bool = False,
) -> Optional[set[int]]:
    """Smallest LO-criticality set to shed at severity ``x``, by ``ordering``.

    Sheds tasks one at a time until both deadline obligations hold. Because
    shedding only ever removes interference, the search is monotone and
    terminates.

    Args:
        taskset: Task set with priorities assigned.
        x: Severity in [0, 1]; 0 charges C(LO) and 1 charges C(HI).
        ordering: Which retained LO-criticality task to shed next.
        charge_carry_in: Charge a shed task for the jobs it released before the
            level was entered (see :func:`response_time`). False recovers the
            earlier instantaneous-shedding assumption, which is optimistic.
        lo_may_trigger: Whether a retained LO-criticality task is permitted to
            fire this rung -- the ``x_LO`` knob. It tightens the analysis as
            well as enabling the mechanism: only a task that can fire the rung
            can bound when the rung fired within its own busy period.

    Returns:
        The task indices to shed, or None if the level is infeasible even with
        every LO-criticality task shed -- which means the task set fails
        AMC-rtb at this severity and no drop policy can rescue it.
    """
    budgets = severity_budgets(taskset, x)
    lo = [i for i in range(taskset.n) if taskset.criticality[i] == "LO"]

    carry_in_for = None
    if charge_carry_in:
        threshold = severity_trigger(taskset, x)

        def carry_in_for(i: int) -> dict[int, float]:  # noqa: F811
            # The level is entered no later than R_i(x) in task i's own busy
            # period: if i is still incomplete then, i fires the rung itself.
            # A retained LO task can only make that argument for a rung it is
            # permitted to fire, so without lo_may_trigger it gets no bound and
            # shed tasks are charged in full.
            if taskset.criticality[i] == "HI" or lo_may_trigger:
                return dict.fromkeys(lo, threshold[i])
            return dict.fromkeys(lo, math.inf)

    dropped: set[int] = set()
    while not is_feasible(taskset, budgets, dropped, carry_in_for, require_lo_deadlines):
        candidates = [i for i in lo if i not in dropped]
        if not candidates:
            return None
        dropped.add(ordering(taskset, candidates))
    return dropped


def drop_set_shed_early(
    taskset: TaskSet,
    operating_severities: list[float],
    trigger_severity: float = 0.0,
    ordering: Ordering = by_utilisation,
    require_lo_deadlines: bool = False,
    max_iterations: int = 10000,
) -> Optional[set[int]]:
    """One drop set, shed in full at the shallowest rung, feasible at every rung.

    The alternative to :func:`drop_ladder`'s progressive shedding, and the one
    the carry-in analysis endorses. A task shed at rung 1 stops releasing at
    R_i(0) = R_i(LO), which is finite for every task in a normal-mode
    schedulable set, so every shed task gets a usable carry-in bound. A task
    shed at a deep rung stops releasing at R_i(chi) for that rung's trigger
    severity, which is *infinite* for a third of HI-criticality tasks at
    chi = 1 -- no bound, so the analysis must charge it in full, and the rung is
    then unschedulable no matter what is shed.

    Measured over 199 non-trivial AMC-rtb-schedulable sets (n=12, U=0.7, CF=2),
    trigger severities (0, 0.5, 1):

    ==================  ============  ==============
    Policy              Infeasible    LO tasks shed
    ==================  ============  ==============
    progressive           93 (47%)             71.5%
    shed-early             0 ( 0%)             68.3%
    two-level AMC          0 ( 0%)            100.0%
    ==================  ============  ==============

    On the 106 sets where both are feasible, shedding early sheds strictly less
    on 12 and the same on 94 -- it is never worse. Grading therefore belongs in
    the *budgets*, not in the drop sets: shedding progressively buys nothing the
    analysis can certify, and costs feasibility on nearly half the population.

    Args:
        taskset: Task set with priorities assigned.
        operating_severities: The severity each rung is analysed at. The
            returned set must satisfy both deadline obligations at all of them.
        trigger_severity: Severity of the rung at which shedding happens, whose
            R_i fixes the carry-in bound. 0.0 is the shallowest rung.
        ordering: Which retained LO-criticality task to shed next.

    Returns:
        The task indices to shed, or None if no set works at every rung.
    """
    lo = [i for i in range(taskset.n) if taskset.criticality[i] == "LO"]
    threshold = severity_trigger(taskset, trigger_severity)
    budgets = [severity_budgets(taskset, x) for x in operating_severities]

    dropped: set[int] = set()
    for _ in range(max_iterations):

        def carry_in_for(i: int) -> dict[int, float]:
            return dict.fromkeys(dropped, threshold[i])

        if all(is_feasible(taskset, b, dropped, carry_in_for, require_lo_deadlines)
               for b in budgets):
            return dropped
        candidates = [i for i in lo if i not in dropped]
        if not candidates:
            return None
        dropped.add(ordering(taskset, candidates))
    return None


def drop_ladder(
    taskset: TaskSet,
    severities: list[float],
    ordering: Ordering = by_utilisation,
    trigger_severities: Optional[list[float]] = None,
    charge_carry_in: bool = False,
    x_lo: int = 0,
    require_lo_deadlines: bool = False,
) -> Optional[list[set[int]]]:
    """Drop sets for a whole severity ladder, forced to be nested.

    The multi-level scheme requires S_1 subset of S_2 subset of ... : a task
    shed at one level stays shed at deeper ones, so a transition only ever adds
    to the drop set and the runtime decision is incremental.

    For any ordering matching :data:`Ordering` this already holds without help.
    An ordering is a pure function of (task set, remaining candidates) and every
    level starts from the same full candidate set, so it sheds the same sequence
    at every severity -- a deeper level merely stops later, making each level a
    prefix of one sequence, and prefixes nest. Carrying the previous level's set
    forward is therefore redundant today; it is retained so that nesting still
    holds if an ordering ever consults the severity or the partial drop set,
    which would break the prefix argument. ``test_levels_are_prefixes_of_one_
    shed_sequence`` pins the argument so its failure is visible.

    Args:
        taskset: Task set with priorities assigned.
        severities: Ascending severities, one per level above normal.
        ordering: Which retained LO-criticality task to shed next.
        trigger_severities: The severities at which the rungs *fire*, which are
            not in general the severities their drop sets are computed at: a
            rung's drop set is charged at its OPERATING severity (look-ahead to
            the next rung) while it fires at its own, lower, TRIGGER severity.
            The shed instant must come from the trigger severity -- using the
            operating one both loosens the bound and makes it infinite far more
            often. Defaults to ``severities``.
        charge_carry_in: Charge each shed task for the jobs it released before
            the rung that shed it was entered (see :func:`response_time`).
            Default False: see the module docstring's "Carry-in" section for
            why the bound is not yet strong enough to switch on.
        x_lo: Deepest rung index a retained LO-criticality task is permitted to
            fire. Rungs at or below it give retained LO tasks a bound on when
            shedding took effect; deeper rungs do not, so their shed tasks are
            charged in full when analysing a retained LO task. ``0`` disables
            LO triggering entirely and is the conservative default.

    Returns:
        One drop set per severity, nested; or None if any level is infeasible.

    Raises:
        ValueError: If ``severities`` is not ascending.
    """
    if any(a > b for a, b in zip(severities, severities[1:])):
        raise ValueError(f"severities must be ascending, got {severities}")

    lo = [i for i in range(taskset.n) if taskset.criticality[i] == "LO"]
    triggers = severities if trigger_severities is None else trigger_severities
    if len(triggers) != len(severities):
        raise ValueError(
            f"trigger_severities has length {len(triggers)}, expected {len(severities)}"
        )
    thresholds = [severity_trigger(taskset, x) for x in triggers] if charge_carry_in else []

    ladder: list[set[int]] = []
    carried: set[int] = set()
    shed_rung: dict[int, int] = {}  # task -> index of the rung that shed it
    for rung, x in enumerate(severities):
        budgets = severity_budgets(taskset, x)
        dropped = set(carried)  # inherit, so the ladder is nested by construction

        carry_in_for = None
        if charge_carry_in:

            def carry_in_for(i: int, _rung: int = rung) -> dict[int, float]:  # noqa: F811
                # Each shed task is charged against the rung that shed IT, not
                # the rung under analysis -- a task shed early stops releasing
                # early, so it contributes fewer carried-in jobs. That is what
                # makes shedding early pay for itself here rather than only in
                # simulation.
                out: dict[int, float] = {}
                for k in dropped:
                    y = shed_rung.get(k, _rung)
                    bounded = taskset.criticality[i] == "HI" or y <= x_lo
                    out[k] = thresholds[y][i] if bounded else math.inf
                return out

        while not is_feasible(taskset, budgets, dropped, carry_in_for,
                              require_lo_deadlines):
            candidates = [i for i in lo if i not in dropped]
            if not candidates:
                return None
            chosen = ordering(taskset, candidates)
            dropped.add(chosen)
            shed_rung[chosen] = rung
        ladder.append(dropped)
        carried = dropped
    return ladder
