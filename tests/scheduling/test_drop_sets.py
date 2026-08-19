"""Tests for deriving drop sets by response-time analysis.

The claim under test is that a drop set can be *computed* rather than guessed,
such that at every degradation level both HI-criticality tasks and retained
LO-criticality tasks provably meet their deadlines. If that holds, LDM is zero
by construction and the multi-level scheme has no JNE-against-LDM trade-off.
"""

from __future__ import annotations

import itertools
import math

import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import (
    amc_rtb,
    busy_period_bound,
    is_nontrivial_amc_taskset,
    normal_mode_schedulable,
    severity_budgets,
)
from amc_tasksim.scheduling.drop_sets import (
    by_execution_time,
    by_priority,
    by_utilisation,
    drop_ladder,
    drop_set_at_severity,
    drop_set_shed_early,
    is_feasible,
    response_time,
)
from amc_tasksim.scheduling.priority import assign_deadline_monotonic


def population(count: int, n: int = 10, **kw) -> list[TaskSet]:
    params = dict(n=n, CP=0.5, U=0.7, CF=2.0)
    params.update(kw)
    out = []
    for seed in range(6000):
        ts = generate_taskset(rng_seed=seed, **params)
        assign_deadline_monotonic(ts)
        if normal_mode_schedulable(ts) and busy_period_bound(ts) is not None:
            out.append(ts)
        if len(out) == count:
            break
    assert len(out) == count
    return out


@pytest.fixture(scope="module")
def tasksets() -> list[TaskSet]:
    return population(30)


@pytest.fixture(scope="module")
def small() -> list[TaskSet]:
    """Small enough to enumerate every subset of the LO-criticality tasks."""
    return population(20, n=8)


SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# The two deadline obligations
# ---------------------------------------------------------------------------


def test_at_severity_zero_nothing_needs_shedding(tasksets):
    """x=0 charges C(LO), which is the normal mode R1 already guarantees."""
    for ts in tasksets:
        assert drop_set_at_severity(ts, 0.0) == set()


def test_returned_drop_set_is_always_feasible(tasksets):
    """The post-condition: both obligations hold for whatever is returned."""
    for ts in tasksets:
        for x in SEVERITIES:
            dropped = drop_set_at_severity(ts, x)
            assert dropped is not None
            assert is_feasible(ts, severity_budgets(ts, x), dropped)


def test_retained_lo_tasks_meet_their_deadlines(tasksets):
    """LDM is zero by construction -- the point of the second obligation."""
    for ts in tasksets:
        for x in SEVERITIES:
            dropped = drop_set_at_severity(ts, x)
            budgets = severity_budgets(ts, x)
            for i in range(ts.n):
                if ts.criticality[i] == "LO" and i not in dropped:
                    assert response_time(ts, i, budgets, dropped) <= ts.D[i]


def test_hi_tasks_meet_their_deadlines_at_every_severity(tasksets):
    """The safety obligation, which is the binding one."""
    for ts in tasksets:
        for x in SEVERITIES:
            dropped = drop_set_at_severity(ts, x)
            budgets = severity_budgets(ts, x)
            for i in range(ts.n):
                if ts.criticality[i] == "HI":
                    assert response_time(ts, i, budgets, dropped) <= ts.D[i]


def test_dropping_only_ever_removes_interference(tasksets):
    """Monotonicity, which is what makes the greedy construction terminate."""
    for ts in tasksets[:10]:
        budgets = severity_budgets(ts, 0.75)
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        for i in range(ts.n):
            base = response_time(ts, i, budgets, set())
            more = response_time(ts, i, budgets, set(lo))
            assert more <= base


def test_shedding_everything_leaves_only_hi_interference(tasksets):
    """With every LO task shed, a HI task sees HI-criticality interference alone.

    Checked against an independent closed-form recurrence rather than against
    the function under test, so the two have to agree for a reason.
    """
    for ts in tasksets[:12]:
        lo = {i for i in range(ts.n) if ts.criticality[i] == "LO"}
        budgets = severity_budgets(ts, 1.0)
        for i in range(ts.n):
            if ts.criticality[i] != "HI":
                continue
            hp_hi = [
                j
                for j in range(ts.n)
                if ts.priority[j] < ts.priority[i] and ts.criticality[j] == "HI"
            ]
            w = budgets[i]
            for _ in range(10000):
                nxt = budgets[i] + sum(
                    -(-w // ts.T[j]) * budgets[j] for j in hp_hi
                )
                if nxt == w or w > ts.D[i]:
                    break
                w = nxt
            expected = float(w) if w <= ts.D[i] else float("inf")
            assert response_time(ts, i, budgets, lo) == expected


# ---------------------------------------------------------------------------
# Severity grades the response
# ---------------------------------------------------------------------------


def test_higher_severity_never_sheds_less(tasksets):
    for ts in tasksets:
        sizes = [len(drop_set_at_severity(ts, x)) for x in SEVERITIES]
        # Not required to be strictly nested set-wise here, only non-decreasing
        # in size; drop_ladder is what guarantees nesting.
        assert sizes == sorted(sizes), f"{sizes} not non-decreasing"


def test_mild_overruns_need_no_degradation(tasksets):
    """The case for grading: a small overrun costs no LO-criticality work."""
    unaffected = sum(1 for ts in tasksets if not drop_set_at_severity(ts, 0.25))
    assert unaffected > len(tasksets) // 2, (
        f"only {unaffected}/{len(tasksets)} task sets survive x=0.25 intact"
    )


def test_full_severity_sheds_less_than_two_level_amc(tasksets):
    """Even at C(HI), the scheme keeps work that AMC-RH would abandon."""
    total_lo = kept = 0
    for ts in tasksets:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        dropped = drop_set_at_severity(ts, 1.0)
        total_lo += len(lo)
        kept += len(lo) - len(dropped)
    assert kept > 0, "at maximum severity the scheme degenerates to AMC-RH"


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------


def test_utilisation_ordering_matches_the_exhaustive_minimum(small):
    """Verified against brute force, since the greedy is not proven optimal."""
    for ts in small:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        for x in (0.5, 1.0):
            budgets = severity_budgets(ts, x)
            greedy = drop_set_at_severity(ts, x, by_utilisation)
            best = None
            for size in range(len(lo) + 1):
                for combo in itertools.combinations(lo, size):
                    if is_feasible(ts, budgets, set(combo)):
                        best = size
                        break
                if best is not None:
                    break
            assert len(greedy) == best, (
                f"greedy shed {len(greedy)}, optimum is {best}"
            )


def test_priority_ordering_is_measurably_worse(small):
    """Guards the rationale for not defaulting to the conventional choice."""
    util = prio = 0
    for ts in small:
        util += len(drop_set_at_severity(ts, 1.0, by_utilisation))
        prio += len(drop_set_at_severity(ts, 1.0, by_priority))
    assert prio > util, (
        f"priority ordering shed {prio}, utilisation shed {util}"
    )


@pytest.mark.parametrize("ordering", [by_utilisation, by_execution_time, by_priority])
def test_every_ordering_produces_a_feasible_set(tasksets, ordering):
    """Orderings trade efficiency, never correctness."""
    for ts in tasksets[:12]:
        for x in (0.5, 1.0):
            dropped = drop_set_at_severity(ts, x, ordering)
            assert dropped is not None
            assert is_feasible(ts, severity_budgets(ts, x), dropped)


# ---------------------------------------------------------------------------
# Ladders
# ---------------------------------------------------------------------------


def test_ladder_is_nested(tasksets):
    """S_1 subset of S_2 subset of ... -- required for incremental transitions."""
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ts in tasksets:
        ladder = drop_ladder(ts, severities)
        assert ladder is not None
        for a, b in zip(ladder, ladder[1:]):
            assert a <= b, f"ladder not nested: {a} then {b}"


def _set_dependent_ordering(taskset, candidates):
    """An ordering that ranks by the candidate set, not by a fixed per-task key."""
    utils = {i: taskset.C_lo[i] / taskset.T[i] for i in candidates}
    mean = sum(utils.values()) / len(utils)
    return min(candidates, key=lambda i: (abs(utils[i] - mean), i))


@pytest.mark.parametrize(
    "ordering", [by_utilisation, by_execution_time, by_priority, _set_dependent_ordering]
)
def test_levels_are_prefixes_of_one_shed_sequence(tasksets, ordering):
    """Why nesting is automatic, stated as the property that actually causes it.

    An ``Ordering`` is a pure function of (task set, remaining candidates), and
    every level starts from the same full candidate set, so the sequence of
    tasks it sheds is identical at every severity -- a higher severity simply
    stops later. Each level is therefore a prefix of one sequence, and prefixes
    nest.

    This is stronger than asserting nesting directly, and it is what would
    actually break first: an ordering that consulted the severity (or the
    partial drop set) would violate it, and ``drop_ladder``'s inheritance would
    then be doing real work rather than being redundant.
    """
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ts in tasksets[:15]:
        levels = [drop_set_at_severity(ts, x, ordering) for x in severities]
        # Independently computed levels already nest, without any inheritance.
        for a, b in zip(levels, levels[1:]):
            assert a <= b, f"levels not nested under {ordering.__name__}: {a}, {b}"
        # And drop_ladder agrees with them exactly, so its inheritance changes
        # nothing for any ordering matching the current signature.
        assert drop_ladder(ts, severities, ordering) == levels


def test_ladder_levels_are_each_feasible(tasksets):
    severities = [0.0, 0.3, 0.6, 1.0]
    for ts in tasksets[:15]:
        ladder = drop_ladder(ts, severities)
        for x, dropped in zip(severities, ladder):
            assert is_feasible(ts, severity_budgets(ts, x), dropped)


def test_ladder_rejects_descending_severities(tasksets):
    with pytest.raises(ValueError, match="ascending"):
        drop_ladder(tasksets[0], [0.5, 0.2])


def test_nesting_can_cost_more_than_independent_levels(tasksets):
    """Honest about the price of nesting, so it is a known trade not a surprise.

    Inheriting the previous level's set can shed a task a deeper level would
    not have chosen. The constraint is worth it -- transitions become
    incremental -- but it is not free.
    """
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    nested_total = independent_total = 0
    for ts in tasksets:
        ladder = drop_ladder(ts, severities)
        nested_total += sum(len(s) for s in ladder)
        independent_total += sum(
            len(drop_set_at_severity(ts, x)) for x in severities
        )
    assert nested_total >= independent_total


# ---------------------------------------------------------------------------
# Infeasibility
# ---------------------------------------------------------------------------


def test_infeasible_level_returns_none():
    """A HI task that cannot meet its deadline even alone is unrescuable."""
    ts = TaskSet(
        n=2,
        criticality=["HI", "LO"],
        T=[100, 50],
        D=[100, 50],
        C_lo=[40, 10],
        C_hi=[150, 10],  # C_hi exceeds its own deadline
        BCET=[40, 10],
    )
    assign_deadline_monotonic(ts)
    assert drop_set_at_severity(ts, 1.0) is None


def test_nontrivial_amc_population_is_always_feasible_at_full_severity():
    """Task sets the papers would keep are rescuable by shedding LO work."""
    kept = []
    for seed in range(4000):
        ts = generate_taskset(n=12, CP=0.5, U=0.7, CF=2.0, rng_seed=seed)
        assign_deadline_monotonic(ts)
        if (
            normal_mode_schedulable(ts)
            and busy_period_bound(ts) is not None
            and is_nontrivial_amc_taskset(ts, use_opa=False)
        ):
            kept.append(ts)
        if len(kept) == 20:
            break
    assert kept, "no qualifying task sets found"
    for ts in kept:
        assert drop_set_at_severity(ts, 1.0) is not None


# ---------------------------------------------------------------------------
# Carry-in: what a shed task still costs before it stops releasing
# ---------------------------------------------------------------------------


def test_carry_in_of_zero_matches_charging_a_shed_task_nothing(tasksets):
    """carry_in[k] = 0 is the old instantaneous-shedding assumption exactly."""
    for ts in tasksets[:8]:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        budgets = severity_budgets(ts, 1.0)
        for i in range(ts.n):
            optimistic = response_time(ts, i, budgets, set(lo))
            zeroed = response_time(ts, i, budgets, set(lo), dict.fromkeys(lo, 0.0))
            assert optimistic == zeroed


def test_infinite_carry_in_matches_never_shedding_at_all(tasksets):
    """carry_in[k] = inf charges k in full, as though it had never been shed."""
    for ts in tasksets[:8]:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        budgets = severity_budgets(ts, 1.0)
        for i in range(ts.n):
            unshed = response_time(ts, i, budgets, set())
            no_bound = response_time(ts, i, budgets, set(lo), dict.fromkeys(lo, math.inf))
            assert unshed == no_bound


def test_carry_in_is_monotone_in_the_shed_instant(tasksets):
    """Shedding earlier never raises the response time -- Mechanism 1's engine."""
    for ts in tasksets[:8]:
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        budgets = severity_budgets(ts, 1.0)
        for i in range(ts.n):
            prev = -1.0
            for s in (0, 5, 25, 100, 1000, math.inf):
                r = response_time(ts, i, budgets, set(lo), dict.fromkeys(lo, s))
                assert r >= prev
                prev = r


def test_carry_in_reproduces_amc_rtb_when_everything_sheds_at_r_lo(tasksets):
    """chi=1, S=all LO, s=R_i(LO) is exactly AMC-rtb's equation (2)."""
    for ts in tasksets[:8]:
        res = amc_rtb(ts)
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        budgets = severity_budgets(ts, 1.0)
        for i in range(ts.n):
            if ts.criticality[i] != "HI" or res.r_hi[i] > ts.D[i]:
                continue
            got = response_time(ts, i, budgets, set(lo), dict.fromkeys(lo, float(res.r_lo[i])))
            assert got == float(res.r_hi[i])


def test_shed_early_is_feasible_exactly_when_amc_rtb_passes(tasksets):
    """Shedding everything at R_i(LO) *is* AMC-rtb's equation (2).

    So the shed-early policy can certify a task set precisely when the classic
    two-level test can -- it never demands more than AMC-rtb, and never rescues
    a set AMC-rtb rejects. That is the property that makes it a drop-in
    replacement for progressive shedding rather than a stricter criterion.
    """
    for ts in tasksets:
        rtb_ok = all(
            r <= d
            for r, d, c in zip(amc_rtb(ts).r_hi, ts.D, ts.criticality)
            if c == "HI"
        )
        assert (drop_set_shed_early(ts, [0.5, 1.0, 1.0]) is not None) == rtb_ok


def test_shed_early_is_feasible_where_progressive_shedding_is_not(tasksets):
    """The result that makes shedding early the endorsed policy.

    Progressive shedding credits a task shed at a deep rung with that rung's
    R_i(chi), which is infinite for a third of HI tasks at chi=1 -- so the
    analysis charges it in full and the rung cannot be certified. Shedding
    everything at rung 1 always has a finite bound, R_i(LO).
    """
    trig, oper = [0.0, 0.5, 1.0], [0.5, 1.0, 1.0]
    early_ok = sum(drop_set_shed_early(ts, oper, trig[0]) is not None for ts in tasksets)
    progressive_ok = sum(
        drop_ladder(ts, oper, trigger_severities=trig, charge_carry_in=True) is not None
        for ts in tasksets
    )
    assert progressive_ok < early_ok, (
        "progressive shedding is expected to fail on part of the population; "
        f"got {progressive_ok} feasible against shed-early's {early_ok}"
    )


def test_shed_early_sheds_less_than_two_level_amc(tasksets):
    """Still a strict improvement on abandoning every LO task, once carry-in is paid."""
    fractions = []
    for ts in tasksets:
        early = drop_set_shed_early(ts, [0.5, 1.0, 1.0])
        if early is None:
            continue  # fails AMC-rtb; no policy can rescue it
        lo = [i for i in range(ts.n) if ts.criticality[i] == "LO"]
        fractions.append(len(early) / len(lo))
    assert fractions
    assert sum(fractions) / len(fractions) < 1.0
