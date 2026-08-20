"""Tests for the k-level severity-ladder engine.

The central claim under test is exact reproduction at k=2, split into two
separately-checkable pieces per the finding that motivated the ladder's
trigger/operating severity split (multilevel.py, SeverityLadder docstring):

1. With ``drop_policy="full"`` (drop every LO-criticality task once
   degraded, matching the classic scheme exactly), the k-level engine must
   be BIT-IDENTICAL to the two-level engine's AMC_RA. This isolates whether
   the k-level engine's own mechanics (event loop, entry, exit, budget
   escalation, metrics) are correct, independent of the admissible-drop-set
   optimisation.
2. With ``drop_policy="admissible"`` (the actual scheme), JNE must be no
   greater than the "full" policy's -- never bit-identical, since retaining
   admissible LO work is the entire point, but never worse either
   (Corollary 1's practical consequence).
"""

from __future__ import annotations

import math

import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import (
    busy_period_bound,
    is_nontrivial_amc_taskset,
    normal_mode_schedulable,
)
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import AMC_RA, simulate
from amc_tasksim.simulation.multilevel import (
    MultiLevelJob,
    SeverityLadder,
    _natural_level,
    build_ladder,
    simulate_multilevel,
)


def population(count: int, **kw) -> list[TaskSet]:
    params = dict(n=12, CP=0.5, U=0.7, CF=2.0, period_mode="semi_harmonic")
    params.update(kw)
    out = []
    for seed in range(4000):
        ts = generate_taskset(rng_seed=seed, **params)
        assign_deadline_monotonic(ts)
        if (
            normal_mode_schedulable(ts)
            and busy_period_bound(ts) is not None
            and is_nontrivial_amc_taskset(ts, use_opa=False)
        ):
            out.append(ts)
        if len(out) == count:
            break
    assert len(out) == count
    return out


def tight_population(count: int, alpha: float = 0.3, **kw) -> list[TaskSet]:
    """Task sets whose deadlines sit close to their normal-mode response times.

    ``D_i`` is pulled down to ``alpha`` of the way from ``R_i(LO)`` to ``T_i``,
    which preserves normal-mode schedulability exactly while removing most of
    the slack. Terminations do not occur at all in the implicit-deadline
    population -- a retained LO task there has a median 48-96% of its deadline
    still to run when it first falls behind -- so any test about termination
    has to construct a population where the behaviour actually happens.
    """
    from amc_tasksim.scheduling.amc_rtb import amc_rtb
    from amc_tasksim.scheduling.priority import assign_audsley_opa
    params = dict(n=12, CP=0.5, U=0.7, CF=2.0)
    params.update(kw)
    out = []
    for seed in range(4000):
        ts = generate_taskset(rng_seed=seed, **params)
        assign_deadline_monotonic(ts)
        r = amc_rtb(ts).r_lo
        if any(r[i] > ts.T[i] for i in range(ts.n)):
            continue
        ts.D = [int(r[i] + alpha * (ts.T[i] - r[i])) for i in range(ts.n)]
        # Audsley's algorithm, as the papers use: deadline-monotonic priorities
        # on a tightened-deadline set leave the LO tasks too well protected for
        # any termination to occur, so the behaviour under test never appears.
        if not assign_audsley_opa(ts):
            continue
        if normal_mode_schedulable(ts) and is_nontrivial_amc_taskset(ts, use_opa=False):
            out.append(ts)
        if len(out) == count:
            break
    assert len(out) == count, f"only found {len(out)} of {count}"
    return out


@pytest.fixture(scope="module")
def tasksets() -> list[TaskSet]:
    return population(12)


FIELDS = ["nid", "jne", "lo_terminated", "hdm", "tid"]


# ---------------------------------------------------------------------------
# build_ladder
# ---------------------------------------------------------------------------


def test_operating_severity_of_deepest_level_is_always_one(tasksets):
    for ts in tasksets:
        for severities in [[0.0], [0.0, 0.3], [0.0, 0.2, 0.6]]:
            ladder = build_ladder(ts, severities)
            assert ladder is not None
            assert ladder.operating_severities[-1] == 1.0


def test_operating_severity_looks_ahead_to_the_next_trigger(tasksets):
    ts = tasksets[0]
    ladder = build_ladder(ts, [0.0, 0.3, 0.6])
    assert ladder is not None
    assert ladder.operating_severities[0] == 0.3  # level 1 operates at level 2's trigger
    assert ladder.operating_severities[1] == 0.6  # level 2 operates at level 3's trigger
    assert ladder.operating_severities[2] == 1.0  # deepest level: always full


def test_k_equals_two_has_trigger_zero_and_operating_one(tasksets):
    """The specific case that forced the trigger/operating split to exist."""
    ladder = build_ladder(tasksets[0], [0.0])
    assert ladder is not None
    assert ladder.severities == [0.0]
    assert ladder.operating_severities == [1.0]


def test_severities_must_start_at_zero(tasksets):
    with pytest.raises(ValueError, match="0.0"):
        build_ladder(tasksets[0], [0.1])


def test_severities_must_be_ascending(tasksets):
    with pytest.raises(ValueError, match="ascending"):
        build_ladder(tasksets[0], [0.0, 0.5, 0.3])


def test_full_drop_policy_drops_every_lo_task_at_every_level(tasksets):
    ts = tasksets[0]
    lo = {i for i in range(ts.n) if ts.criticality[i] == "LO"}
    ladder = build_ladder(ts, [0.0, 0.3], drop_policy="full")
    assert ladder is not None
    assert all(s == lo for s in ladder.drop_sets)


def test_admissible_drop_sets_are_no_larger_than_full(tasksets):
    for ts in tasksets:
        admissible = build_ladder(ts, [0.0, 0.4])
        full = build_ladder(ts, [0.0, 0.4], drop_policy="full")
        assert admissible is not None and full is not None
        for a, f in zip(admissible.drop_sets, full.drop_sets):
            assert a <= f


def test_unknown_drop_policy_rejected(tasksets):
    with pytest.raises(ValueError, match="drop_policy"):
        build_ladder(tasksets[0], [0.0], drop_policy="bogus")


# ---------------------------------------------------------------------------
# Exact reproduction at k=2, drop_policy="full"
# ---------------------------------------------------------------------------


DURATION = 300_000
FP = 5e-3
N_SEEDS = 8


def test_k2_full_drop_reproduces_amc_ra_exactly(tasksets):
    """The hard correctness bar: bit-identical, not merely similar."""
    mismatches = 0
    checked = 0
    for ts in tasksets:
        r_lo = _r_lo(ts)
        ladder = build_ladder(ts, [0.0], drop_policy="full")
        assert ladder is not None
        for seed in range(N_SEEDS):
            two_level = simulate(
                ts, duration=DURATION, seed=seed, fp=FP,
                mode_protocol=AMC_RA(r_lo), skip_quiet=False,
            )
            multi = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=FP)
            checked += 1
            for field_name in FIELDS:
                if getattr(two_level, field_name) != getattr(multi, field_name):
                    mismatches += 1
            if two_level.hi_releases_per_task != multi.hi_releases_per_task:
                mismatches += 1
            if two_level.lo_releases_per_task != multi.lo_releases_per_task:
                mismatches += 1
            if two_level.hi_trigger_events != multi.hi_trigger_events:
                mismatches += 1
    assert checked == len(tasksets) * N_SEEDS
    assert mismatches == 0


def _r_lo(taskset: TaskSet) -> list[float]:
    from amc_tasksim.scheduling.amc_rtb import amc_rtb

    return amc_rtb(taskset).r_lo


def test_k2_full_drop_reproduces_amc_ra_across_fp_values(tasksets):
    for fp in [0.0, 1e-3, 1e-2]:
        ts = tasksets[0]
        r_lo = _r_lo(ts)
        ladder = build_ladder(ts, [0.0], drop_policy="full")
        for seed in range(4):
            two_level = simulate(
                ts, duration=DURATION, seed=seed, fp=fp,
                mode_protocol=AMC_RA(r_lo), skip_quiet=False,
            )
            multi = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=fp)
            for field_name in FIELDS:
                assert getattr(two_level, field_name) == getattr(multi, field_name), (
                    f"fp={fp} seed={seed} field={field_name}: "
                    f"{getattr(two_level, field_name)} != {getattr(multi, field_name)}"
                )


# ---------------------------------------------------------------------------
# The admissible policy: never worse than full drop, genuinely different
# ---------------------------------------------------------------------------


def test_admissible_jne_never_exceeds_full_drop_jne(tasksets):
    for ts in tasksets:
        admissible = build_ladder(ts, [0.0, 0.4])
        full = build_ladder(ts, [0.0, 0.4], drop_policy="full")
        for seed in range(6):
            r_adm = simulate_multilevel(ts, admissible, duration=DURATION, seed=seed, fp=FP)
            r_full = simulate_multilevel(ts, full, duration=DURATION, seed=seed, fp=FP)
            assert r_adm.jne <= r_full.jne


def test_admissible_policy_is_sometimes_strictly_better():
    """Guards against the comparison above being vacuously true."""
    ts = population(1, U=0.7)[0]
    admissible = build_ladder(ts, [0.0, 0.3, 0.6])
    full = build_ladder(ts, [0.0, 0.3, 0.6], drop_policy="full")
    total_adm = total_full = 0
    for seed in range(20):
        total_adm += simulate_multilevel(ts, admissible, duration=DURATION, seed=seed, fp=FP).jne
        total_full += simulate_multilevel(ts, full, duration=DURATION, seed=seed, fp=FP).jne
    assert total_adm < total_full


def test_hdm_is_zero_under_the_admissible_policy(tasksets):
    """The safety property admissibility exists to guarantee."""
    for ts in tasksets:
        ladder = build_ladder(ts, [0.0, 0.3, 0.7])
        assert ladder is not None
        for seed in range(6):
            r = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=FP)
            assert r.hdm == 0


def test_inadmissible_ladder_can_produce_hi_deadline_misses():
    """The engine does not silently enforce admissibility -- proven directly.

    Constructs a ladder with an artificially tiny operating severity at a
    deep level (below what safety needs) by hand, bypassing build_ladder's
    admissible computation, and confirms HDM becomes nonzero -- so a future
    change that broke admissibility elsewhere would be caught by *this*
    engine actually exhibiting deadline misses, not by trusting it can't.
    """
    ts = TaskSet(
        n=3,
        criticality=["HI", "LO", "LO"],
        T=[100, 30, 30],
        D=[100, 30, 30],
        C_lo=[20, 8, 8],
        C_hi=[90, 8, 8],
        BCET=[20, 8, 8],
    )
    assign_deadline_monotonic(ts)
    bad_ladder = SeverityLadder(
        severities=[0.0],
        operating_severities=[0.0],  # deliberately NOT 1.0: HI task never gets more budget
        thresholds=[[0.0]],  # fires immediately, forcing degraded mode constantly
        drop_sets=[set()],  # and drops nothing, so LO interference is unbounded
    )
    misses = 0
    for seed in range(10):
        r = simulate_multilevel(ts, bad_ladder, duration=50_000, seed=seed, fp=1.0)
        misses += r.hdm
    assert misses > 0


# ---------------------------------------------------------------------------
# General properties
# ---------------------------------------------------------------------------


def test_level_ticks_sum_to_duration(tasksets):
    for ts in tasksets[:5]:
        ladder = build_ladder(ts, [0.0, 0.3])
        for seed in range(4):
            r = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=FP)
            assert sum(r.level_ticks) == DURATION


def test_tid_matches_level_ticks_above_zero(tasksets):
    ts = tasksets[0]
    ladder = build_ladder(ts, [0.0, 0.3])
    r = simulate_multilevel(ts, ladder, duration=DURATION, seed=1, fp=FP)
    assert r.tid == pytest.approx(sum(r.level_ticks[1:]) / DURATION)


def test_level_trans_at_least_covers_nid(tasksets):
    """Every L0-exit is a level transition; LevelTrans generalises NiD."""
    ts = tasksets[0]
    ladder = build_ladder(ts, [0.0, 0.2, 0.5])
    r = simulate_multilevel(ts, ladder, duration=DURATION, seed=1, fp=FP)
    assert r.level_trans >= r.nid


def test_no_lo_job_ever_completes_after_its_deadline():
    """The safety property that makes termination sound rather than lax.

    A LO-criticality job that cannot finish in time is terminated, never allowed
    to deliver a late result -- a late result may be worse than none. This is
    the invariant the JNC objective rests on, so it is checked directly against
    the trace rather than inferred from the metric counts.
    """
    for ts in population(4):
        ladder = build_ladder(ts, [0.0, 0.3])
        for seed in range(4):
            trace: list[tuple] = []
            simulate_multilevel(ts, ladder, duration=DURATION, seed=seed,
                                fp=0.3, trace=trace)
            last_release: dict[int, int] = {}
            for t, event, i in trace:
                if i < 0:
                    continue
                if event == "release":
                    last_release[i] = t
                elif event == "complete" and ts.criticality[i] == "LO":
                    r = last_release.get(i)
                    assert r is not None, "completion with no preceding release"
                    assert t <= r + ts.D[i], (
                        f"task {i} completed at {t}, after its deadline "
                        f"{r + ts.D[i]} -- a late LO result was delivered"
                    )


def test_wasted_cpu_counts_execution_thrown_away_by_termination():
    """WastedCPU is live under deadline termination, and was not before.

    The abandon-on-release policy never abandons a job mid-execution, so this
    metric used to be zero by construction -- a claim this module's docstring
    made, and which deadline termination falsifies.
    """
    seen_termination = False
    for ts in tight_population(6):
        ladder = build_ladder(ts, [0.0, 0.5, 1.0], drop_policy="shed_early")
        if ladder is None:
            continue
        for seed in range(3):
            r = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=0.2)
            assert (r.wasted_cpu > 0) == (r.lo_terminated > 0), (
                "wasted CPU and terminations must appear together"
            )
            if r.lo_terminated:
                seen_termination = True
                assert r.wasted_cpu >= r.lo_terminated, (
                    "a terminated job executed at least one tick"
                )
    assert seen_termination, "no terminations observed; test proves nothing"


def test_jnc_accounts_for_every_job_that_delivered_no_result():
    """JNC is measured by difference against a perfect scheduler, not by adding up.

    ``lo_expected`` depends only on the run duration and the tasks' periods, so
    it cannot drift with anything the scheme did. JNC must therefore equal the
    ways a job can fail -- abandoned on release, or terminated at its deadline --
    plus at most one in-flight job per task at the horizon.
    """
    for ts in tight_population(4):
        ladder = build_ladder(ts, [0.0, 0.5, 1.0], drop_policy="shed_early")
        if ladder is None:
            continue
        for seed in range(3):
            r = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=0.2)
            accounted = r.jne + r.lo_terminated
            assert accounted <= r.jnc, "JNC must cover every counted failure"
            assert r.jnc - accounted <= ts.n, (
                "unaccounted shortfall exceeds the horizon artefact bound"
            )
            assert 0.0 <= r.service_ratio <= 1.0
            assert abs(r.service_ratio * 100 + r.jnc_pct - 100) < 1e-9


def test_no_budget_overruns_under_the_admissible_policy(tasksets):
    for ts in tasksets:
        ladder = build_ladder(ts, [0.0, 0.4])
        for seed in range(4):
            r = simulate_multilevel(ts, ladder, duration=DURATION, seed=seed, fp=FP)
            assert r.budget_overruns == 0


def test_release_counts_are_rng_independent(tasksets):
    """Matches the periodic ground truth, same property as the two-level engine."""
    ts = tasksets[0]
    ladder = build_ladder(ts, [0.0, 0.3])
    r = simulate_multilevel(ts, ladder, duration=DURATION, seed=1, fp=FP)
    for i in range(ts.n):
        expected = -(-DURATION // ts.T[i])
        got = (
            r.hi_releases_per_task[i]
            if ts.criticality[i] == "HI"
            else r.lo_releases_per_task[i]
        )
        assert got == expected


def test_k_property_matches_severities_length(tasksets):
    ladder = build_ladder(tasksets[0], [0.0, 0.2, 0.5, 0.9])
    assert ladder.k == 5


def test_the_two_operating_points_differ_in_the_expected_direction():
    """Conservative sheds at least as much as termination, and never less.

    The conservative point certifies every retained LO task's deadline; the
    termination point only certifies HI deadlines and terminates the rest if
    they fall behind. The extra obligation can only force more shedding.
    """
    strictly_more = 0
    for ts in population(8):
        conservative = build_ladder(ts, [0.0, 0.5, 1.0], drop_policy="shed_early",
                                    require_lo_deadlines=True)
        termination = build_ladder(ts, [0.0, 0.5, 1.0], drop_policy="shed_early",
                                   require_lo_deadlines=False)
        assert termination is not None, "the safety-only point must always exist"
        if conservative is None:
            continue
        assert termination.drop_sets[0] <= conservative.drop_sets[0]
        if termination.drop_sets[0] < conservative.drop_sets[0]:
            strictly_more += 1
    assert strictly_more, "the two points never differed; the test proves nothing"
