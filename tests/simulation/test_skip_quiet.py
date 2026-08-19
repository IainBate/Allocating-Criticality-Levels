"""Tests for geometric skipping: the trigger schedule and the fast-forward.

Two claims are under test, and they need different kinds of test.

``_TriggerSchedule`` claims to be *distributionally identical* to drawing an
independent Bernoulli(fp) at every HI-criticality release. That is a property of
the sampler alone, so it is tested directly against the Bernoulli process it
replaces, with sample sizes chosen so the sampling error is small compared with
the tolerance.

``skip_quiet`` claims to be *statistically equivalent* to exact simulation, not
identical to it: the two paths consume the random stream in a different order, so
a single seed legitimately diverges and comparing one run against another would
fail on correct code. The equivalence is therefore asserted on aggregates over
many seeds, and separately on the quantities that are RNG-independent and so must
match exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.amc_rtb import busy_period_bound, normal_mode_schedulable
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import (
    OriginalAMC,
    SchedulerState,
    SimulationResult,
    _count_skipped_releases,
    _resume_instant,
    _skip_warm_up,
    _TriggerSchedule,
    simulate,
)

DURATION = 1_000_000
FP = 5e-3

# Seeds for the paired exact/skipping comparison. The two paths are only equal in
# distribution, so the sample has to be large enough to resolve a real bias from
# sampling noise; 120 puts the smallest detectable bias at 8-17% depending on the
# metric (see SENSITIVITY below), against ~5s to compute both paths once.
N_STAT = 120

# Seeds for checks that do not need the statistics -- exact equalities and the
# deliberately-broken comparison, where the effect is either exact or enormous.
N_SEEDS = 24

# Two-sample z beyond which the paths are called different. The seeds are fixed,
# so this is a deterministic threshold rather than a flaky one: on correct code
# the observed z sits below 1.
Z_CRIT = 4.0

# Largest relative bias each metric can hide from the z-test at N_STAT, measured
# and given ~25% headroom. Asserting these keeps a future change that shrinks the
# sample (or picks a noisier task set) from quietly blinding the comparison.
SENSITIVITY = {
    "nid": 0.10,
    "jne": 0.22,
    "degraded_ticks": 0.14,
    "hi_trigger_events": 0.10,
}


def quiet_taskset() -> TaskSet:
    """A task set that is R1-schedulable, so skip_quiet applies to it.

    Periods are short relative to the horizon so that many mode changes occur,
    and the LO-criticality tasks are frequent enough to be abandoned during
    degraded mode, which keeps JNE off the floor.
    """
    ts = TaskSet(
        n=4,
        criticality=["HI", "HI", "LO", "LO"],
        T=[200, 500, 300, 700],
        D=[200, 500, 300, 700],
        C_lo=[40, 80, 45, 105],
        C_hi=[90, 200, 45, 105],
        BCET=[32, 64, 36, 84],
    )
    assign_deadline_monotonic(ts)
    return ts


def unschedulable_taskset() -> TaskSet:
    """A task set that misses deadlines even when every job complies with C(LO)."""
    ts = TaskSet(
        n=2,
        criticality=["HI", "LO"],
        T=[100, 120],
        D=[100, 120],
        C_lo=[80, 80],
        C_hi=[95, 80],
        BCET=[80, 80],
    )
    assign_deadline_monotonic(ts)
    return ts


def run(ts: TaskSet, *, skip: bool, seeds=range(N_SEEDS), duration=DURATION, fp=FP):
    return [
        simulate(
            ts,
            duration=duration,
            seed=s,
            fp=fp,
            mode_protocol=OriginalAMC(),
            skip_quiet=skip,
        )
        for s in seeds
    ]


@pytest.fixture(scope="module")
def paired_runs():
    """The exact and skipping paths over the same seeds, computed once.

    Module-scoped because N_STAT seeds of exact simulation is the most expensive
    thing here by far, and every equivalence check below wants the same sample.
    """
    ts = quiet_taskset()
    seeds = range(N_STAT)
    return run(ts, skip=False, seeds=seeds), run(ts, skip=True, seeds=seeds)


def two_sample_z(exact, skipped, metric: str) -> tuple[float, float]:
    """Welch z for a metric, and the smallest relative bias it could detect.

    Returns ``(z, detectable)`` where ``detectable`` is the bias, as a fraction of
    the exact mean, that would put the z-score at ``Z_CRIT``.
    """
    a = np.array([getattr(r, metric) for r in exact], dtype=float)
    b = np.array([getattr(r, metric) for r in skipped], dtype=float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se, Z_CRIT * se / a.mean()


# ---------------------------------------------------------------------------
# _TriggerSchedule: the geometric sampler
# ---------------------------------------------------------------------------


class _Periodic:
    """Minimal stand-in exposing only the fields _TriggerSchedule reads."""

    def __init__(self, n_tasks: int = 1, period: int = 10):
        self.T = [period] * n_tasks
        self.criticality = ["HI"] * n_tasks
        self.n = n_tasks


def _observed_rate(fp: float, n_releases: int, seed: int) -> float:
    """Fraction of releases that trigger under the geometric schedule."""
    rng = np.random.default_rng(seed)
    sched = _TriggerSchedule(_Periodic(), fp, [0], rng)
    hits = 0
    for idx in range(n_releases):
        if sched.triggers(0, idx):
            hits += 1
            sched.consume(0, rng)
    return hits / n_releases


@pytest.mark.parametrize("fp", [0.5, 0.1, 0.01])
def test_trigger_rate_matches_bernoulli(fp):
    """The long-run trigger rate is fp, as a per-release Bernoulli(fp) would be."""
    # ~5000 expected hits, i.e. a sampling error near 1.4%.
    n_releases = int(5000 / fp)
    rate = _observed_rate(fp, n_releases, seed=0)
    assert rate == pytest.approx(fp, rel=0.1)


def test_first_trigger_index_is_not_offset():
    """The very first release triggers with probability fp.

    This is what pins down the ``- 1`` in the schedule's initialisation. Without
    it every task's first trigger would be pushed one release later; the long-run
    rate would still converge to fp, so only the first index reveals the error.
    """
    fp = 0.25
    rng = np.random.default_rng(11)
    first_is_zero = sum(
        _TriggerSchedule(_Periodic(), fp, [0], rng).next_index[0] == 0
        for _ in range(20_000)
    )
    assert first_is_zero / 20_000 == pytest.approx(fp, rel=0.1)


def test_gaps_between_triggers_are_geometric():
    """Consecutive trigger gaps have the mean and variance of Geometric(fp)."""
    fp = 0.05
    rng = np.random.default_rng(7)
    sched = _TriggerSchedule(_Periodic(), fp, [0], rng)
    gaps, last = [], None
    for idx in range(200_000):
        if sched.triggers(0, idx):
            if last is not None:
                gaps.append(idx - last)
            last = idx
            sched.consume(0, rng)

    gaps = np.array(gaps)
    assert gaps.min() >= 1, "two triggers cannot land on the same release"
    assert gaps.mean() == pytest.approx(1 / fp, rel=0.05)
    assert gaps.var() == pytest.approx((1 - fp) / fp**2, rel=0.15)


def test_fp_zero_disables_the_schedule():
    sched = _TriggerSchedule(_Periodic(), 0.0, [0], np.random.default_rng(0))
    assert not sched.enabled()
    assert sched.next_trigger_time(10_000) is None
    assert not any(sched.triggers(0, idx) for idx in range(1000))


def test_fp_one_triggers_every_release():
    rng = np.random.default_rng(0)
    sched = _TriggerSchedule(_Periodic(), 1.0, [0], rng)
    for idx in range(500):
        assert sched.triggers(0, idx)
        sched.consume(0, rng)


def test_lo_criticality_tasks_are_not_tracked():
    ts = _Periodic(n_tasks=2)
    ts.criticality = ["HI", "LO"]
    sched = _TriggerSchedule(ts, 0.5, [0, 0], np.random.default_rng(0))
    assert sched.hi_tasks == [0]
    assert not sched.triggers(1, 0)


def test_tasks_are_tracked_independently():
    """Consuming one task's trigger leaves the other task's schedule untouched."""
    rng = np.random.default_rng(3)
    sched = _TriggerSchedule(_Periodic(n_tasks=2), 0.2, [0, 0], rng)
    before = sched.next_index[1]
    sched.consume(0, rng)
    assert sched.next_index[1] == before


def test_next_trigger_time_uses_each_tasks_own_period():
    """The earliest trigger over tasks accounts for offsets and periods."""
    ts = _Periodic(n_tasks=2)
    ts.T = [10, 3]
    sched = _TriggerSchedule(ts, 0.5, [0, 1], np.random.default_rng(0))
    sched.next_index[0] = 4  # task 0 triggers at 0 + 4*10 = 40
    sched.next_index[1] = 2  # task 1 triggers at 1 + 2*3  =  7
    assert sched.next_trigger_time(100) == 7
    assert sched.next_trigger_time(5) is None  # both beyond the horizon


# ---------------------------------------------------------------------------
# _skip_warm_up: when fast-forwarding is allowed, and how far back it reaches
# ---------------------------------------------------------------------------


def test_warm_up_is_none_when_not_requested():
    assert _skip_warm_up(quiet_taskset(), False) is None


def test_warm_up_covers_deadline_and_busy_period():
    """The window has to span max(D) plus a busy period, not just one of them."""
    ts = quiet_taskset()
    warm_up = _skip_warm_up(ts, True)
    assert warm_up == max(ts.D) + busy_period_bound(ts)
    assert warm_up > max(ts.D)


def test_warm_up_refuses_a_task_set_that_fails_r1():
    """A task set that misses deadlines in normal mode is simulated exactly."""
    ts = unschedulable_taskset()
    assert not normal_mode_schedulable(ts)
    with pytest.warns(UserWarning, match="misses deadlines"):
        assert _skip_warm_up(ts, True) is None


def test_simulate_falls_back_when_r1_fails():
    """skip_quiet is ignored, with a warning, rather than silently wrong."""
    ts = unschedulable_taskset()
    with pytest.warns(UserWarning, match="misses deadlines"):
        skipped = simulate(ts, duration=50_000, seed=1, fp=FP, skip_quiet=True)
    exact = simulate(ts, duration=50_000, seed=1, fp=FP, skip_quiet=False)
    # Having fallen back, it is the exact simulation, so it matches run for run.
    assert skipped.nid == exact.nid
    assert skipped.jne == exact.jne
    assert skipped.ldm == exact.ldm
    assert skipped.degraded_ticks == exact.degraded_ticks


# ---------------------------------------------------------------------------
# _count_skipped_releases: the closed-form release accounting
# ---------------------------------------------------------------------------


def _counting_taskset() -> TaskSet:
    ts = TaskSet(
        n=2,
        criticality=["HI", "LO"],
        T=[10, 3],
        D=[10, 3],
        C_lo=[1, 1],
        C_hi=[2, 1],
        BCET=[1, 1],
    )
    assign_deadline_monotonic(ts)
    return ts


def test_count_skipped_releases_counts_each_task():
    ts = _counting_taskset()
    result = SimulationResult(duration=1000)
    result.hi_releases_per_task = [0, 0]
    result.lo_releases_per_task = [0, 0]
    heap = [(0, 0), (1, 1)]

    _count_skipped_releases(25, ts, heap, result)

    # Task 0 (HI, T=10) releases at 0, 10, 20 before 25.
    assert result.hi_releases_per_task == [3, 0]
    # Task 1 (LO, T=3) releases at 1, 4, ..., 22 before 25 -- eight of them.
    assert result.lo_releases_per_task == [0, 8]


def test_count_skipped_releases_leaves_the_first_release_at_or_after_resume():
    """The post-condition that stops a release being skipped without being run.

    Advancing a task past ``resume`` by one period too many keeps the running
    total right -- the release is still counted -- while quietly ensuring it is
    never simulated, so only this bracketing catches it.
    """
    ts = _counting_taskset()
    for resume in [1, 7, 25, 26, 100]:
        result = SimulationResult(duration=1000)
        result.hi_releases_per_task = [0, 0]
        result.lo_releases_per_task = [0, 0]
        heap = [(0, 0), (1, 1)]

        _count_skipped_releases(resume, ts, heap, result)

        for t_next, i in heap:
            assert t_next >= resume, f"task {i} left a release before resume"
            assert t_next < resume + ts.T[i], (
                f"task {i} advanced past the first release at or after resume: "
                f"t_next={t_next} resume={resume} T={ts.T[i]}"
            )


def test_count_skipped_releases_is_a_noop_when_nothing_is_due():
    ts = _counting_taskset()
    result = SimulationResult(duration=1000)
    result.hi_releases_per_task = [0, 0]
    result.lo_releases_per_task = [0, 0]
    heap = [(40, 0), (41, 1)]

    _count_skipped_releases(40, ts, heap, result)

    assert result.hi_releases_per_task == [0, 0]
    assert result.lo_releases_per_task == [0, 0]
    assert sorted(heap) == [(40, 0), (41, 1)]


def test_count_skipped_releases_restores_the_heap_invariant():
    ts = _counting_taskset()
    result = SimulationResult(duration=1000)
    result.hi_releases_per_task = [0, 0]
    result.lo_releases_per_task = [0, 0]
    # Task 1 is at the head now, but after the jump task 0 is earliest.
    heap = [(1, 1), (8, 0)]

    _count_skipped_releases(20, ts, heap, result)

    assert heap[0] == min(heap), "heap invariant broken: head is not the minimum"


# ---------------------------------------------------------------------------
# _resume_instant: the eligibility decision
# ---------------------------------------------------------------------------


def _schedule_with_trigger_at(t: int) -> _TriggerSchedule:
    ts = _Periodic(period=1)
    sched = _TriggerSchedule(ts, 0.5, [0], np.random.default_rng(0))
    sched.next_index[0] = t  # period 1 and offset 0, so index == time
    return sched


def test_resume_instant_refuses_a_non_empty_run_queue():
    state = SchedulerState(time=0, mode="normal")
    state.active = ["a placeholder job"]
    assert _resume_instant(0, state, _schedule_with_trigger_at(100_000), 200_000, 10) is None


def test_resume_instant_refuses_degraded_mode():
    state = SchedulerState(time=0, mode="degraded")
    assert _resume_instant(0, state, _schedule_with_trigger_at(100_000), 200_000, 10) is None


def test_resume_instant_stops_one_warm_up_before_the_trigger():
    state = SchedulerState(time=0, mode="normal")
    resume = _resume_instant(0, state, _schedule_with_trigger_at(100_000), 200_000, 10)
    assert resume == 100_000 - 10


def test_resume_instant_declines_an_interval_shorter_than_the_warm_up():
    """Skipping has to save more than the warm-up it costs."""
    state = SchedulerState(time=0, mode="normal")
    # Trigger at 25 with warm_up 10: resume would be 15, only 15 ahead of now,
    # which is not more than the 10-tick warm-up by enough to be worth it.
    assert _resume_instant(0, state, _schedule_with_trigger_at(25), 200_000, 10) == 15
    assert _resume_instant(0, state, _schedule_with_trigger_at(25), 200_000, 20) is None


def test_resume_instant_runs_to_the_horizon_when_no_trigger_remains():
    state = SchedulerState(time=0, mode="normal")
    sched = _TriggerSchedule(_Periodic(), 0.0, [0], np.random.default_rng(0))
    assert not sched.enabled()
    assert _resume_instant(0, state, sched, 200_000, 10) == 200_000 - 10


# ---------------------------------------------------------------------------
# skip_quiet: equivalence to exact simulation
# ---------------------------------------------------------------------------


def test_release_counts_are_exact(paired_runs):
    """Release counts depend only on periods and the horizon, never on the RNG.

    So unlike the other metrics these must agree seed for seed, which makes this
    the sharpest check on the closed-form release accounting across a skip.
    """
    for exact, skipped in zip(*paired_runs):
        assert skipped.hi_releases_per_task == exact.hi_releases_per_task
        assert skipped.lo_releases_per_task == exact.lo_releases_per_task


def test_release_counts_match_the_periodic_ground_truth():
    """Cross-check the counts against the periods directly, not just each other."""
    ts = quiet_taskset()
    result = simulate(
        ts, duration=DURATION, seed=0, fp=FP, mode_protocol=OriginalAMC(), skip_quiet=True
    )
    for i in range(ts.n):
        expected = -(-DURATION // ts.T[i])  # releases at 0, T, 2T, ... below duration
        got = (
            result.hi_releases_per_task[i]
            if ts.criticality[i] == "HI"
            else result.lo_releases_per_task[i]
        )
        assert got == expected, f"task {i}"


def test_hi_criticality_deadlines_are_never_missed(paired_runs):
    """The safety property, under both paths."""
    exact, skipped = paired_runs
    assert all(r.hdm == 0 for r in exact)
    assert all(r.hdm == 0 for r in skipped)


def test_no_budget_overruns(paired_runs):
    exact, skipped = paired_runs
    assert all(r.budget_overruns == 0 for r in exact)
    assert all(r.budget_overruns == 0 for r in skipped)


def test_lo_criticality_deadline_misses_agree(paired_runs):
    """LDM must not appear out of nowhere on the skipping path.

    A fast-forward that resumes with a mis-reconstructed run-queue shows up here
    first: jobs that the exact simulation completes on time start missing their
    deadlines instead.
    """
    exact, skipped = paired_runs
    assert sum(r.ldm for r in exact) == 0, "sanity: this task set misses no LO deadlines"
    assert sum(r.ldm for r in skipped) == 0


@pytest.mark.parametrize("metric", sorted(SENSITIVITY))
def test_aggregate_metrics_match_exact_simulation(paired_runs, metric):
    """Aggregates over many seeds agree, though individual seeds need not.

    Compared with a two-sample z rather than a fixed percentage band, so the
    tolerance follows the metric's own variance instead of being guessed. The
    companion assertion on ``detectable`` is what stops the comparison silently
    losing its teeth: a band wide enough to never fail is also wide enough to
    miss a real bias, and only the second assertion can tell the two apart.
    """
    exact, skipped = paired_runs
    assert sum(getattr(r, metric) for r in exact) > 0, f"sanity: {metric} is all zero"

    z, detectable = two_sample_z(exact, skipped, metric)

    assert detectable <= SENSITIVITY[metric], (
        f"{metric}: the comparison has gone blind -- it can now only detect a "
        f"bias of {detectable:.1%}, worse than the {SENSITIVITY[metric]:.0%} "
        f"budget. Raise N_STAT or pick a less noisy task set."
    )
    assert abs(z) < Z_CRIT, (
        f"{metric}: exact and skipping paths differ, z={z:.2f} "
        f"(detectable bias at this sample size: {detectable:.1%})"
    )


def test_idle_instant_precondition_is_load_bearing(monkeypatch):
    """Dropping the empty-queue precondition must break the run, loudly.

    This is the regression that motivates the precondition: fast-forwarding while
    a job is still in the run-queue discards a job that may have been destined to
    trigger a mode change, silently understating NiD. Reproducing it here pins two
    things at once -- that the precondition does real work, and that the tolerance
    the aggregate tests use is tight enough to catch its removal rather than
    absorbing it as sampling noise.
    """
    from amc_tasksim.simulation import engine

    ts = quiet_taskset()
    exact = sum(r.nid for r in run(ts, skip=False))
    assert exact > 0


    def resume_ignoring_idleness(now, state, schedule, duration, warm_up):
        if state.mode != "normal":  # note: the `state.active` check is gone
            return None
        trigger_at = schedule.next_trigger_time(duration)
        target = duration if trigger_at is None else trigger_at
        resume = target - warm_up
        return resume if resume - now > warm_up else None

    monkeypatch.setattr(engine, "_resume_instant", resume_ignoring_idleness)
    broken = sum(r.nid for r in run(ts, skip=True))

    # Mode changes are largely swallowed: an order-of-magnitude collapse, far
    # beyond the ~10% bias the z-test above resolves.
    assert broken / exact < 0.5, (
        f"removing the idle-instant precondition should collapse NiD, "
        f"but got exact={exact} broken={broken}"
    )


def test_skipping_is_faster_than_exact_simulation():
    """The whole point of the optimisation, asserted loosely enough not to flake."""
    import time

    ts = quiet_taskset()
    seeds = range(4)

    t0 = time.perf_counter()
    run(ts, skip=False, seeds=seeds)
    exact_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run(ts, skip=True, seeds=seeds)
    skip_s = time.perf_counter() - t0

    assert skip_s < exact_s
