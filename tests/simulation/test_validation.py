"""Validation tests for the AMC simulator.

These check the engine against the AMC-RH paper (RTAS 2022) rather than merely
checking that it runs:

1. The Appendix A / Figure 13 scenario, tick by tick.
2. FP = 0 never degrades.
3. TiD is a fraction of the simulation time.
4. HDM is zero for task sets that pass AMC-rtb.
5. LDM is counted when a LO-criticality job misses its deadline.
6. The metrics respond to the failure probability.
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import amc_rtb
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import (
    AMC_RA,
    AMC_RH,
    OriginalAMC,
    SimulationResult,
    simulate,
)


def _appendix_a() -> TaskSet:
    """The Appendix A example task set from the AMC-RH paper.

    tau1: C_lo=1, T=2, D=2, LO
    tau2: C_lo=1, C_hi=5, T=10, D=10, HI
    tau3: C_lo=C_hi=4, T=100, D=18, HI
    """
    ts = TaskSet(
        n=3,
        criticality=["LO", "HI", "HI"],
        T=[2, 10, 100],
        D=[2, 10, 18],
        C_lo=[1, 1, 4],
        C_hi=[1, 5, 4],
        BCET=[1, 1, 4],
        priority=[],
        metadata={"source": "AMC-RH Appendix A"},
    )
    assign_deadline_monotonic(ts)
    return ts


def _schedulable_ensemble(u=0.7, count=25, seed0=6000, **kw):
    """Task sets at utilisation `u` that pass AMC-rtb under DM priorities."""
    out = []
    s = seed0
    while len(out) < count and s < seed0 + 400:
        ts = generate_taskset(n=20, CP=0.5, U=u, CF=1.5, rng_seed=s, **kw)
        assign_deadline_monotonic(ts)
        if amc_rtb(ts).overall_schedulable:
            out.append(ts)
        s += 1
    return out


# ---------------------------------------------------------------------------
# Appendix A / Figure 13
# ---------------------------------------------------------------------------


def test_appendix_a_figure_13_trace():
    """Reproduce the Figure 13 schedule for the original AMC scheme.

    From Appendix A: the worst case for tau3 has tau2 released at t=6 and
    exhibiting HI-criticality behaviour. "At t = 8, tau2 has executed for
    C2(LO) = 1 without completing, and so degraded mode is entered, and hence
    further releases of the LO-criticality task tau1 are not permitted. Task
    tau2 executes for a further 4 time units, followed by tau3, which completes
    its final time unit of execution for a worst-case HI-criticality response
    time of 13."
    """
    trace: list[tuple[int, str, int]] = []
    result = simulate(
        _appendix_a(),
        duration=14,
        seed=0,
        mode_protocol=OriginalAMC(),
        fp=1.0,  # force tau2 to exhibit HI-criticality behaviour
        release_offsets=[0, 6, 0],
        exec_time_mode="wcet",
        trace=trace,
    )

    entries = [t for t, ev, _ in trace if ev == "enter_degraded"]
    exits = [t for t, ev, _ in trace if ev == "exit_degraded"]
    drops = [t for t, ev, i in trace if ev == "drop" and i == 0]
    tau2_done = [t for t, ev, i in trace if ev == "complete" and i == 1]
    tau3_done = [t for t, ev, i in trace if ev == "complete" and i == 2]

    assert entries == [8], f"degraded mode should be entered at t=8, got {entries}"
    assert tau2_done == [12], f"tau2 executes 4 more units to t=12, got {tau2_done}"
    assert tau3_done == [13], (
        f"tau3 should complete at t=13 (response time 13), got {tau3_done}"
    )
    assert exits == [13], f"idle instant at t=13, got {exits}"
    assert drops == [8, 10, 12], f"tau1 releases dropped in degraded mode, got {drops}"

    assert result.nid == 1
    assert result.jne == 3
    assert result.lo_terminated == 0
    assert result.hdm == 0


def test_appendix_a_busy_period_start_times():
    """Appendix B: a job at the head of the run-queue starts its own busy
    period; otherwise it inherits from the next higher priority active task.

    tau3 is released at t=0 into an empty queue, so s[3] = 0 and its trigger is
    at R3(LO) = 10. tau2 is released at t=6 behind the tau1 job released at the
    same instant, so it inherits s[2] = 6 and triggers at 6 + R2(LO) = 8.
    """
    rt = amc_rtb(_appendix_a())
    assert rt.r_lo[1] == 2, "R2(LO) should be 2 (paper, Appendix A)"
    assert rt.r_lo[2] == 10, "R3(LO) should be 10 (paper, Appendix A)"

    seen: dict[tuple[int, int], tuple[int, int]] = {}

    class Probe(AMC_RH):
        def entry_time(self, state):
            for job in state.active:
                seen.setdefault((job.task_id, job.release), (job.busy_start, self.expiry(job)))
            return super().entry_time(state)

    simulate(
        _appendix_a(),
        duration=14,
        seed=0,
        mode_protocol=Probe(rt.r_lo),
        fp=1.0,
        release_offsets=[0, 6, 0],
        exec_time_mode="wcet",
    )

    assert seen[(2, 0)] == (0, 10), "tau3 starts its own busy period at t=0"
    assert seen[(1, 6)] == (6, 8), "tau2 inherits the busy period started at t=6"


# ---------------------------------------------------------------------------
# FP = 0 must never degrade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_name", ["original_amc", "amc_ra", "amc_rh"])
def test_fp_zero_never_degrades(protocol_name):
    """With FP=0 no job can exceed C_i(LO), so degraded mode is unreachable."""
    ts = TaskSet(
        n=5,
        criticality=["HI", "HI", "LO", "LO", "LO"],
        T=[10, 20, 30, 40, 50],
        D=[10, 20, 30, 40, 50],
        C_lo=[3, 5, 4, 5, 6],
        C_hi=[6, 10, 4, 5, 6],
        BCET=[1, 1, 1, 1, 1],
        priority=[],
        metadata={},
    )
    assign_deadline_monotonic(ts)
    r_lo = amc_rtb(ts).r_lo
    protocol = {
        "original_amc": OriginalAMC(),
        "amc_ra": AMC_RA(r_lo),
        "amc_rh": AMC_RH(r_lo),
    }[protocol_name]

    result = simulate(ts, duration=200_000, seed=1, mode_protocol=protocol, fp=0.0)

    assert result.hi_trigger_events == 0
    assert result.nid == 0
    assert result.tid == 0.0
    assert result.jne == 0


def test_cf_one_never_degrades():
    """With C_i(HI) = C_i(LO) no job can overrun, even at FP=1."""
    ts = TaskSet(
        n=3,
        criticality=["HI", "HI", "LO"],
        T=[10, 20, 30],
        D=[10, 20, 30],
        C_lo=[3, 5, 8],
        C_hi=[3, 5, 8],
        BCET=[2, 4, 6],
        priority=[],
        metadata={},
    )
    assign_deadline_monotonic(ts)
    result = simulate(ts, duration=50_000, seed=42, fp=1.0)

    assert result.hi_trigger_events > 0, "HI behaviour should be selected at FP=1"
    assert result.nid == 0, "but no job can execute beyond C_i(LO), so no mode change"
    assert result.jne == 0


# ---------------------------------------------------------------------------
# Metric well-formedness
# ---------------------------------------------------------------------------


def test_tid_is_a_fraction_of_the_simulation():
    """TiD must lie in [0, 1] — degraded intervals are disjoint and bounded."""
    for ts in _schedulable_ensemble(u=0.8, count=12, seed0=8000):
        result = simulate(ts, duration=200_000, seed=3, fp=1e-2)
        assert 0.0 <= result.tid <= 1.0, f"tid={result.tid}"
        assert 0 <= result.degraded_ticks <= result.duration
        assert result.tid == pytest.approx(result.degraded_ticks / result.duration)


def test_release_counts_match_the_periodic_schedule():
    """JNE's denominator must count every LO-criticality release, dropped or not."""
    ts = _schedulable_ensemble(u=0.6, count=1, seed0=1234)[0]
    duration = 100_000
    result = simulate(ts, duration=duration, seed=5, fp=1e-2)

    for i in range(ts.n):
        expected = -(-duration // ts.T[i])  # releases at 0, T, 2T, ... < duration
        got = (
            result.hi_releases_per_task[i]
            if ts.criticality[i] == "HI"
            else result.lo_releases_per_task[i]
        )
        assert got == expected, f"task {i}: {got} releases, expected {expected}"

    assert result.jne <= result.total_lo_releases
    assert result.nid <= result.hi_trigger_events, (
        "every degraded-mode entry needs at least one job exhibiting HI behaviour"
    )


def test_no_job_exceeds_its_budget():
    """The RTOS enforces C_i(HI) for HI tasks and C_i(LO) for LO tasks."""
    for ts in _schedulable_ensemble(u=0.7, count=8, seed0=4200):
        result = simulate(ts, duration=100_000, seed=11, fp=1e-2)
        assert result.budget_overruns == 0


def test_reproducible_for_a_given_seed():
    ts = _schedulable_ensemble(u=0.7, count=1, seed0=777)[0]
    a = simulate(ts, duration=100_000, seed=99, fp=1e-2)
    b = simulate(ts, duration=100_000, seed=99, fp=1e-2)
    assert (a.nid, a.tid, a.jne, a.lo_terminated, a.hdm) == (b.nid, b.tid, b.jne, b.lo_terminated, b.hdm)


# ---------------------------------------------------------------------------
# Deadline misses
# ---------------------------------------------------------------------------


def test_hdm_zero_for_amc_rtb_schedulable_tasksets():
    """AMC-rtb is a sufficient test for all three protocols, so no HI-criticality
    job may miss its deadline (Section V-E: HDM was zero for all schemes)."""
    for ts in _schedulable_ensemble(u=0.8, count=10, seed0=8000):
        r_lo = amc_rtb(ts).r_lo
        for protocol in (OriginalAMC(), AMC_RA(r_lo), AMC_RH(r_lo)):
            result = simulate(ts, duration=200_000, seed=13, mode_protocol=protocol, fp=1e-2)
            assert result.hdm == 0, f"HDM={result.hdm} on an AMC-rtb-schedulable task set"


def test_ldm_counts_late_lo_jobs():
    """A LO-criticality task that cannot meet its deadlines must register LDM."""
    ts = TaskSet(
        n=2,
        criticality=["HI", "LO"],
        T=[10, 10],
        D=[10, 10],
        C_lo=[8, 5],
        C_hi=[8, 5],
        BCET=[8, 5],
        priority=[0, 1],
        metadata={},
    )
    result = simulate(ts, duration=1000, seed=1, fp=0.0)

    assert result.lo_terminated > 0, "the LO task has 1.3 total utilisation and must miss deadlines"
    assert result.hdm == 0, "the HI task is the highest priority and always fits"
    assert result.jne == 0, "degraded mode is never entered, so nothing is dropped"


# ---------------------------------------------------------------------------
# Response to the failure probability
# ---------------------------------------------------------------------------


def test_metrics_scale_with_failure_probability():
    """NiD, TiD and JNE must all fall as FP falls — this is the sweep's whole point."""
    ts = _schedulable_ensemble(u=0.7, count=1, seed0=6000)[0]
    prev = None
    for fp in [1e-1, 1e-2, 1e-3]:
        result = simulate(ts, duration=500_000, seed=21, fp=fp)
        if prev is not None:
            assert result.nid < prev.nid
            assert result.tid < prev.tid
            assert result.jne < prev.jne
        prev = result
    assert simulate(ts, duration=500_000, seed=21, fp=0.0).nid == 0


def test_nid_percentage_tracks_failure_probability():
    """Under the original AMC scheme every overrunning HI job triggers one entry,
    so NiD as a percentage of HI-criticality jobs is of the order of FP.

    The observed value sits below FP because an integer execution time drawn
    from U{C_i(LO)..C_i(HI)} can land exactly on C_i(LO), in which case the job
    signals completion at its budget and correctly does not trigger.
    """
    nid = 0
    hi_jobs = 0
    fp = 1e-3
    for ts in _schedulable_ensemble(u=0.7, count=10, seed0=6000):
        result = simulate(ts, duration=1_000_000, seed=31, fp=fp)
        nid += result.nid
        hi_jobs += result.total_hi_releases

    observed = nid / hi_jobs
    assert 0.3 * fp < observed < 1.1 * fp, (
        f"NiD per HI job = {observed:.2e}, expected the same order as FP = {fp:.0e}"
    )
