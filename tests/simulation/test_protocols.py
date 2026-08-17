"""Tests for the three mode-change protocols.

The distinguishing behaviour of each scheme, from Section IV of the AMC-RH
paper (RTAS 2022):

- OriginalAMC (AMC+) enters degraded mode when a HI-criticality job has
  executed for C_i(LO) without completing, and leaves on an idle instant.
- AMC-RA enters when an active HI-criticality job reaches R_i(LO) from the
  start of its priority level-i busy period, and leaves on an idle instant.
- AMC-RH enters on the same condition, and leaves as soon as a HI-criticality
  job completes and no active HI-criticality job is past its own trigger.
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import amc_rtb
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.simulation.engine import AMC_RA, AMC_RH, OriginalAMC, simulate


def _three_task_set() -> TaskSet:
    """A task set that separates all three protocols.

    tau1: C_lo=1, T=4, D=4, LO       (highest priority)
    tau2: C_lo=2, C_hi=6, T=20, D=20, HI
    tau3: C_lo=3, T=50, D=50, LO     (lowest priority)

    R2(LO) = 3. With tau2 released at t=1 into the head of the run-queue, its
    trigger is at t=4, whereas it reaches C2(LO) at t=3.
    """
    ts = TaskSet(
        n=3,
        criticality=["LO", "HI", "LO"],
        T=[4, 20, 50],
        D=[4, 20, 50],
        C_lo=[1, 2, 3],
        C_hi=[1, 6, 3],
        BCET=[1, 2, 3],
        priority=[],
        metadata={},
    )
    assign_deadline_monotonic(ts)
    return ts


def _run(protocol, ts=None, offsets=(0, 1, 0), duration=20):
    ts = ts if ts is not None else _three_task_set()
    trace: list[tuple[int, str, int]] = []
    result = simulate(
        ts,
        duration=duration,
        seed=0,
        mode_protocol=protocol,
        fp=1.0,
        release_offsets=list(offsets),
        exec_time_mode="wcet",
        trace=trace,
    )
    entries = [t for t, ev, _ in trace if ev == "enter_degraded"]
    exits = [t for t, ev, _ in trace if ev == "exit_degraded"]
    return result, entries, exits


def test_r_lo_for_the_shared_scenario():
    rt = amc_rtb(_three_task_set())
    assert rt.overall_schedulable
    assert rt.r_lo[1] == 3, "R2(LO) = 2 + one job of tau1 = 3"


def test_original_amc_enters_at_c_lo():
    """tau2 starts running at t=1 and reaches C2(LO)=2 at t=3."""
    _, entries, exits = _run(OriginalAMC())
    assert entries == [3]
    assert exits == [10], "leaves on the idle instant after tau3 completes"


def test_response_time_protocols_enter_later_than_original():
    """AMC-RA and AMC-RH procrastinate until R2(LO) from the busy period start.

    tau2 is released at t=1 at the head of the run-queue, so s[2] = 1 and the
    trigger is at 1 + R2(LO) = 4 — one tick later than the original scheme.
    """
    rt = amc_rtb(_three_task_set())
    _, entries_orig, _ = _run(OriginalAMC())
    _, entries_ra, _ = _run(AMC_RA(rt.r_lo))
    _, entries_rh, _ = _run(AMC_RH(rt.r_lo))

    assert entries_ra == [4]
    assert entries_rh == [4]
    assert entries_ra[0] > entries_orig[0], (
        "AMC-RH/RA cannot enter degraded mode before the original scheme does"
    )


def test_amc_rh_exits_before_the_idle_instant():
    """S3 versus S5: AMC-RH returns to normal mode the moment the HI-criticality
    job completes, while AMC-RA waits for the processor to go idle."""
    rt = amc_rtb(_three_task_set())
    res_ra, _, exits_ra = _run(AMC_RA(rt.r_lo))
    res_rh, _, exits_rh = _run(AMC_RH(rt.r_lo))

    assert exits_rh == [7], "tau2 completes at t=7 and no other HI job is past its trigger"
    assert exits_ra == [10], "tau3 is still active, so AMC-RA stays degraded until t=10"
    assert res_rh.jne < res_ra.jne, "the earlier exit saves a release of tau1"
    assert res_rh.tid < res_ra.tid


def test_protocol_ordering_over_an_ensemble():
    """Section V-E: AMC-RH and AMC-RA both improve on the original scheme, and
    AMC-RH improves on AMC-RA because it exits degraded mode no later."""
    totals = {k: {"deg": 0, "jl": 0, "nid": 0} for k in ("amc", "ra", "rh")}
    n_sets = 0
    seed = 8000
    while n_sets < 20 and seed < 8300:
        ts = generate_taskset(n=20, CP=0.5, U=0.8, CF=1.5, rng_seed=seed)
        assign_deadline_monotonic(ts)
        rt = amc_rtb(ts)
        seed += 1
        if not rt.overall_schedulable:
            continue
        n_sets += 1
        for key, protocol in (
            ("amc", OriginalAMC()),
            ("ra", AMC_RA(rt.r_lo)),
            ("rh", AMC_RH(rt.r_lo)),
        ):
            r = simulate(ts, duration=200_000, seed=17, mode_protocol=protocol, fp=1e-2)
            totals[key]["deg"] += r.degraded_ticks
            totals[key]["jl"] += r.jne + r.ldm
            totals[key]["nid"] += r.nid

    assert n_sets >= 10, "need a reasonable number of schedulable task sets"
    assert totals["rh"]["deg"] <= totals["ra"]["deg"] <= totals["amc"]["deg"]
    assert totals["rh"]["jl"] <= totals["ra"]["jl"] <= totals["amc"]["jl"]
    assert totals["rh"]["nid"] < totals["amc"]["nid"]


def test_r_lo_is_rounded_up_to_an_integer():
    """Float response times must never bring the trigger forward."""
    protocol = AMC_RH([1.0, 2.5, 10.2])
    assert protocol.r_lo == [1, 3, 11]
