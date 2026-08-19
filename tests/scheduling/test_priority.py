"""Tests for priority assignment.

Deadline monotonic is optimal for single-criticality FPPS with constrained
deadlines, but not for AMC-rtb; both reference papers use Audsley's Optimal
Priority Assignment instead.
"""

import numpy as np
import pytest

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import (
    amc_rtb,
    amc_rtb_single,
    fpps_schedulable,
    is_nontrivial_amc_taskset,
)
from amc_tasksim.scheduling.priority import assign_audsley_opa, assign_deadline_monotonic


def _appendix_a() -> TaskSet:
    return TaskSet(
        n=3,
        criticality=["LO", "HI", "HI"],
        T=[2, 10, 100],
        D=[2, 10, 18],
        C_lo=[1, 1, 4],
        C_hi=[1, 5, 4],
        BCET=[1, 1, 4],
        priority=[],
        metadata={},
    )


def test_deadline_monotonic_orders_by_deadline():
    ts = _appendix_a()
    assign_deadline_monotonic(ts)
    assert ts.priority == [0, 1, 2]


def test_amc_rtb_single_matches_the_full_test():
    """The single-task test must agree with the whole-task-set analysis."""
    for seed in range(25):
        ts = generate_taskset(n=12, CP=0.5, U=0.7, CF=2.0, rng_seed=500 + seed)
        assign_deadline_monotonic(ts)
        full = amc_rtb(ts)
        for i in range(ts.n):
            hp = {j for j in range(ts.n) if ts.priority[j] < ts.priority[i]}
            single = amc_rtb_single(ts, i, hp)
            expected = full.schedulable_lo[i] and full.schedulable_hi[i]
            assert single == expected, f"task {i} of seed {seed}: {single} != {expected}"


def test_opa_assignment_is_a_permutation_and_is_schedulable():
    """A successful OPA assignment must be a valid priority order that passes
    the very test it was built from."""
    found = 0
    for seed in range(40):
        ts = generate_taskset(n=15, CP=0.5, U=0.75, CF=2.0, rng_seed=700 + seed)
        if not assign_audsley_opa(ts):
            continue
        found += 1
        assert sorted(ts.priority) == list(range(ts.n)), "priorities must be a permutation"
        assert amc_rtb(ts).overall_schedulable
    assert found > 5, "expected several schedulable task sets in this sample"


def test_opa_finds_task_sets_deadline_monotonic_misses():
    """Audsley's algorithm is optimal for AMC-rtb; deadline monotonic is not.

    The gap is what makes the choice matter: at high utilisation, DM rejects
    task sets that are in fact schedulable, changing which task sets the
    experiment ever simulates.
    """
    dm_only = opa_only = 0
    for seed in range(120):
        ts = generate_taskset(n=20, CP=0.5, U=0.85, CF=2.0, rng_seed=900 + seed)
        assign_deadline_monotonic(ts)
        dm = amc_rtb(ts).overall_schedulable
        opa = assign_audsley_opa(ts)
        if dm and not opa:
            dm_only += 1
        if opa and not dm:
            opa_only += 1

    assert dm_only == 0, "OPA is optimal, so it cannot miss what DM finds"
    assert opa_only > 0, "OPA should find task sets DM rejects at this utilisation"


def test_opa_failure_leaves_a_usable_priority_order():
    """A task set that no ordering can schedule still has to be simulatable."""
    ts = TaskSet(
        n=2,
        criticality=["HI", "HI"],
        T=[10, 10],
        D=[10, 10],
        C_lo=[8, 8],
        C_hi=[9, 9],
        BCET=[8, 8],
        priority=[],
        metadata={},
    )
    assert not assign_audsley_opa(ts)
    assert sorted(ts.priority) == [0, 1]


# ---------------------------------------------------------------------------
# The paper's task-set filter
# ---------------------------------------------------------------------------


def test_fpps_schedulable_ignores_criticality():
    """FPPS at max(C_lo, C_hi) must reject the Appendix A example: tau3 needs
    4 + 5 (one job of tau2 at C_hi) + interference from tau1 within 18."""
    ts = _appendix_a()
    assert not fpps_schedulable(ts)

    # Halving the execution times makes it comfortably schedulable.
    easy = _appendix_a()
    easy.C_lo = [1, 1, 2]
    easy.C_hi = [1, 2, 2]
    assert fpps_schedulable(easy)


def test_nontrivial_filter_requires_both_conditions():
    """Section V-C: unschedulable under FPPS ignoring criticality, but
    schedulable under AMC-rtb."""
    # Schedulable under plain FPPS -> does not need AMC -> excluded.
    easy = _appendix_a()
    easy.C_lo = [1, 1, 2]
    easy.C_hi = [1, 2, 2]
    assert fpps_schedulable(easy)
    assert not is_nontrivial_amc_taskset(easy)

    # Appendix A: fails FPPS, and also fails AMC-rtb -> excluded.
    hard = _appendix_a()
    assert not fpps_schedulable(hard)
    assert not is_nontrivial_amc_taskset(hard)

    # Relaxing tau3's deadline to 20 makes AMC-rtb pass while FPPS still fails.
    both = _appendix_a()
    both.D = [2, 10, 20]
    assert not fpps_schedulable(both)
    assert is_nontrivial_amc_taskset(both)


def test_nontrivial_population_is_where_the_papers_work():
    """The filtered population should be substantial around U = 0.8, which is
    the operating point of the AMC-RH evaluation."""
    kept = 0
    total = 60
    for seed in range(total):
        ts = generate_taskset(n=20, CP=0.5, U=0.8, CF=2.0, rng_seed=1300 + seed)
        if is_nontrivial_amc_taskset(ts):
            kept += 1
    assert kept > total * 0.4, f"only {kept}/{total} task sets qualified at U=0.8"
