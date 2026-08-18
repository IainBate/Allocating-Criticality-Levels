"""Experiment sweep orchestration for AMC task-set generation and simulation.

Runs the utilisation x failure-probability sweep and stores results as a tidy
DataFrame, one row per (U, N, protocol, replicate).

Three things follow the papers rather than convenience:

- Task sets are filtered to the population the papers evaluate: unschedulable
  under single-criticality FPPS, but schedulable under AMC-rtb with Audsley's
  optimal priority assignment. Generation continues until the requested number
  of *qualifying* task sets is reached.
- Every protocol runs on the same task sets with the same seeds, so the
  comparison is like-for-like and Table I of the AMC-RH paper can be computed
  directly from one sweep.
- The simulation duration is expressed in jobs of the longest-period task, as
  the paper does ("sufficient for 10^6 jobs of the task with the longest
  period"), so it scales with the periods actually drawn.

One thing does not follow the paper, and is documented where it matters: the
paper runs every configuration for the same fixed duration regardless of the
failure probability being studied. Sweeping N (this toolkit's own extension,
not something the papers do) makes that wasteful -- a cell at FP=1e-2 sees
enough HI-behaviour jobs to say something meaningful almost immediately, while
a cell at FP=1e-5 needs a very long run to see any at all. `Scale.target_hi_events`
lets duration track the failure probability instead of a single fixed job
count, clamped to `[duration_jobs_min, duration_jobs]` jobs of the longest
period. `duration_jobs` still acts as a ceiling: no cell ever runs longer than
the paper's own choice.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from amc_tasksim.generation.taskset import TaskSet, generate_taskset
from amc_tasksim.scheduling.amc_rtb import amc_rtb, fpps_schedulable
from amc_tasksim.scheduling.priority import assign_audsley_opa
from amc_tasksim.simulation.engine import AMC_RA, AMC_RH, OriginalAMC, simulate

PROTOCOLS = ("original_amc", "amc_ra", "amc_rh")


@dataclass(frozen=True)
class Scale:
    """A sizing preset for the sweep.

    Attributes:
        n_replicates: Qualifying task sets per utilisation level.
        duration_jobs: Simulation length, in jobs of the longest-period task.
            Acts as an upper bound when `target_hi_events` is set.
        U_values: Utilisation levels.
        N_values: Failure-probability levels (FP = 1/N).
        target_hi_events: If set, duration for a (task set, N) cell is scaled
            to the length expected to produce this many HI-behaviour jobs,
            rather than always running the full `duration_jobs`.
        duration_jobs_min: Floor on duration when `target_hi_events` is set, so
            high-FP cells don't run for a handful of periods.
    """

    n_replicates: int
    duration_jobs: int
    U_values: tuple[float, ...]
    N_values: tuple[int, ...]
    target_hi_events: Optional[int] = None
    duration_jobs_min: int = 0


SCALES = {
    # Small enough to iterate on in minutes, large enough that the metrics are
    # within reach of the paper's. Utilisations are restricted to the band
    # where the non-trivial population actually exists. Duration is flat here
    # -- debug runs aren't meant to have real statistical power, just to prove
    # the pipeline runs end to end.
    "debug": Scale(
        n_replicates=20,
        duration_jobs=200,
        U_values=(0.6, 0.7, 0.8, 0.9),
        N_values=(100, 1000, 10000),
    ),
    # The task-set configuration of RTAS 2022 Section V-D (500 replicates,
    # duration "sufficient for 10^6 jobs of the longest-period task"), with
    # duration scaled per N to ~1000 expected HI-behaviour jobs instead of a
    # flat 10^6-job run for every cell. At the paper's own default FP = 1e-4,
    # that is still a comfortable ~1000-observation sample; it cuts total
    # compute by roughly two orders of magnitude across the N sweep, mostly at
    # the high-FP end where the fixed duration bought nothing. Replicates are
    # reduced from 500 to 200 for the same reason: a documented compute
    # trade-off, not a silent shortcut. Use `duration_jobs=10**6,
    # target_hi_events=None, n_replicates=500` to run the literal paper
    # configuration if compute allows.
    "paper": Scale(
        n_replicates=200,
        duration_jobs=10**6,
        duration_jobs_min=1000,
        target_hi_events=1000,
        U_values=(0.6, 0.7, 0.8, 0.9),
        N_values=(100, 1000, 10000, 100000),
    ),
}


def _default_output() -> str:
    return str(Path("results") / "sweep.parquet")


def build_population(
    n_replicates: int,
    U: float,
    seed: int,
    max_attempts_factor: int = 400,
    **kwargs,
) -> tuple[list[TaskSet], int]:
    """Generate task sets until `n_replicates` of them qualify.

    A task set qualifies if it is unschedulable under single-criticality FPPS
    (so it genuinely needs AMC) and schedulable under AMC-rtb with Audsley's
    optimal priority assignment. Qualifying task sets are returned with their
    OPA priorities already assigned.

    Returns:
        (task sets, number of candidates generated)
    """
    base = seed + int(round(U * 1000)) * 1_000_003
    accepted: list[TaskSet] = []
    attempts = 0
    limit = n_replicates * max_attempts_factor

    while len(accepted) < n_replicates and attempts < limit:
        ts = generate_taskset(U=U, rng_seed=base + attempts, **kwargs)
        attempts += 1
        if fpps_schedulable(ts):
            continue  # no mixed-criticality scheme needed
        if not assign_audsley_opa(ts):
            continue  # not schedulable under AMC-rtb by any priority ordering
        accepted.append(ts)

    return accepted, attempts


def _hi_release_rate(ts: TaskSet) -> float:
    """HI-criticality releases per tick, summed over HI-criticality tasks."""
    return sum(1.0 / ts.T[i] for i in range(ts.n) if ts.criticality[i] == "HI")


def _duration_ticks(
    T_max: int,
    hi_rate: float,
    fp: float,
    preset: Scale,
) -> int:
    """Simulation duration for one (task set, N) cell, in ticks.

    With no `target_hi_events`, always runs the full `duration_jobs`. With one
    set, scales to the length expected to produce that many HI-behaviour jobs,
    clamped to `[duration_jobs_min, duration_jobs]` jobs of the longest period.
    """
    if preset.target_hi_events is None or hi_rate <= 0 or fp <= 0:
        return preset.duration_jobs * T_max

    ticks_for_target = preset.target_hi_events / (hi_rate * fp)
    jobs = ticks_for_target / T_max
    jobs = max(preset.duration_jobs_min, min(preset.duration_jobs, jobs))
    return int(round(jobs)) * T_max


@dataclass
class _SimJob:
    """One (task set, N, protocol) simulation, as a unit of parallel work."""

    U: float
    N: int
    fp: float
    hi_mode: str
    protocol: str
    replicate_index: int
    ts: TaskSet
    r_lo: list[float]
    duration: int
    seed: int


def _make_protocol(name: str, r_lo: list[float]):
    if name == "amc_rh":
        return AMC_RH(r_lo)
    if name == "amc_ra":
        return AMC_RA(r_lo)
    return OriginalAMC()


def _execute_job(job: _SimJob) -> dict:
    """Run one simulation and return its result row.

    Module-level so it can be pickled and sent to worker processes.
    """
    r = simulate(
        job.ts,
        duration=job.duration,
        seed=job.seed,
        mode_protocol=_make_protocol(job.protocol, job.r_lo),
        fp=job.fp,
    )
    return {
        "U": job.U,
        "N": job.N,
        "FP": job.fp,
        "hi_mode": job.hi_mode,
        "protocol": job.protocol,
        "replicate_index": job.replicate_index,
        "duration": job.duration,
        "nid": r.nid,
        "tid": r.tid,
        "jne": r.jne,
        "ldm": r.ldm,
        "hdm": r.hdm,
        "nid_pct": r.nid_pct,
        "tid_pct": r.tid_pct,
        "jne_ldm_pct": r.jne_ldm_pct,
        "hi_trigger_events": r.hi_trigger_events,
        "total_hi_releases": r.total_hi_releases,
        "total_lo_releases": r.total_lo_releases,
        "budget_overruns": r.budget_overruns,
        "zero_budget_count": job.ts.zero_budget_count,
        "aggregate_hi_utilisation": job.ts.aggregate_hi_utilisation,
    }


def run_sweep(
    scale: str = "debug",
    U_values: Optional[Sequence[float]] = None,
    N_values: Optional[Sequence[int]] = None,
    n_replicates: Optional[int] = None,
    duration_jobs: Optional[int] = None,
    protocols: Sequence[str] = PROTOCOLS,
    hi_mode: str = "drs_independent",
    n: int = 20,
    CP: float = 0.5,
    CF: float = 2.0,
    seed: int = 42,
    output: Optional[str] = None,
    power_threshold: int = 100,
    n_workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the AMC experiment sweep.

    Args:
        scale: Sizing preset, "debug" or "paper". Any of the sizing arguments
            below override it individually.
        U_values: Utilisation levels.
        N_values: Failure-probability levels (FP = 1/N).
        n_replicates: Qualifying task sets per utilisation level.
        duration_jobs: Simulation length in jobs of the longest-period task
            (upper bound, if the preset scales duration to statistical power).
        protocols: Which mode-change protocols to simulate.
        hi_mode: Task-set generation mode.
        n: Tasks per task set.
        CP: Criticality proportion.
        CF: Criticality factor.
        seed: Base random seed.
        output: Output parquet path. None writes nothing.
        power_threshold: Warn when a cell sees fewer HI-behaviour jobs than this.
        n_workers: Simulations run in parallel across this many processes.
            1 (the default) runs serially in this process, which is what the
            test suite relies on for deterministic, low-overhead runs. Every
            (task set, N, protocol) simulation is independent, so this scales
            close to linearly with core count.
        verbose: Print progress.

    Returns:
        DataFrame with one row per (U, N, protocol, replicate).
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
    preset = SCALES[scale]

    U_values = tuple(preset.U_values if U_values is None else U_values)
    N_values = tuple(preset.N_values if N_values is None else N_values)
    n_replicates = preset.n_replicates if n_replicates is None else n_replicates
    duration_jobs = preset.duration_jobs if duration_jobs is None else duration_jobs
    if duration_jobs != preset.duration_jobs:
        # An explicit override means "run this many jobs, flat" -- scaling to
        # statistical power on top of a value the caller just set defeats the
        # point of overriding it.
        preset = Scale(
            n_replicates=preset.n_replicates,
            duration_jobs=duration_jobs,
            U_values=preset.U_values,
            N_values=preset.N_values,
        )

    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocols {sorted(unknown)}; expected {list(PROTOCOLS)}")

    # --- Phase 1: generate qualifying task sets and their AMC-rtb analysis ---
    # Cheap relative to simulation, so this stays sequential.
    jobs: list[_SimJob] = []
    for U in U_values:
        population, attempts = build_population(
            n_replicates=n_replicates,
            U=U,
            seed=seed,
            n=n,
            CP=CP,
            CF=CF,
            hi_mode=hi_mode,
        )
        if verbose:
            yield_pct = 100.0 * len(population) / attempts if attempts else 0.0
            print(
                f"U={U:.2f}: {len(population)} qualifying task sets "
                f"from {attempts} candidates ({yield_pct:.1f}% yield)"
            )
        if not population:
            if verbose:
                print(f"  WARNING: no non-trivial schedulable task sets at U={U:.2f}; skipping")
            continue
        if len(population) < n_replicates and verbose:
            print(
                f"  WARNING: only {len(population)} of {n_replicates} task sets found at "
                f"U={U:.2f}; the non-trivial population is thin here"
            )

        for i, ts in enumerate(population):
            r_lo = amc_rtb(ts).r_lo
            hi_rate = _hi_release_rate(ts)
            T_max = max(ts.T)
            for N in N_values:
                fp = 1.0 / N
                duration = _duration_ticks(T_max, hi_rate, fp, preset)
                for name in protocols:
                    jobs.append(
                        _SimJob(
                            U=U,
                            N=N,
                            fp=fp,
                            hi_mode=hi_mode,
                            protocol=name,
                            replicate_index=i,
                            ts=ts,
                            r_lo=r_lo,
                            duration=duration,
                            seed=seed + i,
                        )
                    )

    if not jobs:
        return pd.DataFrame()

    if verbose:
        print(f"\n{len(jobs)} simulations queued across {n_workers} worker(s)")

    # --- Phase 2: run the simulations -----------------------------------------
    rows = (
        _run_parallel(jobs, n_workers, verbose)
        if n_workers > 1
        else _run_serial(jobs, verbose)
    )

    df = pd.DataFrame(rows)

    # --- Phase 3: statistical-power warnings, computed post-hoc so they work
    # the same whether the run was serial or parallel ---------------------------
    if verbose and not df.empty:
        base = df[df["protocol"] == protocols[0]]
        thin = base.groupby(["U", "N"])["hi_trigger_events"].sum()
        for (U, N), triggers in thin.items():
            if triggers < power_threshold:
                print(
                    f"  WARNING: only {int(triggers)} HI-behaviour jobs at "
                    f"U={U:.2f}, N={N}; NiD/TiD/JNE estimates will be noisy"
                )

    if output and not df.empty:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        df.to_parquet(output, index=False)
        if verbose:
            print(f"\nResults saved to {output} ({len(df)} rows)")
    return df


def _run_serial(jobs: list[_SimJob], verbose: bool) -> list[dict]:
    rows = []
    total = len(jobs)
    report_every = max(1, total // 20)
    t0 = time.monotonic()
    for k, job in enumerate(jobs):
        rows.append(_execute_job(job))
        if verbose and (k + 1) % report_every == 0:
            elapsed = time.monotonic() - t0
            rate = (k + 1) / elapsed if elapsed > 0 else 0
            eta = (total - k - 1) / rate if rate > 0 else float("nan")
            print(f"  {k + 1}/{total} ({rate:.1f}/s, eta {eta / 60:.1f} min)")
    return rows


def _run_parallel(jobs: list[_SimJob], n_workers: int, verbose: bool) -> list[dict]:
    rows = []
    total = len(jobs)
    report_every = max(1, total // 20)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_execute_job, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if verbose and (k + 1) % report_every == 0:
                elapsed = time.monotonic() - t0
                rate = (k + 1) / elapsed if elapsed > 0 else 0
                eta = (total - k - 1) / rate if rate > 0 else float("nan")
                print(f"  {k + 1}/{total} ({rate:.1f}/s, eta {eta / 60:.1f} min)")
    return rows
