"""Event-driven simulator for AMC fixed-priority preemptive scheduling.

Simulates a single :class:`TaskSet` on one core using discrete-event simulation
and collects the five metrics defined in Section V-A of the AMC-RH paper
(RTAS 2022): HDM, JNE, LDM, TiD and NiD.

Model
-----
The system has a *normal* mode and a *degraded* mode.  In degraded mode, new
releases of LO-criticality jobs are abandoned; jobs already released continue to
execute.  Entry to and exit from degraded mode is delegated to a
:class:`ModeChangeProtocol`, which sees the run-queue and, crucially, gets to
name the *absolute time* at which the transition must occur.  The engine then
schedules that instant as a first-class event, so a mode change happens on the
exact tick the protocol specifies rather than at the next unrelated event.

Job execution times are drawn once at release, per Section V-D of the paper:

===========================  ==============================
Job                          Execution time
===========================  ==============================
LO-criticality               ``U[BCET_i, C_i(LO)]``
HI-criticality, normal       ``U[BCET_i, C_i(LO)]``
HI-criticality, HI behaviour ``U[C_i(LO), C_i(HI)]``
===========================  ==============================

with HI-criticality behaviour drawn per job with probability ``fp``.

Priority level-i busy period start times ``s[i]`` are tracked using the O(1)
per-release rule from Appendix B of the paper; protocols that trigger on
response times (AMC-RH, AMC-RA) read them off each job.
"""

from __future__ import annotations

import heapq
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from amc_tasksim.generation.taskset import TaskSet

ExecTimeMode = Literal["random", "wcet"]


@dataclass
class Job:
    """A single job release.

    Attributes:
        task_id: Index of the task this job belongs to.
        seq: Release sequence number, used to break priority ties in arrival order.
        release: Release time (integer tick).
        deadline: Absolute deadline.
        c_lo: LO-criticality execution time budget of the task.
        c_hi: HI-criticality execution time budget of the task.
        criticality: "HI" or "LO".
        priority: Priority level (lower number = higher priority).
        exec_time: Execution time drawn for this job at release.
        budget: Enforced execution time budget (C_hi for HI, C_lo for LO).
        hi_behaviour: Whether this job was selected to exhibit HI-criticality behaviour.
        busy_start: Start time of the priority level-i busy period in which this
            job was released (Appendix B of the AMC-RH paper).
        executed: Execution time consumed so far.
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
        """Execution time still to be consumed."""
        return self.exec_time - self.executed

    @property
    def overruns(self) -> bool:
        """True if this job executes beyond C_i(LO) without signalling completion.

        A job whose execution time is exactly C_i(LO) signals completion at the
        instant it reaches that budget, so it does not trigger a mode change.
        """
        return self.exec_time > self.c_lo

    def sort_key(self) -> tuple[int, int]:
        return (self.priority, self.seq)


@dataclass
class SchedulerState:
    """The view of the system handed to a mode-change protocol.

    Attributes:
        time: Current simulation time.
        active: Jobs released but not yet completed or abandoned, ordered by
            priority (highest first) then arrival. This is the run-queue.
        running: The job currently executing, or None if the processor is idle.
        mode: "normal" or "degraded".
    """

    time: int = 0
    active: list[Job] = field(default_factory=list)
    running: Optional[Job] = None
    mode: str = "normal"


@dataclass
class SimulationResult:
    """Metrics from a single simulation run.

    The five metrics of Section V-A of the AMC-RH paper, plus the denominators
    needed to express them as the percentages the paper actually plots.

    Attributes:
        nid: Number of times degraded mode was entered.
        tid: Fraction of the simulation spent in degraded mode.
        jne: LO-criticality jobs abandoned on release in degraded mode.
        ldm: LO-criticality jobs that executed but missed their deadline.
        hdm: HI-criticality jobs that missed their deadline (should be zero).
        hi_releases_per_task: HI-criticality job releases per task.
        lo_releases_per_task: LO-criticality job releases per task, including
            jobs abandoned in degraded mode.
        hi_trigger_events: Jobs selected to exhibit HI-criticality behaviour.
        degraded_ticks: Total time spent in degraded mode.
        duration: Simulation duration.
        budget_overruns: Jobs that exceeded their enforced budget (defensive
            check; should always be zero).
    """

    nid: int = 0
    tid: float = 0.0
    jne: int = 0
    ldm: int = 0
    hdm: int = 0
    hi_releases_per_task: list[int] = field(default_factory=list)
    lo_releases_per_task: list[int] = field(default_factory=list)
    hi_trigger_events: int = 0
    degraded_ticks: int = 0
    duration: int = 0
    budget_overruns: int = 0

    @property
    def total_hi_releases(self) -> int:
        return sum(self.hi_releases_per_task)

    @property
    def total_lo_releases(self) -> int:
        return sum(self.lo_releases_per_task)

    @property
    def nid_pct(self) -> float:
        """NiD as a percentage of the number of HI-criticality jobs."""
        n = self.total_hi_releases
        return 100.0 * self.nid / n if n else 0.0

    @property
    def tid_pct(self) -> float:
        """TiD as a percentage of total simulation time."""
        return 100.0 * self.tid

    @property
    def jne_ldm_pct(self) -> float:
        """JNE + LDM as a percentage of the number of LO-criticality jobs."""
        n = self.total_lo_releases
        return 100.0 * (self.jne + self.ldm) / n if n else 0.0


# ---------------------------------------------------------------------------
# Mode-change protocols
# ---------------------------------------------------------------------------


class ModeChangeProtocol(ABC):
    """Entry and exit rules for degraded mode.

    ``entry_time`` returns the absolute time at which degraded mode must be
    entered given the current run-queue, so the engine can schedule it as an
    event; returning a time at or before ``state.time`` means "enter now".
    """

    name = "protocol"

    @abstractmethod
    def entry_time(self, state: SchedulerState) -> Optional[int]:
        """Absolute time at which degraded mode must be entered, or None."""
        ...

    @abstractmethod
    def should_exit(self, state: SchedulerState) -> bool:
        """Whether degraded mode should be left at ``state.time``."""
        ...


class OriginalAMC(ModeChangeProtocol):
    """The original AMC runtime protocol (referred to as AMC+ in the paper).

    Enter degraded mode when a job of a HI-criticality task has executed for
    C_i(LO) without signalling completion; return to normal mode on an idle
    instant.
    """

    name = "original_amc"

    def entry_time(self, state: SchedulerState) -> Optional[int]:
        best: Optional[int] = None
        for job in state.active:
            if job.criticality != "HI" or not job.overruns:
                continue
            if job.executed >= job.c_lo:
                return state.time
            # Only the running job accrues execution time, so only it has a
            # predictable time at which it will reach C_i(LO).
            if job is state.running:
                cand = state.time + (job.c_lo - job.executed)
                best = cand if best is None else min(best, cand)
        return best

    def should_exit(self, state: SchedulerState) -> bool:
        return not state.active


class _ResponseTimeTrigger(ModeChangeProtocol):
    """Shared entry rule for AMC-RH and AMC-RA (specification S2).

    Degraded mode is entered when an active job of a HI-criticality task reaches
    a time R_i(LO) after the start of the priority level-i busy period in which
    it was released.
    """

    def __init__(self, r_lo):
        # R_i(LO) is integral for integer task parameters; round up defensively
        # so the trigger is never brought forward by floating-point noise.
        self.r_lo = [int(math.ceil(float(x))) for x in r_lo]

    def expiry(self, job: Job) -> int:
        """Absolute time at which this job reaches R_i(LO) from its busy period."""
        return job.busy_start + self.r_lo[job.task_id]

    def entry_time(self, state: SchedulerState) -> Optional[int]:
        best: Optional[int] = None
        for job in state.active:
            if job.criticality != "HI":
                continue
            e = self.expiry(job)
            if e <= state.time:
                return state.time
            best = e if best is None else min(best, e)
        return best


class AMC_RH(_ResponseTimeTrigger):
    """AMC-RH: specifications S1, S2, S3, S4 of the AMC-RH paper.

    Exit degraded mode when a job of a HI-criticality task completes and no
    other active HI-criticality job has reached R_k(LO) after the start of the
    priority level-k busy period in which it was released.
    """

    name = "amc_rh"

    def should_exit(self, state: SchedulerState) -> bool:
        for job in state.active:
            if job.criticality == "HI" and state.time >= self.expiry(job):
                return False
        return True


class AMC_RA(_ResponseTimeTrigger):
    """AMC-RA: specifications S1, S2, S5, S4 of the AMC-RH paper.

    Same entry rule as AMC-RH; exit on an idle instant, as in the original AMC
    scheme.
    """

    name = "amc_ra"

    def should_exit(self, state: SchedulerState) -> bool:
        return not state.active


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _skip_warm_up(taskset: TaskSet, skip_quiet: bool) -> Optional[int]:
    """Warm-up window for fast-forwarding, or None if it is not safe.

    Skipping an interval is only sound if nothing happens in it that the
    metrics care about. With no HI-criticality behaviour the system stays in
    normal mode, so no job is abandoned and no mode change occurs; the only
    remaining risk is a deadline miss, which requirement R1 rules out for a
    task set that passes normal-mode response-time analysis. A task set that
    fails it is simulated exactly instead.
    """
    if not skip_quiet:
        return None

    from amc_tasksim.scheduling.amc_rtb import busy_period_bound, normal_mode_schedulable

    if not normal_mode_schedulable(taskset):
        warnings.warn(
            "skip_quiet requested but the task set misses deadlines even when every "
            "job complies with C_i(LO); simulating exactly instead",
            stacklevel=3,
        )
        return None

    bound = busy_period_bound(taskset)
    if bound is None or bound <= 0:
        warnings.warn(
            "skip_quiet requested but the busy-period bound does not converge "
            "(LO-criticality utilisation at or above 1); simulating exactly instead",
            stacklevel=3,
        )
        return None

    # Two separate lookbacks are needed, and the busy period alone is not
    # enough -- it is typically shorter than the longest period.
    #
    #  - A job still pending at the resume instant may have been released up to
    #    its own deadline earlier, so the queue cannot be reconstructed without
    #    going back max(D_i).
    #  - For those jobs' own busy-period start times to be right, the queue has
    #    to be correct from that point too, which needs a further busy period.
    #
    # Warming up for less produces a queue that is too empty at the trigger,
    # which understates interference and suppresses mode changes.
    return max(taskset.D) + bound


class _TriggerSchedule:
    """Which HI-criticality job releases exhibit HI-criticality behaviour.

    The paper's model draws an independent Bernoulli(fp) per HI-criticality
    release. Drawing them one at a time forces every release to be simulated;
    but the number of releases of a given task between one HI-behaviour job and
    the next is Geometric(fp), so the same process can be sampled by jumping
    straight to the next triggering release index. Because the tasks are
    independent, each is tracked separately and the earliest wins -- no merged
    release sequence or binary search is needed.

    This is distributionally identical to the per-release Bernoulli, not an
    approximation of it.
    """

    def __init__(
        self,
        taskset: TaskSet,
        fp: float,
        offsets: list[int],
        rng: np.random.Generator,
    ) -> None:
        self.fp = fp
        self.T = taskset.T
        self.offsets = offsets
        self.hi_tasks = [i for i in range(taskset.n) if taskset.criticality[i] == "HI"]
        # Index (0-based, per task) of that task's next triggering release.
        self.next_index: dict[int, int] = {}
        for i in self.hi_tasks:
            self.next_index[i] = self._gap(rng) - 1 if fp > 0 else -1

    def _gap(self, rng: np.random.Generator) -> int:
        """Releases from now to and including the next triggering one (>= 1)."""
        return int(rng.geometric(self.fp))

    def enabled(self) -> bool:
        return self.fp > 0 and bool(self.hi_tasks)

    def triggers(self, task_id: int, release_index: int) -> bool:
        return self.next_index.get(task_id, -1) == release_index

    def consume(self, task_id: int, rng: np.random.Generator) -> None:
        """Advance past a triggering release that has just been issued."""
        self.next_index[task_id] += self._gap(rng)

    def next_trigger_time(self, horizon: int) -> Optional[int]:
        """Earliest time at which any HI-criticality task exhibits HI behaviour."""
        if not self.enabled():
            return None
        best: Optional[int] = None
        for i in self.hi_tasks:
            t = self.offsets[i] + self.next_index[i] * self.T[i]
            if t < horizon and (best is None or t < best):
                best = t
        return best


def _draw_exec_time(
    rng: np.random.Generator,
    taskset: TaskSet,
    i: int,
    hi_behaviour: bool,
    exec_time_mode: ExecTimeMode,
) -> int:
    """Draw the execution time for one job, per Section V-D of the paper."""
    c_lo = taskset.C_lo[i]
    c_hi = taskset.C_hi[i]

    if hi_behaviour:
        if exec_time_mode == "wcet":
            return c_hi
        if c_hi <= c_lo:
            return c_hi
        return int(rng.integers(c_lo, c_hi + 1))

    if exec_time_mode == "wcet":
        return c_lo
    bcet = min(taskset.BCET[i], c_lo)
    if bcet >= c_lo:
        return c_lo
    return int(rng.integers(bcet, c_lo + 1))


def simulate(
    taskset: TaskSet,
    duration: int = 10**6,
    seed: Optional[int] = None,
    mode_protocol: Optional[ModeChangeProtocol] = None,
    fp: float = 1e-4,
    release_offsets: Optional[list[Optional[int]]] = None,
    exec_time_mode: ExecTimeMode = "random",
    trace: Optional[list[tuple[int, str, int]]] = None,
    skip_quiet: bool = False,
) -> SimulationResult:
    """Simulate a task set under fixed-priority preemptive scheduling.

    Events are job releases, job completions, absolute deadlines, and the
    mode-change trigger instant named by the protocol. Time advances directly
    from one event to the next.

    A job that has not completed by its absolute deadline is counted as a
    deadline miss (LDM or HDM) and abandoned at that instant, which keeps at
    most one job of each task in the run-queue as the constrained-deadline model
    requires.

    Args:
        taskset: Task set to simulate. Priorities are assigned if absent.
        duration: Simulation duration in ticks.
        seed: Random seed for execution times and HI-behaviour draws.
        mode_protocol: Mode-change protocol (default: :class:`OriginalAMC`).
        fp: Per-job probability that a HI-criticality task exhibits
            HI-criticality behaviour.
        release_offsets: Per-task phase of the first release (default: 0 for
            every task, i.e. a synchronous arrival at t=0). Subsequent releases
            follow strictly periodically.
        exec_time_mode: "random" draws execution times as the paper describes;
            "wcet" makes every job execute exactly its budget, which is used to
            reproduce the paper's worked scenarios deterministically.
        trace: If given, ``(time, event, task_id)`` tuples are appended for each
            release, drop, completion, deadline miss and mode change. Intended
            for validating scripted scenarios, not for long runs.
        skip_quiet: Fast-forward through stretches with no HI-criticality
            behaviour instead of simulating them. Requires the task set to
            satisfy requirement R1 (every task meets its deadline when all jobs
            comply with C_i(LO)); the request is ignored, with a warning, if it
            does not. Statistically equivalent to the exact simulation, but a
            different sample path -- see the module docstring.

    Returns:
        A :class:`SimulationResult`.
    """
    if mode_protocol is None:
        mode_protocol = OriginalAMC()

    rng = np.random.default_rng(seed)

    if not taskset.priority:
        from amc_tasksim.scheduling.priority import assign_deadline_monotonic

        assign_deadline_monotonic(taskset)

    n = taskset.n
    result = SimulationResult(duration=duration)
    result.hi_releases_per_task = [0] * n
    result.lo_releases_per_task = [0] * n

    # Pending releases as a min-heap on release time, rather than an array
    # rescanned every event: with n tasks and usually 0-1 release per event,
    # rescanning all n every step dominates the runtime at long durations
    # (measured: ~2/3 of wall time) for no reason, since almost every task
    # is not due.
    offsets = [0] * n
    for i in range(n):
        if release_offsets is not None and release_offsets[i] is not None:
            offsets[i] = int(release_offsets[i])
    release_heap: list[tuple[int, int]] = [(offsets[i], i) for i in range(n)]
    heapq.heapify(release_heap)

    priority = taskset.priority  # local binding: hot-path attribute access

    schedule = _TriggerSchedule(taskset, fp, offsets, rng)

    # --- fast-forward setup -------------------------------------------------
    warm_up = _skip_warm_up(taskset, skip_quiet)
    skipping = warm_up is not None and schedule.enabled() and exec_time_mode == "random"

    state = SchedulerState()
    active: list[Job] = state.active
    seq = 0
    degraded_start = -1

    def enter_if_triggered(now: int) -> None:
        """Enter degraded mode if the protocol's trigger condition already holds.

        Called before the releases at `now`, because the trigger depends only on
        execution accrued strictly before `now` and a job released at the instant
        of the transition is released into degraded mode. This is what Figure 13
        of the AMC-RH paper shows for the release of tau1 at t=8.
        """
        nonlocal degraded_start
        if state.mode != "normal":
            return
        state.running = active[0] if active else None
        e = mode_protocol.entry_time(state)
        if e is not None and e <= now:
            state.mode = "degraded"
            result.nid += 1
            degraded_start = now
            if trace is not None:
                trace.append((now, "enter_degraded", -1))

    while state.time < duration:
        now = state.time

        # --- fast-forward through a stretch with no HI-criticality behaviour --
        # Only from normal mode, and only far enough back from the next
        # triggering release for the warm-up to re-establish the run-queue.
        if skipping and state.mode == "normal":
            trigger_at = schedule.next_trigger_time(duration)
            target = duration if trigger_at is None else trigger_at
            resume = target - warm_up
            if resume - now > warm_up:
                # Releases in [now, resume) are the only thing the skipped
                # interval contributes, and they are periodic, so count them
                # rather than simulating them.
                for idx in range(len(release_heap)):
                    t_next, i = release_heap[idx]
                    if t_next < resume:
                        missed = -(-(resume - t_next) // taskset.T[i])
                        if taskset.criticality[i] == "HI":
                            result.hi_releases_per_task[i] += missed
                        else:
                            result.lo_releases_per_task[i] += missed
                        release_heap[idx] = (t_next + missed * taskset.T[i], i)
                heapq.heapify(release_heap)
                # The pending jobs are abandoned: they are guaranteed to
                # complete by their deadlines, so they leave no trace in any
                # metric, and the warm-up rebuilds an equivalent queue.
                active.clear()
                state.running = None
                if trace is not None:
                    trace.append((now, "skip_to", resume))
                state.time = resume
                now = resume

        # --- deadline expiries at `now` --------------------------------------
        if active:
            expired = [j for j in active if j.deadline <= now]
            if expired:
                for job in expired:
                    if job.criticality == "HI":
                        result.hdm += 1
                    else:
                        result.ldm += 1
                    active.remove(job)
                    if trace is not None:
                        trace.append((now, "deadline_miss", job.task_id))

        # --- leave degraded mode ---------------------------------------------
        # Evaluated before releases at `now`, because an idle instant is defined
        # over jobs released strictly before the current time.
        if state.mode == "degraded" and mode_protocol.should_exit(state):
            state.mode = "normal"
            result.degraded_ticks += now - degraded_start
            degraded_start = -1
            if trace is not None:
                trace.append((now, "exit_degraded", -1))

        # --- enter degraded mode ----------------------------------------------
        enter_if_triggered(now)

        # --- releases at `now` ------------------------------------------------
        # Pop everything due, in priority order (highest first) so simultaneous
        # releases inherit busy-period start times correctly (Appendix B).
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

            # Which releases exhibit HI-criticality behaviour comes from the
            # schedule, which samples the same Bernoulli process the paper
            # describes but by geometric jumps rather than one draw per
            # release. The execution-time draw happens for every release,
            # including one about to be abandoned, so the stream depends only
            # on the release sequence -- that keeps runs of the same task set
            # and seed under different protocols a precise like-for-like
            # comparison (Section V-D of the AMC-RH paper).
            hi_behaviour = False
            if crit == "HI" and schedule.enabled():
                release_index = (now - offsets[i]) // taskset.T[i]
                if schedule.triggers(i, release_index):
                    hi_behaviour = True
                    schedule.consume(i, rng)
            exec_time = _draw_exec_time(rng, taskset, i, hi_behaviour, exec_time_mode)

            if state.mode == "degraded" and crit == "LO":
                # Abandoned on release: not queued, not executed.
                result.jne += 1
                if trace is not None:
                    trace.append((now, "drop", i))
            else:
                if hi_behaviour:
                    result.hi_trigger_events += 1
                if exec_time > 0:
                    job = Job(
                        task_id=i,
                        seq=seq,
                        release=now,
                        deadline=now + taskset.D[i],
                        c_lo=taskset.C_lo[i],
                        c_hi=taskset.C_hi[i],
                        criticality=crit,
                        priority=taskset.priority[i],
                        exec_time=exec_time,
                        budget=taskset.C_hi[i] if crit == "HI" else taskset.C_lo[i],
                        hi_behaviour=hi_behaviour,
                    )
                    seq += 1
                    if job.exec_time > job.budget:
                        result.budget_overruns += 1
                        warnings.warn(
                            f"task {i}: execution time {job.exec_time} exceeds its "
                            f"enforced budget {job.budget}; job truncated",
                            stacklevel=2,
                        )
                        job.exec_time = job.budget
                    # Insert in run-queue order and inherit the busy period start
                    # from the job immediately ahead (Appendix B).
                    key = job.sort_key()
                    idx = 0
                    while idx < len(active) and active[idx].sort_key() < key:
                        idx += 1
                    job.busy_start = now if idx == 0 else active[idx - 1].busy_start
                    active.insert(idx, job)
                    if trace is not None:
                        trace.append((now, "release", i))
                    if crit == "HI":
                        # A HI job that inherits a busy period already longer
                        # than R_i(LO) is past its trigger the moment it is
                        # queued, so lower-priority releases at this same
                        # instant must see degraded mode.
                        enter_if_triggered(now)
                # exec_time == 0 (a task whose C_i(LO) rounded to zero) occupies
                # no processor time; it is counted as released and completes at
                # its release instant.
            heapq.heappush(release_heap, (now + taskset.T[i], i))

        # --- select the running job -------------------------------------------
        state.running = active[0] if active else None

        # --- the trigger instant is itself an event ----------------------------
        entry = mode_protocol.entry_time(state) if state.mode == "normal" else None

        # --- next event --------------------------------------------------------
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
        if state.mode == "normal" and entry is not None and entry < next_t:
            next_t = entry

        if next_t <= now:
            # Defensive: never stall. Should be unreachable.
            warnings.warn(
                f"simulation made no progress at t={now}; forcing advance",
                stacklevel=2,
            )
            next_t = now + 1

        # --- advance ------------------------------------------------------------
        if state.running is not None:
            state.running.executed += next_t - now
        state.time = next_t

        if state.running is not None and state.running.remaining <= 0:
            active.remove(state.running)
            if trace is not None:
                trace.append((state.time, "complete", state.running.task_id))
            state.running = None

    # Close out an interval of degraded mode still open at the horizon.
    if state.mode == "degraded" and degraded_start >= 0:
        result.degraded_ticks += duration - degraded_start

    result.tid = result.degraded_ticks / duration if duration > 0 else 0.0
    return result
