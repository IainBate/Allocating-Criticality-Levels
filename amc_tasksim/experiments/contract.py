"""The measurement contract: how any two configurations get compared.

Every phase of the multi-level study imports this module rather than restating
its own grid, so results from different phases compose. That is the point of
putting it in code: a grid written in prose drifts, and four documents
specifying four different utilisation ranges is how a study loses the ability
to say "k=4 beats k=3" at all.

Pairing is the part that cannot be added afterwards
---------------------------------------------------
Configurations are compared on the *same* task sets and the *same* seeds --
common random numbers -- and the statistic is the mean of the per-pair
differences, not the difference of the two means. Measured on this simulator at
a fixed budget of 20 task sets x 20 seeds per configuration:

======================================  =========  ============
Design                                   Std err    Resolvable
======================================  =========  ============
Different task sets per configuration      19.504         89.8%
Same task sets, analysed unpaired           7.719         35.5%
Same task sets, analysed paired             0.866          4.0%
======================================  =========  ============

So pairing is worth ~79x against the same population analysed carelessly, and
~507x against the natural but wrong instinct of generating a fresh population
per configuration. Only the third row can support the 5% threshold in
:data:`EFFECT_FLOOR`.

Most methodology can be tightened retrospectively; this cannot. An unpaired run
is not merely weaker, it is unrecoverable, because the pairing has to exist at
generation time. Hence :func:`paired_compare` refuses unequal-length samples
rather than falling back to an unpaired test.

Pair at the task set, not at the individual run
-----------------------------------------------
Feed :func:`paired_compare` one value per task set -- the mean over that task
set's seeds -- via :func:`aggregate_by_taskset`, not one value per simulation.

Two different configurations consume the random stream differently, so their
individual runs at the same seed decorrelate and per-run pairing recovers only
a few times the variance. Task-set means stay strongly correlated because the
task set itself is shared, and between-task-set variance is the dominant term
(measured: sd 24.5 between task sets against 14.4 within). Averaging first and
differencing second is what cancels it.

``amc_tasksim.experiments.sweep`` already runs every protocol on the same task
sets and seeds, so it satisfies the contract; this module supplies the frozen
grid and the analysis half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# The frozen grid
# ---------------------------------------------------------------------------

#: Seeds reused for every configuration. Fixed once; never derived from the
#: configuration under test, because that would break the pairing.
SEED_BLOCK: tuple[int, ...] = tuple(range(120))

#: Held out for confirming a winner, so that selection and confirmation use
#: independent randomness. Disjoint from SEED_BLOCK by construction.
CONFIRM_SEED_BLOCK: tuple[int, ...] = tuple(range(1000, 1120))

#: Practical-significance threshold: a difference smaller than this is not
#: claimed as a result even when it is statistically significant.
EFFECT_FLOOR: float = 0.05

#: Two-sided normal quantiles for the reporting conventions below.
Z_CI: float = 1.96      # 95% confidence interval
Z_POWER: float = 1.64   # 95% power


@dataclass(frozen=True)
class Grid:
    """The canonical experiment grid.

    Attributes:
        n_tasks: Tasks per set.
        CP: Criticality proportion.
        CF: Criticality factor.
        U_levels: Utilisation levels swept.
        duration: Simulation horizon in ticks.
        n_tasksets: Qualifying task sets per utilisation level.
        k_levels: Degradation level counts studied. Excludes k=5: separating
            k=4 from k=5 needs an effect size below what this budget resolves,
            so running it would produce a number that cannot be defended.
        severity_grid_step: Resolution of the severity ladder enumeration.
    """

    n_tasks: int = 10
    CP: float = 0.5
    CF: float = 2.0
    U_levels: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    duration: int = 1_000_000
    n_tasksets: int = 40
    k_levels: tuple[int, ...] = (2, 3, 4)
    severity_grid_step: float = 0.05

    def severity_grid(self) -> list[float]:
        """Candidate severities for a ladder level, excluding the pinned x=0."""
        n = int(round(1.0 / self.severity_grid_step))
        return [i * self.severity_grid_step for i in range(1, n + 1)]

    def taskset_params(self) -> dict:
        return dict(n=self.n_tasks, CP=self.CP, CF=self.CF)


CANONICAL = Grid()


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedResult:
    """Outcome of comparing two configurations on identical randomness.

    Attributes:
        n: Number of pairs.
        baseline_mean: Mean of the baseline sample.
        mean_diff: Mean of (candidate - baseline) over pairs. Negative means
            the candidate scored lower, which is an improvement for a cost
            metric such as JNE.
        std_err: Standard error of ``mean_diff``.
        relative: ``mean_diff / baseline_mean``, or nan if the baseline is zero.
        resolvable: Smallest relative difference this sample could distinguish
            from zero at the reporting threshold. If this exceeds EFFECT_FLOOR,
            the comparison cannot support a practical-significance claim
            whatever it happens to show.
    """

    n: int
    baseline_mean: float
    mean_diff: float
    std_err: float
    relative: float
    resolvable: float

    @property
    def ci(self) -> tuple[float, float]:
        """95% confidence interval on the mean difference."""
        half = Z_CI * self.std_err
        return (self.mean_diff - half, self.mean_diff + half)

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        lo, hi = self.ci
        return lo > 0 or hi < 0

    @property
    def practically_significant(self) -> bool:
        """Significant *and* larger than the effect floor.

        Both halves are required. A tiny difference measured precisely is not a
        result worth reporting, and a large difference measured imprecisely is
        not a result at all.
        """
        return self.significant and abs(self.relative) >= EFFECT_FLOOR

    @property
    def underpowered(self) -> bool:
        """Whether the sample is too small to detect an effect at the floor."""
        return self.resolvable > EFFECT_FLOOR

    def summary(self) -> str:
        lo, hi = self.ci
        verdict = (
            "practically significant"
            if self.practically_significant
            else "significant but below the effect floor"
            if self.significant
            else "not distinguishable from zero"
        )
        warn = "  [UNDERPOWERED]" if self.underpowered else ""
        return (
            f"n={self.n}  diff={self.mean_diff:+.4g} "
            f"({self.relative:+.1%})  95% CI [{lo:+.4g}, {hi:+.4g}]  "
            f"resolvable={self.resolvable:.1%}  {verdict}{warn}"
        )


def aggregate_by_taskset(
    values: Sequence[float],
    n_seeds: int,
) -> list[float]:
    """Collapse per-run values to one value per task set, the pairing unit.

    ``values`` is ordered task-set-major: all ``n_seeds`` runs of the first task
    set, then all runs of the second, and so on -- the order produced by
    iterating task sets in the outer loop and seeds in the inner one.

    Args:
        values: Per-run metric values, task-set-major.
        n_seeds: Runs per task set.

    Returns:
        One mean per task set.

    Raises:
        ValueError: If ``values`` is not a whole number of task sets, which
            means the runs were not laid out as assumed and the aggregation
            would silently mix task sets together.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be positive, got {n_seeds}")
    if len(values) % n_seeds:
        raise ValueError(
            f"{len(values)} values is not a whole number of task sets at "
            f"{n_seeds} seeds each; check the run ordering is task-set-major"
        )
    return [
        sum(values[i : i + n_seeds]) / n_seeds
        for i in range(0, len(values), n_seeds)
    ]


def paired_compare(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> PairedResult:
    """Compare two configurations evaluated on identical (task set, seed) pairs.

    Args:
        baseline: Per-task-set values for the baseline configuration.
        candidate: Per-task-set values for the candidate, in the *same order* --
            element i of each must come from the same task set.

    Note:
        Pass one value per *task set*, via :func:`aggregate_by_taskset`, not one
        per simulation run. See the module docstring for why the unit matters.

    Returns:
        A :class:`PairedResult`.

    Raises:
        ValueError: If the samples differ in length, which almost always means
            they were not run on the same task sets and seeds. Falling back to
            an unpaired test here would silently discard the variance reduction
            the whole protocol depends on, so it is an error instead.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal-length samples, got "
            f"{len(baseline)} and {len(candidate)}; configurations must be run "
            f"on the same task sets and the same seeds"
        )
    n = len(baseline)
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")

    diffs = [c - b for b, c in zip(baseline, candidate)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_err = math.sqrt(var / n)

    baseline_mean = sum(baseline) / n
    relative = mean_diff / baseline_mean if baseline_mean else math.nan
    resolvable = (
        Z_CI * std_err / abs(baseline_mean) if baseline_mean else math.inf
    )

    return PairedResult(
        n=n,
        baseline_mean=baseline_mean,
        mean_diff=mean_diff,
        std_err=std_err,
        relative=relative,
        resolvable=resolvable,
    )


def required_pairs(
    pilot_baseline: Sequence[float],
    pilot_candidate: Sequence[float],
    effect: float = EFFECT_FLOOR,
    power_z: float = Z_POWER,
) -> int:
    """Task sets needed to detect ``effect`` at the reporting threshold.

    Estimates the per-task-set variability from a pilot run and extrapolates.
    Use it to size a study *before* committing to it, rather than discovering
    afterwards that the result was never reachable.

    This answers "how many task sets", not "how many seeds per task set" --
    confirmed the two are not interchangeable by running both: on one
    severity-ladder comparison, quadrupling seeds per task set (10 -> 40,
    fixed ensemble) left the standard error *unchanged* (0.92x, not the 2x
    that sqrt(n) scaling would predict), while the ensemble size this
    function actually recommends (16 -> 64 task sets, seeds held fixed)
    narrowed the resolvable bound as expected. The task sets in that pilot
    had a between-task-set coefficient of variation of 0.89, well above the
    per-seed noise -- when that dominates, more seeds per task set mostly
    re-measures the same task-set-level value more precisely and barely
    touches the standard error of the comparison. Adding seeds is only the
    right lever when a pilot shows *within*-task-set noise is the bottleneck;
    check that before assuming it (see research/mode_optimization.tex,
    "Budget Calibration", for the worked example this note is drawn from).

    Args:
        pilot_baseline: Baseline values from a small paired pilot, one per
            task set (already aggregated across seeds -- see
            :func:`aggregate_by_taskset`).
        pilot_candidate: Candidate values from the same pilot, same order.
        effect: Target effect size, relative to the baseline mean.
        power_z: Normal quantile for the desired power.

    Returns:
        Required number of task sets, at the pilot's seed-per-task-set count.
    """
    pilot = paired_compare(pilot_baseline, pilot_candidate)
    n = len(pilot_baseline)
    # Per-pair relative sd, recovered from the standard error of the mean.
    rel_sd = pilot.std_err * math.sqrt(n) / abs(pilot.baseline_mean)
    return math.ceil(((Z_CI + power_z) ** 2) * (rel_sd / effect) ** 2)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """A winner plus everything statistically indistinguishable from it.

    Reporting a bare argmin over many noisy evaluations overstates the winner
    by roughly ``sigma * sqrt(2 ln M)`` -- the winner's curse. Reporting the
    indifference set instead, and confirming it on held-out seeds, is what
    removes that bias.

    Attributes:
        best: Key of the lowest-scoring configuration.
        best_score: Its score.
        indifference_set: Keys within one standard error of the best, including
            the best itself. If this holds more than one entry, the study has
            not identified a unique optimum and should say so.
    """

    best: object
    best_score: float
    indifference_set: list = field(default_factory=list)

    @property
    def is_unique(self) -> bool:
        return len(self.indifference_set) == 1


def select(scores: dict, std_errs: dict) -> Selection:
    """Pick the best configuration and everything tied with it.

    Args:
        scores: Configuration key -> mean objective (lower is better).
        std_errs: Configuration key -> standard error of that mean.

    Returns:
        A :class:`Selection`. Confirm its members on ``CONFIRM_SEED_BLOCK``
        before declaring a winner.
    """
    if not scores:
        raise ValueError("no configurations to select from")
    best = min(scores, key=lambda k: scores[k])
    threshold = scores[best] + std_errs.get(best, 0.0)
    return Selection(
        best=best,
        best_score=scores[best],
        indifference_set=[k for k, v in scores.items() if v <= threshold],
    )
