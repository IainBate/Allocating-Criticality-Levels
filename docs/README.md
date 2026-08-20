# Multi-Level Mixed-Criticality Scheduling Documentation

This directory contains documentation for the multi-level mixed-criticality scheduling research project.

## Overview

This project extends traditional two-mode AMC (Adaptive Mixed-Criticality) scheduling to support **multiple criticality levels**, allowing low-criticality tasks to be dropped in stages rather than all at once when high-criticality behavior is detected.

---

## Research Plan

### `../research/multi_mode_amc.md` - Main Research Plan

The primary research plan that outlines the entire project structure:

- **Phase 1**: Foundational Analysis (task set specification, safety proofs)
- **Phase 2**: Objective Function & Metrics
- **Phase 3**: Optimal Mode Design
- **Phase 4**: Drop Strategy Research
- **Phase 5**: Implementation & Validation

---

## Phase 1: Foundational Analysis

### Task Set Specification

| File | Description |
|------|-------------|
| `task_set_specification.md` | Worked example from AMC-RH Appendix A with manual response time calculations |

### Safety Proofs

| File | Description |
|------|-------------|
| `safety_proof.md` | Proven properties: HI tasks meet deadlines, no worse than AMC-RH, monotonic drop sets |
| `proof_requirements.md` | Categorization of proven (4), assumed (6), and open (5) proof questions |

---

## Phase 2: Metrics & Complexity

### Metrics and Objective Function

| File | Description |
|------|-------------|
| `metrics_objective.md` | Extended metrics (WastedCPU, ServiceRatio, LevelTrans); objective function candidates; "Meaningful Improvement" criteria |
| `complexity_analysis.md` | Time/space complexity analysis: O(k×n) trigger check, O(m) drop decision |

---

## Phase 3: Optimal Mode Design

| File | Description |
|------|-------------|
| `mode_optimization.md` | **Phase 3 plan** - severity-lattice enumeration under common random numbers (single method; rationale below) |
| `../research/mode_optimization.tex` | Paper: problem characterisation (search-space size, objective noise), related work on optimisation techniques, and the recommendation |

**Key questions addressed**:
- What is the optimal number of degradation levels?
- How should severities be spaced (χ_2 ≤ χ_3 ≤ ... ≤ χ_{k-2}, with χ_1 = 0 and χ_{k-1} pinned)?
- Which optimisation method is appropriate, given the search space is small enough to enumerate exhaustively and the binding constraint is Monte-Carlo noise rather than search cost (see `mode_optimization.tex`)?

---

## Phase 4: Drop Strategy Research

| File | Description |
|------|-------------|
| `drop_strategy_research.md` | **Phase 4 plan** - Strategies for selecting which LO tasks to drop and when to exit degraded modes. Written before the revised protocol ran; Task 4.1's core question is now answered (below), Task 4.2 is not |

**Status**:
- **Task 4.1 (which tasks to drop): answered.** Utilisation-ordered shed-early beats
  priority-based dropping by 73–78% (`scheduling/drop_sets.py`), and Stage 3's exhaustive
  search (`../research/mode_optimization.tex` §"Bounded-Gain Configuration Selection")
  bounds any method's further gain over that default at ≤0.79% — an order of magnitude
  under this study's 5% threshold. No further drop-strategy comparison is planned.
- **Task 4.2 (exit strategy / hysteresis): still open, and currently blocked.**
  `safety_proof.md` (Theorem 2 scope note) records that a hysteresis-based exit rule
  cannot be measured exactly in the current simulator — `ModeChangeProtocol` has no
  `exit_time()`, only a `should_exit()` predicate, and a timed exit rule was measured to
  inflate `degraded_ticks` by ~15–20%. Adding `exit_time()` is a prerequisite for this
  task, not just an unrun experiment.

---

## Phase 5: Implementation & Validation

| File | Description |
|------|-------------|
| `implementation_validation_plan.md` | **Phase 5 plan** - Protocol interface extension, validation tests, experiment framework. Written as pseudocode before the engine existed; the actual implementation is in `amc_tasksim/`, below |

**Status**:
- **Task 5.1 (core k-level engine): built and tested**, not merely planned —
  `amc_tasksim/simulation/multilevel.py` and `amc_tasksim/experiments/multilevel_protocol.py`,
  with tests in `tests/simulation/test_multilevel.py` and `tests/experiments/test_multilevel_protocol.py`.
- **Task 5.3 (experiment/sweep framework): built and used**, not just specified —
  `amc_tasksim/experiments/contract.py` (paired evaluation, `required_pairs()`) and
  `sweep.py` ran all four stages of the revised protocol
  (`../research/mode_optimization.tex` §"Stage Results").
- **Task 5.2 (validation suite): partially covered.** `tests/simulation/test_validation.py`
  checks the engine against the AMC-RH paper scenario directly; it has not been confirmed
  against every specific check the original plan lists (k=2 reproduces AMC-RH exactly,
  monotonic JNE across k=2..5, edge cases at k=1 and very large k) — worth a pass to
  confirm coverage rather than assuming it from the file's existence.

---

## Documentation Map by Phase

```
Phase 1: Foundational Analysis
├── task_set_specification.md    (Worked example from AMC-RH)
├── safety_proof.md              (Proven properties)
└── proof_requirements.md        (Open questions)

Phase 2: Metrics & Complexity
├── metrics_objective.md         (Objective function, criteria)
└── complexity_analysis.md       (Time/space overhead analysis)

Phase 3: Optimal Mode Design
├── mode_optimization.md              (Plan: severity-lattice enumeration, single method)
└── ../research/mode_optimization.tex (Paper: landscape, related work, recommendation)

Phase 4: Drop Strategy Research
└── drop_strategy_research.md    (Drop strategies, exit policies)

Phase 5: Implementation & Validation
└── implementation_validation_plan.md  (Implementation plan, tests)
```

---

## Quick Reference

### Multi-Level Model Extension

| Level | Name | Trigger Condition | Behavior |
|-------|------|-------------------|----------|
| L_0 | Normal | None | All tasks run normally |
| L_1 | Degraded-1 | HI task reaches R_1 | Drop some LO tasks |
| L_2 | Degradated-2 | HI task reaches R_2 | Drop more LO tasks |
| ... | ... | ... | ... |
| L_{k-1} | Fully Degraded | HI task reaches R_trigger | Drop all LO tasks |

### Key Metrics

The primary metric is now **JNC** (Jobs Not Completed), not JNE — see
`metrics_objective.md` for why: JNE alone is blind to jobs abandoned via deadline
termination rather than at a level transition.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **JNC** | Exp − Comp (released but not completed) | Primary metric: LO jobs lost by any means, dropped or terminated |
| **JNE** | LO jobs dropped in degraded mode | Component of JNC: lost on abandonment at a level transition |
| **TiD** | Time fraction in degraded mode | System degradation duration |
| **WastedCPU** | Executed but abandoned LO work | CPU cycles wasted |
| **LevelTrans** | Count of level transitions (any direction) | Oscillation within the degraded levels, invisible to TiD/NiD |
| **ServiceRatio** | Comp / Exp = 1 − JNC/Exp | Fraction of LO service preserved |

### Objective Function

Φ = α(U)·JNC + β(U)·TiD + γ·WastedCPU + δ·LevelTrans

α, β adapt to utilisation U; γ, δ are fixed small weights. See `metrics_objective.md`
"Proposed Objective: Hybrid Approach" for why LevelTrans is included alongside TiD.

---

## Related Documentation

- **Task Set Model**: `../research/task_set_model.tex` and `../research/task_set_model.md`
- **Memory Entries**: `../../memory/` (auto-generated summaries)
