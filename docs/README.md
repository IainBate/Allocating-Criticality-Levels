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
| `drop_strategy_research.md` | **Phase 4 plan** - Strategies for selecting which LO tasks to drop and when to exit degraded modes |

**Key questions addressed**:
- Static vs dynamic task assignment to levels
- Priority-based, utilization-based, deadline-based, proportional, and hybrid drop strategies
- Exit strategy options (direct, cascade, hysteresis, adaptive)

---

## Phase 5: Implementation & Validation

| File | Description |
|------|-------------|
| `implementation_validation_plan.md` | **Phase 5 plan** - Protocol interface extension, validation tests, experiment framework |

**Key components**:
- Extended `MultiLevelProtocol` interface with k-level support
- Correctness, monotonicity, and consistency test suites
- Sweep framework for experiments across utilisation levels

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

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **JNE** | LO jobs dropped in degraded mode | Lost low-criticality work |
| **TiD** | Time fraction in degraded mode | System degradation duration |
| **WastedCPU** | Executed but abandoned LO work | CPU cycles wasted |
| **ServiceRatio** | (LO completions) / (LO releases) | Fraction of LO service preserved |

### Objective Function

Φ = α·JNE + β·TiD + γ·WastedCPU

Weights adapt to utilisation U for balanced optimization.

---

## Related Documentation

- **Task Set Model**: `../research/task_set_model.tex` and `../research/task_set_model.md`
- **Memory Entries**: `../../memory/` (auto-generated summaries)
