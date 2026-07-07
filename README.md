# amc-tasksim

Adaptive Mixed-Criticality (AMC) task-set generation and simulation for real-time scheduling research.

## Purpose

Generate large-scale synthetic AMC task sets across a utilisation sweep and a
HI-criticality failure-probability sweep (FP = 1/N, N ranging over several
orders of magnitude), then simulate each for a fixed duration to empirically
measure degraded-mode behaviour (NiD, TiD, JNE, LDM, HDM).

## Quick start

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run a quick sweep (smoke test)
uv run amc_tasksim --quick
```

## Directory layout

```
amc_tasksim/
  generation/       # DRS algorithm + task-set generator
  scheduling/       # priority assignment + analytic schedulability tests
  simulation/       # event-driven simulator core
  experiments/      # sweep orchestration + result storage
  analysis/         # plotting/summary statistics
tests/              # pytest suite mirroring the package structure
docs/               # reference PDFs (DRS, RTSS 2020; AMC-RH, RTAS 2022; AMC, RTNS 2022)
results/            # experiment output (gitignored)
```

## Phased build

This project is structured in 9 phases (see `initial_specification`):

1. Project scaffold
2. DRS core algorithm
3. Task-set generator
4. Priority assignment & analytic schedulability filter
5. Event-driven simulator core
6. Validation suite
7. Experiment orchestration (the sweep)
8. Analysis & plotting
9. Pluggable scheduler variants: AMC-RH / AMC-RA (optional extension)
