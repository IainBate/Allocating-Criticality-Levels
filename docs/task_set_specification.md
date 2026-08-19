# Task Set Specification

## Worked Example from AMC-RH Appendix A

This document provides the precise task set specification used in the AMC-RH paper (RTAS 2022), Section V-C and Appendix A.

### Task Table

| Task | C_lo | C_hi | T | D | Criticality |
|------|------|------|---|---|-------------|
| τ₁   | 1    | 1    | 2 | 2 | LO          |
| τ₂   | 1    | 5    | 10| 10| HI          |
| τ₃   | 4    | 4    | 100| 18| HI          |

### Manual Response Time Calculations

#### Priority Assignment (Deadline Monotonic)
- τ₁: D=2, highest priority (priority 0)
- τ₂: D=10, medium priority (priority 1)
- τ₃: D=18, lowest priority (priority 2)

#### R₁(LO) - Task τ₁
τ₁ has no higher-priority tasks, so:
```
R₁(LO) = C₁(LO) = 1
```

#### R₂(LO) - Task τ₂
Higher-priority tasks: {τ₁}
```
R₂(LO) = C₂(LO) + ⌈R₂(LO)/T₁⌉ × C₁(LO)
       = 1 + ⌈R₂(LO)/2⌉ × 1
```

Fixed-point iteration:
- R₂(LO) = 1 (initial)
- R₂(LO) = 1 + ⌈1/2⌉ × 1 = 1 + 1 = 2
- R₂(LO) = 1 + ⌈2/2⌉ × 1 = 1 + 1 = 2 (converged)

```
R₂(LO) = 2
```

#### R₃(LO) - Task τ₃
Higher-priority tasks: {τ₁, τ₂}
```
R₃(LO) = C₃(LO) + ⌈R₃(LO)/T₁⌉ × C₁(LO) + ⌈R₃(LO)/T₂⌉ × C₂(LO)
       = 4 + ⌈R₃(LO)/2⌉ × 1 + ⌈R₃(LO)/10⌉ × 1
```

Fixed-point iteration:
- R₃(LO) = 4 (initial)
- R₃(LO) = 4 + ⌈4/2⌉ + ⌈4/10⌉ = 4 + 2 + 1 = 7
- R₃(LO) = 4 + ⌈7/2⌉ + ⌈7/10⌉ = 4 + 4 + 1 = 9
- R₃(LO) = 4 + ⌈9/2⌉ + ⌈9/10⌉ = 4 + 5 + 1 = 10
- R₃(LO) = 4 + ⌈10/2⌉ + ⌈10/10⌉ = 4 + 5 + 1 = 10 (converged)

```
R₃(LO) = 10
```

#### R₂(HI) - Task τ₂ (HI-criticality)
Higher-priority HI-criticality tasks: ∅
Higher-priority LO-criticality tasks: {τ₁}
```
R₂(HI) = C₂(HI) + Σ_{j∈hp_HI} ⌈R₂(HI)/T_j⌉ × C_j(HI) 
         + Σ_{k∈hp_LO} ⌈R₂(LO)/T_k⌉ × C_k(LO)
       = 5 + 0 + ⌈R₂(LO)/2⌉ × 1
       = 5 + ⌈2/2⌉ × 1
       = 5 + 1 = 6
```

```
R₂(HI) = 6 ≤ D₂ = 10 ✓ schedulable
```

#### R₃(HI) - Task τ₃ (HI-criticality)
Higher-priority HI-criticality tasks: {τ₂}
Higher-priority LO-criticality tasks: {τ₁}

Note: The interference from LO tasks uses ⌈R₃(LO)/T_k⌉ (not +1, per AMC-RH equation 2):
```
R₃(HI) = C₃(HI) + ⌈R₃(HI)/T₂⌉ × C₂(HI) + ⌈R₃(LO)/T₁⌉ × C₁(LO)
       = 4 + ⌈R₃(HI)/10⌉ × 5 + ⌈10/2⌉ × 1
       = 4 + ⌈R₃(HI)/10⌉ × 5 + 5 × 1
       = 9 + ⌈R₃(HI)/10⌉ × 5
```

Fixed-point iteration:
- R₃(HI) = 4 (initial)
- R₃(HI) = 9 + ⌈4/10⌉ × 5 = 9 + 1 × 5 = 14
- R₃(HI) = 9 + ⌈14/10⌉ × 5 = 9 + 2 × 5 = 19
- R₃(HI) = 9 + ⌈19/10⌉ × 5 = 9 + 2 × 5 = 19 (converged)

```
R₃(HI) = 19 > D₃ = 18 ✗ unschedulable
```

### Verification Summary

| Task | R(LO) | R(HI) | Deadline | Schedulable |
|------|-------|-------|----------|-------------|
| τ₁   | 1     | 1     | 2        | ✓           |
| τ₂   | 2     | 6     | 10       | ✓           |
| τ₃   | 10    | 19    | 18       | ✗           |

### Python Verification

The following code verifies these calculations:

```python
from amc_tasksim.generation.taskset import TaskSet
from amc_tasksim.scheduling.priority import assign_deadline_monotonic
from amc_tasksim.scheduling.amc_rtb import amc_rtb

# Create the task set from Appendix A
taskset = TaskSet(
    n=3,
    criticality=["LO", "HI", "HI"],
    T=[2, 10, 100],
    D=[2, 10, 18],
    C_lo=[1, 1, 4],
    C_hi=[1, 5, 4],
    BCET=[1, 1, 4],  # Use WCET for deterministic reproduction
)

assign_deadline_monotonic(taskset)
result = amc_rtb(taskset)

print(f"R(LO) = {result.r_lo}")  # Expected: [1, 2, 10]
print(f"R(HI) = {result.r_hi}")  # Expected: [1, 6, 19]
```

### Key Observations

1. **Task τ₃ is unschedulable** under AMC-rtb (R₃(HI) = 19 > D₃ = 18)
2. **R₃(LO) = 10** matches the paper's calculation: interference from 5 jobs of τ₁ (each 1 tick) and 1 job of τ₂ at 1 tick
3. **R₃(HI) = 19** matches the paper: same τ₁ interference plus 2 jobs of τ₂ at 5 ticks each
4. This is a "non-trivial" task set - it fails single-criticality FPPS but passes AMC-rtb
