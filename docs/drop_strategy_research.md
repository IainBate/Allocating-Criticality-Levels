# Drop Strategy Research Plan

## Phase 4 Overview

This phase investigates how to optimally select which LO-criticality tasks to drop at each degradation level transition.

---

## Task 4.1: LO Task Assignment to Criticality Levels

### Objective
Determine the optimal strategy for assigning low-criticality tasks to degradation levels.

### Key Questions

#### Q1: Static vs Dynamic Assignment
| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Static at generation** | Assign each LO task to a "drop tier" when task set is created | Simple, no runtime overhead | May not adapt to current load |
| **Dynamic at transition** | Decide which tasks to drop based on current run-queue state | Adaptive to current conditions | Requires sorting/computation |

**Recommendation**: Hybrid approach - static priority ordering with dynamic threshold selection.

#### Q2: Assignment Criteria
Possible criteria for LO task ordering:

| Criterion | Formula | Interpretation |
|-----------|---------|----------------|
| **Priority** | Already assigned (deadline monotonic) | Lower index = higher priority, less likely to drop |
| **Utilization** | U_i = C_i(LO)/T_i | Higher util = more important, harder to drop |
| **Deadline tightness** | (D_i - R_i(LO))/T_i | Smaller slack = tighter deadline, more critical |
| **Combined** | α·priority + β·U_i + γ·tightness | Weighted combination |

#### Q3: Drop Decision Point
When entering level L_x, which LO tasks to drop?

**Option A**: Drop lowest-priority tasks (inherent in FPPS)
- Simple: just take first m from sorted queue
- No computation needed beyond maintaining priority order

**Option B**: Drop highest-utilization tasks
- More "fair" - spreads the burden
- Requires O(n log n) sort by utilization

**Option C**: Drop with slack-based selection
- Drop tasks that have excess slack (deadline - response time)
- Preserves tasks with tight deadlines

### Analysis Framework

```python
def evaluate_drop_strategy(tasksets, strategy, k):
    """Evaluate a drop strategy across an ensemble."""
    
    metrics = {
        'jne': [],
        'wasted_cpu': [],
        'service_ratio': [],
        'level_transitions': []
    }
    
    for ts in tasksets:
        result = simulate_with_drop_strategy(ts, k, strategy)
        
        metrics['jne'].append(result.jne)
        metrics['wasted_cpu'].append(result.wasted_cpu)
        metrics['service_ratio'].append(
            1 - result.jne / result.total_lo_releases
        )
        metrics['level_transitions'].append(count_transitions(result.trace))
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### Candidate Strategies

#### Strategy 1: Priority-Based (Baseline)
```python
# Already implemented in FPPS - lowest priority = first to drop
def priority_drop(state, count):
    # state.active is already sorted by priority
    return state.active[:count]
```

**Complexity**: O(1) per decision (after queue is sorted)

#### Strategy 2: Utilization-Based
```python
def utilization_drop(taskset, state, count):
    lo_tasks = [j for j in state.active if j.criticality == 'LO']
    # Sort by utilization descending (highest first)
    lo_tasks.sort(key=lambda j: taskset.C_lo[j.task_id]/taskset.T[j.task_id], reverse=True)
    return lo_tasks[:count]
```

**Complexity**: O(m log m) where m = active LO tasks

#### Strategy 3: Deadline Slack-Based
```python
def slack_drop(taskset, state, count):
    lo_tasks = [j for j in state.active if j.criticality == 'LO']
    
    # Computeslack for each job
    for job in lo_tasks:
        # Response time in current mode (approximate)
        response_time = compute_response_time(job, state.mode)
        job.slack = job.deadline - (state.time + response_time)
    
    # Drop jobs with smallest slack (most urgent to remove)
    lo_tasks.sort(key=lambda j: j.slack)
    return lo_tasks[:count]
```

**Complexity**: O(m log m) for sorting

#### Strategy 4: Proportional Reduction
```python
def proportional_drop(state, drop_fraction):
    """Drop a fraction of each task's active jobs."""
    dropped = []
    
    # Group by task
    tasks_with_jobs = group_by_task(state.active)
    
    for task_id, jobs in tasks_with_jobs.items():
        n_to_drop = int(len(jobs) * drop_fraction)
        dropped.extend(jobs[:n_to_drop])
    
    return dropped
```

**Complexity**: O(m) with hash map

#### Strategy 5: Hybrid Weighted
```python
def weighted_strategy(taskset, state, count, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Combined score = α·normalized_priority + β·normalized_utilization + γ·normalized_slack
    Lower score = more likely to drop
    """
    lo_tasks = [j for j in state.active if j.criticality == 'LO']
    
    priorities = np.array([taskset.priority[j.task_id] for j in lo_tasks])
    utils = np.array([taskset.C_lo[j.task_id]/taskset.T[j.task_id] for j in lo_tasks])
    slacks = np.array([compute_slack(j, state) for j in lo_tasks])
    
    # Normalize to [0, 1]
    p_norm = (priorities - priorities.min()) / (priorities.max() - priorities.min() + 1e-6)
    u_norm = (utils - utils.min()) / (utils.max() - utils.min() + 1e-6)
    s_norm = (slacks - slacks.min()) / (slacks.max() - slacks.min() + 1e-6)
    
    scores = alpha * p_norm + beta * u_norm + gamma * s_norm
    
    # Sort by score ascending (lowest score first to drop)
    indices = np.argsort(scores)
    return [lo_tasks[i] for i in indices[:count]]
```

### Experimental Comparison

**Test setup**: Run all strategies at k=3,4,5 across U ∈ {0.6, 0.7, 0.8, 0.9}

| Strategy | JNE ↓ | TiD ↓ | WastedCPU ↓ | Complexity |
|----------|-------|-------|-------------|------------|
| Priority | ? | ? | ? | O(1) |
| Utilization | ? | ? | ? | O(m log m) |
| Slack | ? | ? | ? | O(m log m) |
| Proportional | ? | ? | ? | O(m) |
| Hybrid (α=0.4) | ? | ? | ? | O(m log m) |

**Hypothesis**: Priority-based is near-optimal because FPPS already prioritizes tasks by deadline tightness.

---

## Task 4.2: Exit Strategy Research

### Objective
Determine optimal strategy for exiting degraded modes.

### Exit Strategy Options

#### Strategy A: Direct to L_0
```python
def should_exit_direct(state):
    """Exit immediately when trigger clears."""
    return not any(job.criticality == 'HI' and job.reaches_trigger() 
                   for job in state.active)
```

**Pros**: Fast recovery, minimal degradation time
**Cons**: May oscillate if HI behavior persists

#### Strategy B: Cascade Exit (One Level at a Time)
```python
def should_exit_cascade(state):
    """Exit one level at a time with hysteresis."""
    current_level = get_current_level()
    
    # Check if we can exit to level-1
    if current_level > 0:
        threshold = get_trigger_point(current_level - 1)
        return not any(job.criticality == 'HI' and job.reaches_trigger() 
                       for job in state.active and time_in_level > HYSTERESIS)
    
    # Normal mode exit (idle instant)
    return not state.active
```

**Pros**: Smoother transitions, less oscillation
**Cons**: Slower recovery to normal mode

#### Strategy C: Hysteresis-Based
```python
def should_exit_hysteresis(state):
    """Require trigger < threshold - hysteresis to exit."""
    if time_in_level[state.mode] < HYSTERESIS_THRESHOLD:
        return False  # Must stay in current level minimum time
    
    # For exiting to L_0, require no HI tasks at ANY trigger point
    for job in state.active:
        if job.criticality == 'HI':
            for level in range(1, get_current_level() + 1):
                if job.reaches_trigger(level):
                    return False
    
    return True
```

**Pros**: Reduces mode-switching overhead
**Cons**: Adds delay before recovery

#### Strategy D: Adaptive Exit (Prediction-Based)
```python
def should_exit_adaptive(state):
    """Exit based on expected future triggers."""
    
    # Estimate remaining HI behavior based on current state
    if estimate_remaining_hi_work(state) < MIN_WORK_THRESHOLD:
        return True
    
    return False

def estimate_remaining_hi_work(state):
    """Predict how much HI work remains."""
    total = 0
    for job in state.active:
        if job.criticality == 'HI':
            remaining_exec = job.remaining
            # Account for potential future HI behavior
            if job.executed < job.c_lo:  # Not yet triggered HI behavior
                remaining_exec += job.c_hi - job.c_lo  # Add buffer
            total += remaining_exec
    
    return total
```

**Pros**: Potentially optimal exit timing
**Cons**: Complex to implement, requires prediction model

### Hysteresis Parameter Tuning

**Key question**: What hysteresis value minimizes the objective function?

```python
def evaluate_hysteresis(tasksets, hysteresis_values):
    """Compare different hysteresis values."""
    
    results = {}
    
    for h in hysteresis_values:
        global HYSTERESIS_THRESHOLD
        HYSTERESIS_THRESHOLD = h
        
        metrics = {'jne': [], 'tid': 'level_transitions': []}
        
        for ts in tasksets:
            result = simulate(ts, exit_strategy='hysteresis')
            
            metrics['jne'].append(result.jne)
            metrics['tid'].append(result.tid)
            metrics['level_transitions'].append(count_transitions(result.trace))
        
        results[h] = {k: np.mean(v) for k, v in metrics.items()}
    
    return results
```

**Expected finding**: Small hysteresis (5-10% of degradation duration) reduces oscillations without significant TiD increase.

---

## Implementation Plan

### Phase 4.1: Drop Strategy Implementation

```python
# In amc_tasksim/simulation/engine.py

class ModeChangeProtocol(ABC):
    # ... existing methods ...
    
    @abstractmethod
    def select_jobs_to_drop(self, state: SchedulerState, count: int) -> list[Job]:
        """Select which jobs to drop when entering a degraded level."""
        ...
```

### Phase 4.2: Exit Strategy Implementation

```python
class HysteresisProtocol(ModeChangeProtocol):
    """Protocol with hysteresis-based exit decision."""
    
    def __init__(self, r_lo, hysteresis_factor=0.1):
        super().__init__(r_lo)
        self.hysteresis = hysteresis_factor
    
    def should_exit(self, state: SchedulerState) -> bool:
        if time_in_level(state.mode) < self.hysteresis * total_degradation_time:
            return False
        
        # Check trigger conditions...
```

---

## Validation Framework

### Test Case 1: Monotonicity
**Requirement**: More levels should never increase JNE for same task set.

```python
def test_monotonic_jne(taskset):
    """Verify JNE decreases or stays same as k increases."""
    
    results = {}
    for k in [2, 3, 4, 5]:
        r = simulate(taskset, k=k)
        results[k] = r.jne
    
    # Check monotonicity
    assert all(results[k] <= results[k-1] for k in [3, 4, 5])
```

### Test Case 2: Objective Improvement
**Requirement**: Optimal configuration should improve objective over baseline.

```python
def test_objective_improvement(tasksets):
    """Verify optimal k > 2 improves Φ over k = 2."""
    
    phi_k2 = evaluate_objective(tasksets, k=2, strategy='priority')
    phi_k3_opt = evaluate_objective(tasksets, k=3, strategy='optimal')
    
    assert phi_k3_opt < phi_k2 * 0.95  # At least 5% improvement
```

---

## Deliverables

1. **docs/drop_strategy_research.md** - This document (completed)
2. **Implementation** of k drop strategies in simulation engine
3. **docs/exit_strategy_analysis.md** - Hysteresis and exit strategy analysis
4. **Results table**: Best strategy per utilisation level

---

## References

- **AMC-RH Section IV-B**: Current two-mode exit strategies
- **Liu & Layland (1973)**: Deadline monotonic priority assignment
- **Stankovic et al. (2000)**: Quality of service in real-time systems
