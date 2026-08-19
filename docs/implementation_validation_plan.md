# Implementation & Validation Plan

## Phase 5 Overview

This phase implements the multi-level scheduling protocol and validates it against the two-mode baseline.

---

## Task 5.1: Core Multi-Level Protocol

### 5.1.1: Extended ModeChangeProtocol Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class LevelState:
    """Current degradation level and transition state."""
    current_level: int  # L_0 = 0 (normal), L_1 = 1, ..., L_{k-1}
    entry_time: int     # When we entered current level
    time_in_level: int  # Duration in current level


class MultiLevelProtocol(ABC):
    """Entry and exit rules for multi-level degradation scheduling."""
    
    @abstractmethod
    def get_degradation_level(self, state: SchedulerState) -> int:
        """Return current degradation level (0 = normal)."""
        ...
    
    @abstractmethod
    def entry_time(self, state: SchedulerState, target_level: int) -> Optional[int]:
        """When to enter the specified degradation level."""
        ...
    
    @abstractmethod
    def should_exit_level(self, state: SchedulerState, current_level: int) -> bool:
        """Whether we should exit current level at state.time."""
        ...
    
    @abstractmethod
    def select_jobs_to_drop(self, state: SchedulerState, 
                           target_level: int) -> List[Job]:
        """Select jobs to drop when entering a degraded level."""
        ...
```

### 5.1.2: Multi-Level Engine Extension

```python
def simulate_multi_level(
    taskset: TaskSet,
    k: int,                        # Number of degradation levels
    protocol: MultiLevelProtocol,
    duration: int = 10**6,
    seed: Optional[int] = None,
    fp: float = 1e-4,
) -> SimulationResult:
    """Simulate under multi-level scheduling."""
    
    rng = np.random.default_rng(seed)
    state = SchedulerState()
    result = SimulationResult(duration=duration)
    
    # Track per-job degradation level
    job_level: Dict[Job, int] = {}
    
    degraded_start = -1  # Start time of current degradation period
    
    while state.time < duration:
        now = state.time
        
        # Check for mode transitions
        current_level = protocol.get_degradation_level(state)
        
        # Can we exit current level?
        if current_level > 0 and protocol.should_exit_level(state, current_level):
            if current_level == k - 1:
                # Fully degraded -> normal
                state.mode = "normal"
                result.degraded_ticks += now - degraded_start
                degraded_start = -1
            else:
                # Downgrade one level
                pass  # Implement cascade exit if desired
        
        # Can we enter a deeper level?
        next_level = current_level + 1
        if state.mode == "normal" and next_level < k:
            entry_time = protocol.entry_time(state, next_level)
            if entry_time is not None and entry_time <= now:
                jobs_to_drop = protocol.select_jobs_to_drop(state, next_level)
                
                # Abandon selected jobs
                for job in jobs_to_drop:
                    state.active.remove(job)
                    result.jne += 1
                
                if current_level == 0:
                    degraded_start = now
                
                state.mode = f"degraded_{next_level}"
                result.nid += 1
        
        # ... rest of simulation engine (releases, completions, etc.) ...
    
    return result
```

### 5.1.3: Implementation Checklist

- [ ] **5.1a**: Extend `SchedulerState` to track current degradation level
- [ ] **5.1b**: Implement `MultiLevelProtocol` interface with abstract methods
- [ ] **5.1c**: Add `get_degradation_level()` method to return current level
- [ ] **5.1d**: Add `should_exit_level(level)` method for exit decisions
- [ ] **5.1e**: Track per-job degradation level state
- [ ] **5.1f**: Implement drop decision logic for each level

---

## Task 5.2: Validation Suite

### 5.2.1: Correctness Tests

#### Test 5.2.1a: Reproduce Two-Mode Results
```python
def test_two_level_reproduces_amc_rh():
    """When k=2, results should match AMC-RH exactly."""
    
    tasksets = generate_qualifying_tasksets(n=10, U=0.7)
    
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        
        # Run two-level (AMC-RH)
        result_2level = simulate_multi_level(
            ts, k=2, protocol=TwoLevelProtocol(r_lo), duration=10**5
        )
        
        result_amc_rh = simulate(
            ts, mode_protocol=AMC_RH(r_lo), duration=10**5
        )
        
        # Compare metrics (allow small numerical differences)
        assert abs(result_2level.jne - result_amc_rh.jne) < 5
        assert abs(result_2level.tid - result_amc_rh.tid) < 0.01
        assert abs(result_2level.nid - result_amc_rh.nid) <= 1
```

#### Test 5.2.1b: HI Tasks Always Meet Deadlines
```python
def test_hi_tasks_meet_deadlines():
    """HI-criticality tasks should never miss deadlines."""
    
    tasksets = generate_qualifying_tasksets(n=20, U=0.8)
    
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        
        result = simulate_multi_level(
            ts, k=4, protocol=MultiLevelProtocol(r_lo), duration=10**6
        )
        
        assert result.hdm == 0, f"HI deadline misses: {result.hdm}"
```

#### Test 5.2.1c: Monotonic JNE
```python
def test_monotonic_jne_across_levels():
    """More levels should never increase JNE for same task set."""
    
    ts = generate_qualifying_tasksets(n=10, U=0.7)[0]
    r_lo = amc_rtb(ts).r_lo
    
    jne_values = {}
    for k in [2, 3, 4, 5]:
        result = simulate_multi_level(
            ts, k=k, protocol=MultiLevelProtocol(r_lo), duration=10**5
        )
        jne_values[k] = result.jne
    
    # Verify monotonicity: JNE(k) <= JNE(k-1)
    for k in [3, 4, 5]:
        assert jne_values[k] <= jne_values[k-1], \
            f"JNE not monotonic: {jne_values[k]} > {jne_values[k-1]}"
```

### 5.2.2: Consistency Tests

#### Test 5.2.2a: Objective Function Improvement
```python
def test_objective_improvement():
    """Multi-level should improve objective over two-level baseline."""
    
    tasksets = generate_qualifying_tasksets(n=100, U=0.7)
    
    phi_two_level = []
    phi_multi_level = []
    
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        
        result_2 = simulate_multi_level(
            ts, k=2, protocol=TwoLevelProtocol(r_lo), duration=10**5
        )
        
        result_k = simulate_multi_level(
            ts, k=4, protocol=MultiLevelProtocol(r_lo), duration=10**5
        )
        
        phi_2 = compute_objective(result_2)
        phi_k = compute_objective(result_k)
        
        phi_two_level.append(phi_2)
        phi_multi_level.append(phi_k)
    
    # Multi-level should have lower expected objective
    mean_phi_2 = np.mean(phi_two_level)
    mean_phi_k = np.mean(phi_multi_level)
    
    assert mean_phi_k < mean_phi_2 * 0.95, \
        f"Multi-level objective {mean_phi_k} not better than two-level {mean_phi_2}"
```

#### Test 5.2.2b: Service Preservation
```python
def test_service_preservation():
    """Service ratio should increase or stay same with more levels."""
    
    tasksets = generate_qualifying_tasksets(n=50, U=0.7)
    
    for ts in tasksets:
        r_lo = amc_rtb(ts).r_lo
        
        result_2 = simulate_multi_level(
            ts, k=2, protocol=TwoLevelProtocol(r_lo), duration=10**5
        )
        
        result_k = simulate_multi_level(
            ts, k=4, protocol=MultiLevelProtocol(r_lo), duration=10**5
        )
        
        service_2 = 1 - result_2.jne / result_2.total_lo_releases
        service_k = 1 - result_k.jne / result_k.total_lo_releases
        
        assert service_k >= service_2 * 0.95, \
            f"Service not preserved: {service_k} < {service_2}"
```

### 5.2.3: Edge Case Tests

#### Test 5.2.3a: k=1 (Single Mode)
```python
def test_single_mode_equivalent_to_fpps():
    """k=1 should behave like single-criticality FPPS."""
    
    ts = generate_qualifying_tasksets(n=5, U=0.6)[0]
    
    result = simulate_multi_level(
        ts, k=1, protocol=MultiLevelProtocol([]), duration=10**4
    )
    
    # No jobs should be abandoned (no degradation)
    assert result.jne == 0
    
    # All HI tasks should meet deadlines
    assert result.hdm == 0
```

#### Test 5.2.3b: Very Large k
```python
def test_very_large_k():
    """With very large k, we should approach ideal proportional dropping."""
    
    ts = generate_qualifying_tasksets(n=10, U=0.9)[0]
    
    # Run with very large k
    result = simulate_multi_level(
        ts, k=20, protocol=MultiLevelProtocol(r_lo), duration=10**5
    )
    
    # Should not crash or produce invalid results
    assert result.jne >= 0
    assert result.hdm == 0
```

#### Test 5.2.3c: Extreme Utilisation
```python
def test_extreme_utilisation():
    """Test at very low and very high utilisation."""
    
    for U in [0.3, 0.95]:
        ts = generate_qualifying_tasksets(n=10, U=U)[0]
        
        result = simulate_multi_level(
            ts, k=4, protocol=MultiLevelProtocol(r_lo), duration=10**5
        )
        
        # Should handle without errors
        assert hasattr(result, 'jne')
```

---

## Task 5.3: Experiment Framework

### 5.3.1: Parameter Sweep Configuration

```python
@dataclass
class MultiLevelExperiment:
    """Configuration for multi-level experiments."""
    
    U_values: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    k_values: Tuple[int, ...] = (2, 3, 4, 5)
    drop_strategies: Tuple[str, ...] = (
        'priority', 'utilization', 'deadline', 'proportional'
    )
    exit_strategies: Tuple[str, ...] = ('direct', 'hysteresis')
    
    n_replicates: int = 100
    duration_jobs: int = 10**5
    
    output_path: str = "results/multi_level_sweep.parquet"
```

### 5.3.2: Sweep Implementation

```python
def run_multi_level_sweep(experiment: MultiLevelExperiment, verbose=True):
    """Run the multi-level experiment sweep."""
    
    results = []
    
    for U in experiment.U_values:
        # Generate qualifying task sets
        tasksets, attempts = build_population(
            n_replicates=experiment.n_replicates,
            U=U,
            seed=42 + int(U * 1000)
        )
        
        if verbose:
            print(f"U={U:.2f}: {len(tasksets)} qualifying task sets")
        
        for ts in tasksets:
            r_lo = amc_rtb(ts).r_lo
            
            # Run all combinations
            for k in experiment.k_values:
                for drop_strat in experiment.drop_strategies:
                    for exit_strat in experiment.exit_strategies:
                        result = simulate_multi_level(
                            ts,
                            k=k,
                            protocol=MultiLevelProtocol(
                                r_lo=r_lo,
                                drop_strategy=drop_strat,
                                exit_strategy=exit_strat
                            ),
                            duration=experiment.duration_jobs
                        )
                        
                        # Compute objective value
                        phi = compute_objective(result)
                        
                        results.append({
                            'U': U,
                            'k': k,
                            'drop_strategy': drop_strat,
                            'exit_strategy': exit_strat,
                            'jne': result.jne,
                            'tid': result.tid,
                            'wasted_cpu': result.wasted_cpu,
                            'service_ratio': 1 - result.jne / result.total_lo_releases,
                            'phi': phi,
                        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_parquet(experiment.output_path, index=False)
    
    if verbose:
        print(f"\nResults saved to {experiment.output_path}")
    
    return df
```

### 5.3.3: Analysis Pipeline

```python
def analyze_multi_level_results(df: pd.DataFrame):
    """Analyze sweep results and generate statistics."""
    
    analysis = {}
    
    # Baseline comparison (k=2, priority drop)
    baseline = df[
        (df['k'] == 2) & 
        (df['drop_strategy'] == 'priority') &
        (df['exit_strategy'] == 'direct')
    ].groupby('U').mean(numeric_only=True)
    
    # For each k > 2, compute improvement over baseline
    for k in [3, 4, 5]:
        results = df[df['k'] == k]
        
        for U in experiment.U_values:
            subset = results[results['U'] == U]
            
            best_config = subset.loc[subset['phi'].idxmin()]
            
            analysis[f'U={U},k={k}'] = {
                'best_drop_strategy': best_config.drop_strategy,
                'best_exit_strategy': best_config.exit_strategy,
                'best_phi': best_config.phi,
                'jne_improvement': (baseline.loc[U, 'jne'] - best_config.jne) / baseline.loc[U, 'jne'],
                'service_improvement': best_config.service_ratio - baseline.loc[U, 'service_ratio'],
            }
    
    return analysis
```

### 5.3.4: Expected Output Format

```python
{
    "U=0.6,k=3": {
        "best_drop_strategy": "priority",
        "best_exit_strategy": "hysteresis",
        "best_phi": 12.5,
        "jne_improvement": 0.12,  # 12% better than baseline
        "service_improvement": 0.08,
    },
    "U=0.6,k=4": { ... },
    "U=0.6,k=5": { ... },
    # ... for all U values
}
```

---

## Implementation Timeline

### Week 1: Protocol Implementation
- [ ] Extend `SchedulerState` with level tracking
- [ ] Implement `MultiLevelProtocol` interface
- [ ] Add drop decision logic
- [ ] Integrate with simulation engine

### Week 2: Validation Tests
- [ ] Write correctness tests (AMC-RH reproduction)
- [ ] Write monotonicity tests
- [ ] Write edge case tests
- [ ] Run validation suite

### Week 3-4: Experiment Runs
- [ ] Implement sweep framework
- [ ] Run experiments at U ∈ {0.6, 0.7, 0.8, 0.9}
- [ ] Analyze results
- [ ] Generate reports

---

## Deliverables

1. **`amc_tasksim/simulation/multi_level.py`** - Multi-level protocol implementation
2. **`docs/validation_results.md`** - Test results and validation summary
3. **`results/multi_level_sweep.parquet`** - Experiment results
4. **`docs/experiment_analysis.md`** - Analysis of optimal configurations

---

## References

- **AMC-RH Section V**: Experimental evaluation methodology
- **DRS paper**: Task set generation distribution
- **Pandas documentation**: Data analysis and parquet I/O
