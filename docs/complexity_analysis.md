# Complexity Analysis for Multi-Level Scheduling

## Overview

This document analyzes the computational overhead introduced by multi-level degradation scheduling relative to the base AMC-RH protocol.

## Baseline: AMC-RH Complexity

### Per-Event Processing (AMC-RH)

| Event Type | Time Complexity | Space Overhead |
|------------|-----------------|----------------|
| Job release | O(1) | +1 job in queue |
| Job completion | O(1) | -1 job from queue |
| Deadline miss | O(m) to find expired | 0 (in-place removal) |
| Mode entry check | O(n) to scan active jobs | 0 |
| Mode exit check | O(n) to scan active jobs | 0 |

Where $n$ = number of active tasks, $m$ = number of expired deadlines.

### Space Complexity
- Run queue: O(active jobs)
- Busy period start times: O(n) for all tasks

## Multi-Level Overhead Analysis

### Data Structure Extensions

| Extension | Size per Job | Total Overhead |
|-----------|--------------|----------------|
| Current degradation level | 1 byte | O(n) |
| Drop priority (sorted list index) | O(1) | O(n) |
| Level exit timer (if hysteresis) | O(1) | O(n) |

**Total space overhead**: O(n) - linear in task count

### Event Processing Complexity

#### 1. Job Release

**Normal mode ($L_0$)**:
- Same as baseline: O(1) to insert into priority queue
- No level tracking needed

**Degraded levels ($L_x, x > 0$)**:
- Insert job at priority position: O(active jobs) worst case
- Check if job should be dropped immediately: O(1) with precomputed thresholds

**Complexity**: O(m) where m = active jobs in queue (same as baseline)

#### 2. Mode Entry Decision

**Baseline (AMC-RH)**:
```python
# Scan all active HI jobs for trigger condition
for job in active_jobs:
    if is_HI(job) and job.reached_R_lo():
        return True
```
- **Time**: O(n) where n = number of active tasks
- **Space**: O(1)

**Multi-Level Extension**:
```python
# Find minimum trigger among all levels
best_trigger = infinity
for level in [L_1, ..., L_{k-1}]:
    if job.reached_R_level(level):
        best_trigger = min(best_trigger, R_level)
return best_trigger
```

- **Time**: O(k × n) where k = number of degradation levels
- Since $k$ is typically small (2-4), this is effectively O(n)

#### 3. Drop Decision at Level Transition

When entering level $L_x$, we must decide which LO tasks to drop.

**Option A: Priority-based (O(1) per drop)**
```python
# Already sorted by priority, just take first m
drop_count = compute_drop_count(level)
for i in range(drop_count):
    drop_job(queue[i])  # O(1) each
```
- **Time**: O(m) where m = jobs to drop
- **Space**: O(1)

**Option B: Deadline-based (requires sorting)**
```python
# Sort LO tasks by deadline
lo_jobs = [j for j in active if is_LO(j)]
lo_jobs.sort(key=lambda j: j.deadline)  # O(m log m)
drop_count = compute_drop_count(level)
for i in range(drop_count):
    drop_job(lo_jobs[i])
```
- **Time**: O(m log m) for sorting + O(m) for dropping
- **Space**: O(m) for temporary list

**Option C: Proportional drop (O(n))**
```python
# Drop fraction f of each task's active jobs
f = compute_drop_fraction(level)
for job in active_LO_jobs:
    if random() < f:
        drop_job(job)
```
- **Time**: O(m) where m = active LO jobs
- **Space**: O(1)

**Recommendation**: Priority-based (Option A) since FPPS already maintains queue order.

#### 4. Exit Decision

**Baseline**:
```python
# Check for idle instant
return len(active_jobs) == 0
```
- **Time**: O(1)

**Multi-Level with Hysteresis**:
```python
# Check if we've been in current level long enough
if time_in_level > hysteresis_threshold:
    return len(active_HI_jobs) == 0 or all_jobs_under_trigger(level)
return False
```
- **Time**: O(n) to scan active HI jobs
- **Space**: O(1) for timer state

#### 5. Busy Period Start Time Inheritance

**Baseline (AMC-RH Appendix B)**:
```python
# O(1) per release - inherit from previous job in queue
job.busy_start = current_time if queue_empty else queue[-1].busy_start
```

**Multi-Level Extension**:
- No change needed - busy period tracking is independent of degradation level

- **Time**: O(1)
- **Space**: O(1) per job (already present)

## Summary Table

| Operation | Baseline | Multi-Level | Overhead Factor |
|-----------|----------|-------------|-----------------|
| Job release | O(m) | O(m) | 1× |
| Job completion | O(1) | O(1) | 1× |
| Mode entry trigger check | O(n) | O(k × n) | k× (k typically 2-4) |
| Level transition drop decision | - | O(m log m) worst case | depends on strategy |
| Exit decision | O(1) | O(n) with hysteresis | variable |

## Practical Complexity Analysis

### Worst Case
- **Trigger computation**: $O(kn)$ where $k \ll n$ (typically $k \leq 4$, $n = 20$)
- **Drop decision**: $O(m \log m)$ for deadline-based, $O(m)$ for priority-based
- **Exit check**: $O(n)$ with hysteresis

### Average Case (under typical load)
With $\sim 10$ active jobs and $k = 3$:
- Trigger check: $3 \times 10 = 30$ comparisons vs. $10$ baseline
- **Overhead**: ~20 extra comparisons per event - negligible on modern CPUs

## Space Complexity

| Component | Baseline | Multi-Level | Increase |
|-----------|----------|-------------|----------|
| Run queue | O(active) | O(active) | 0 |
| Priority array | O(n) | O(n) | 0 |
| Level state per job | - | O(1) per job | +O(n) bytes |
| Drop priority cache | - | O(m) temporary | +O(n) temporary |

**Total space overhead**: Linear in task count, typically < 1 KB for $n = 20$.

## Comparison to Current Implementation

### Current AMC-RH Overhead
The current implementation already tracks:
- Per-job busy period start times (Appendix B)
- HI-criticality behavior detection

### Multi-Level Additions
1. **Current degradation level**: 1 byte per active job
2. **Level exit timer** (if hysteresis): 4 bytes per level state
3. **Drop decision cache** (optional optimization): O(n)

**Estimated overhead for typical system**:
- $n = 20$ tasks × 5 extra bytes = 100 bytes
- Negligible compared to job structures (~64 bytes each)

## Algorithmic Optimizations

### 1. Early Termination in Trigger Check
```python
# Stop at first level that triggers
for level in levels:
    if any(job.reaches_trigger(level) for job in active_HI):
        return get_level_entry_time(level)
```

### 2. Incremental Drop Priority Updates
Instead of re-sorting on each transition:
- Maintain sorted order incrementally as jobs complete
- Only update priorities when jobs are dropped

### 3. Bitmask Level Tracking
For small $k$, use bitmasks for O(1) level membership tests:
```python
level_flags[job] = (1 << current_level)
can_drop_job(job, target_level) = ((level_flags[job] >> target_level) & 1) == 0
```

## Conclusion

The multi-level scheduling overhead is:
- **Time**: Linear in task count with small constant factor (k × n instead of n)
- **Space**: Linear in task count with tiny constant (< 5 bytes per job)
- **Practical impact**: Negligible for typical systems ($n \leq 20$, $k \leq 4$)

**Recommendation**: Implement with priority-based drop strategy to achieve O(m) worst case, matching the baseline complexity class.
