---
name: multi-level-complexity
description: Time and space complexity of multi-level scheduling
metadata:
  type: reference
---

# Multi-Level Scheduling Complexity Analysis

## Baseline (AMC-RH) Per-Event Processing

| Event Type | Time | Space |
|------------|------|-------|
| Job release | O(1) | +1 job in queue |
| Job completion | O(1) | -1 job from queue |
| Deadline miss | O(m) to find expired | 0 (in-place removal) |
| Mode entry check | O(n) to scan active jobs | 0 |
| Mode exit check | O(n) to scan active jobs | 0 |

Where $n$ = number of active tasks, $m$ = number of expired deadlines.

## Multi-Level Additions

### Time Complexity

| Operation | Baseline | Multi-Level | Factor |
|-----------|----------|-------------|--------|
| Trigger check | O(n) | O(k × n) | k× (k typically 2-4) |
| Drop decision | - | O(m) priority-based | - |
| Exit decision | O(1) | O(n) with hysteresis | variable |

### Space Complexity

| Component | Baseline | Multi-Level | Increase |
|-----------|----------|-------------|----------|
| Run queue | O(active) | O(active) | 0 |
| Level state per job | - | O(1) per job | +O(n) bytes |
| Drop priority cache | - | O(m) temporary | +O(n) temporary |

**Total space overhead**: Linear in task count, typically < 1 KB for $n = 20$.

## Worst Case Analysis

- **Trigger computation**: $O(kn)$ where $k \ll n$ (typically $k \leq 4$, $n = 20$)
- **Drop decision**: $O(m \log m)$ for deadline-based, $O(m)$ for priority-based
- **Exit check**: $O(n)$ with hysteresis

## Average Case (typical load)

With $\sim 10$ active jobs and $k = 3$:
- Trigger check: $3 \times 10 = 30$ comparisons vs. $10$ baseline
- **Overhead**: ~20 extra comparisons per event - negligible on modern CPUs

## Algorithmic Optimizations

1. **Early termination in trigger check**: Stop at first level that triggers
2. **Incremental drop priority updates**: Maintain sorted order as jobs complete
3. **Bitmask level tracking**: O(1) level membership tests for small k using bitmasks

## Practical Impact

The multi-level scheduling overhead is:
- **Time**: Linear in task count with small constant factor (k × n instead of n)
- **Space**: Linear in task count with tiny constant (< 5 bytes per job)
- **Practical impact**: Negligible for typical systems ($n \leq 20$, $k \leq 4$)

## Recommendation

Implement with priority-based drop strategy to achieve O(m) worst case, matching the baseline complexity class.
