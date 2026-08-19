# Task Set Model for Multi-Mode Mixed-Criticality Scheduling

This document presents the formal task set model used in our research on multi-mode mixed-criticality (MC) scheduling.

## Overview

The model extends traditional two-mode Adaptive Mixed-Criticality (AMC) scheduling to support **multiple criticality levels**, enabling graded degradation rather than binary normal/degraded transitions.

### Key Extensions from Two-Mode AMC

| Aspect | Two-Mode AMC | Multi-Mode AMC |
|--------|-------------|----------------|
| Degradation Levels | 2 (normal, degraded) | $k \geq 2$ levels ($L_0$ through $L_{k-1}$) |
| Trigger Points | 1 ($R_{\mathrm{trigger}}$) | $k-1$ points ($R_1, R_2, \dots, R_{k-1}$) |
| LO Task Handling | All dropped at once | Dropped incrementally per level |
| Exit Strategy | Idle instant (original) or RTA-based (AMC-RH/RA) | Per-level exit conditions |

---

## Formal Model

### Task Properties

Each task $\tau_i$ is characterized by:

$$
\begin{aligned}
T_i &\in \mathbb{N}^+ & \text{(Period)} \\
D_i &\in \mathbb{N}^+ & \text{(Deadline, } D_i = T_i \text{ for implicit deadlines)} \\
C_i^{\mathrm{lo}} &\in \mathbb{N}^+ & \text{(LO execution budget in normal mode)} \\
C_i^{\mathrm{hi}} &\in \mathbb{N}^+ & \text{(HI execution budget in degraded mode)} \\
\mathrm{BCET}_i &\in \mathbb{N}^+ & \text{(Best-case execution time, } \mathrm{BCET}_i \leq C_i^{\mathrm{lo}}) \\
\kappa_i &\in \{\mathrm{HI}, \mathrm{LO}\} & \text{(Criticality assignment)}
\end{aligned}
$$

### Utilisation

Per-task utilisation:
$$
U_i^{\mathrm{lo}} = \frac{C_i^{\mathrm{lo}}}{T_i}, \quad
U_i^{\mathrm{hi}} = \begin{cases}
    \frac{C_i^{\mathrm{hi}}}{T_i} & \text{if } \kappa_i = \mathrm{HI} \\
    0 & \text{if } \kappa_i = \mathrm{LO}
\end{cases}
$$

Total LO-criticality utilisation:
$$
U = \sum_{i=1}^n U_i^{\mathrm{lo}}
$$

HI-criticality utilisation factor:
$$
CP = \frac{|\tau^{\mathrm{HI}}|}{n}, \quad
U^{\mathrm{hi}} = CP \cdot CF \cdot U
$$

### Degradation Level Structure

| Level | Name | Trigger Condition | LO Task Handling |
|-------|------|-------------------|------------------|
| $L_0$ | Normal | None | All tasks run normally |
| $L_1$ | Degraded-1 | HI task reaches $R_1$ | Drop lowest-priority LO tasks |
| $L_2$ | Degraded-2 | HI task reaches $R_2$ | Drop more LO tasks |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $L_{k-1}$ | Fully Degraded | HI task reaches $R_{k-1} \approx R_{\mathrm{trigger}}$ | Drop all LO tasks |

The trigger points satisfy:
$$
0 < R_1 \leq R_2 \leq \dots \leq R_{k-1} = R_{\mathrm{trigger}}
$$

### Response Time Analysis (RTA)

**Normal Mode:**
$$
R_i^{\mathrm{lo}} = C_i^{\mathrm{lo}} + \sum_{\tau_j \in \mathrm{hp}(i)} \left\lceil \frac{R_i^{\mathrm{lo}}}{T_j} \right\rceil C_j^{\mathrm{lo}}
$$

**HI-Criticality:**
$$
R_i^{\mathrm{hi}} = C_i^{\mathrm{hi}} + \sum_{\tau_j \in \mathrm{hp}_{\mathrm{HI}}(i)} \left\lceil \frac{R_i^{\mathrm{hi}}}{T_j} \right\rceil C_j^{\mathrm{hi}}
+ \sum_{\tau_k \in \mathrm{hp}_{\mathrm{LO}}(i)} \left\lceil \frac{R_i^{\mathrm{lo}}}{T_k} \right\rceil C_k^{\mathrm{lo}}
$$

---

## Task Set Generation (DRS Algorithm)

Following Baruah et al. (RTSS 2020), utilisation vectors are generated via the Dirichlet-Rescale algorithm:

1. Draw $\mathbf{x} \sim \mathrm{Dirichlet}(1, \dots, 1)$ (uniform on simplex)
2. Scale: $u_i = U \cdot x_i$
3. Apply constraints via rescaling to standard simplex

Task parameters:
$$
\begin{aligned}
T_i &\sim \mathrm{LogUniform}(\log(100), \log(10000)) \\
D_i &= T_i \\
C_i^{\mathrm{lo}} &= \lfloor u_i \cdot T_i + 0.5 \rfloor \\
\mathrm{BCET}_i &\sim \mathrm{Uniform}(0.8 \cdot C_i^{\mathrm{lo}}, C_i^{\mathrm{lo}}) \\
C_i^{\mathrm{hi}} &= \min(\lfloor CF \cdot C_i^{\mathrm{lo}} + 0.5 \rfloor, T_i)
\end{aligned}
$$

---

## Metrics

| Metric | Definition |
|--------|------------|
| **NiD** | Number of times degraded mode is entered |
| **TiD** | Fraction of time spent in degraded mode |
| **JNE** | Jobs Not Executed (LO jobs dropped) |
| **LDM** | Late Degraded Mode (LO jobs missing deadline in degraded mode) |
| **HDM** | High Degradation Mode (HI jobs missing deadline) |

**Objective Function:**
$$
\Phi = \alpha \cdot \mathbb{E}[\mathrm{JNE}] + \beta \cdot \mathbb{E}[\mathrm{TiD}] + \gamma \cdot \mathbb{E}[\mathrm{WastedCPU}]
$$

---

## References

1. **AMC-RH**: "Analysis-Runtime Co-design for Adaptive Mixed-Criticality Scheduling", Bate et al., RTAS 2022
2. **AMC**: "Compensating Adaptive Mixed-Criticality Scheduling", Bate et al., RTNS 2022
3. **DRS**: "Generating Utilization Vectors for the Evaluation of Real-Time Scheduling Algorithms", Baruah et al., RTSS 2020

---

## Related Documents

- [LaTeX Source (task_set_model.tex)](./task_set_model.tex) - Formal mathematical specification
- [Multi-Mode AMC Research Plan](./multi_mode_amc.md) - Overall research approach including this model
- [SPECIFICATION.md](../SPECIFICATION.md) - Implementation specification
