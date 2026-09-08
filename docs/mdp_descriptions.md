# Centralized Roundabout Traffic Signal MDP ($N=6$ Approaches)

---

## Notation & Symbol Reference Table

| Category              | Symbol                                    | Domain / Units                            | Description / Definition                                                            | Typical Default                                             |
| --------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Topology & Scale**  | $N$                            | $\mathbb{Z}^+$                 | Number of inbound approaches                                                        | $6$                                                         |
|                       | $M$                            | $\mathbb{Z}^+$                 | Number of internal circulatory ring segments                                        | Scenario-dependent                                          |
|                       | $K$                            | $\mathbb{Z}^+$                 | Number of distinct selectable green phase stages                                    | Scenario-dependent                                          |
|                       | $\mathcal{L}_i^{\text{in}}$    | Set                                       | Set of lanes belonging to inbound approach $i$                           | —                                                           |
|                       | $v$                            | Entity                                    | Individual vehicle instance currently on the network                                | —                                                           |
| **Timing & Control**  | $\Delta t$                     | Seconds ($\text{s}$)           | Decision time step between consecutive actions                                      | $5.0\text{ s}$                                   |
|                       | $t_{\text{elapsed}}$           | Seconds ($\text{s}$)           | Elapsed time under the current active green phase                                   | —                                                           |
|                       | $t_{\text{min\_green}}$        | Seconds ($\text{s}$)           | Minimum green duration enforced for safety                                          | $10.0\text{ s}$                                  |
|                       | $t_{\text{yellow}}$            | Seconds ($\text{s}$)           | Yellow clearance transition time between phases                                     | $3.0\text{ s}$                                   |
|                       | $t_{\text{max\_green}}$        | Seconds ($\text{s}$)           | Upper threshold for phase duration normalization                                    | $60.0\text{ s}$                                  |
|                       | $T_{\text{horizon}}$           | Seconds ($\text{s}$)           | Episode truncation time limit                                                       | $3600\text{ s}$ (1 hour)                         |
| **State Features**    | $\mathbf{s}_t$                 | $\mathbb{R}^{3N + 2M + K + 1}$ | Full normalized observation vector at time step $t$                      | —                                                           |
|                       | $\rho_i$                       | $[0.0, 1.0]$                   | Normalized PCU density on inbound approach $i$                           | —                                                           |
|                       | $q_i$                          | $[0.0, 1.0]$                   | Normalized queue ratio on inbound approach $i$                           | —                                                           |
|                       | $o_i$                          | $[0.0, 1.0]$                   | Sublane lateral road surface occupancy on approach $i$                   | —                                                           |
|                       | $\omega_j$                     | $[0.0, 1.0]$                   | Road surface occupancy of circulatory ring segment $j$                   | —                                                           |
|                       | $v_j$                          | $[0.0, 1.0]$                   | Normalized mean vehicular velocity on ring segment $j$                   | —                                                           |
|                       | $\mathbf{p}_t$                 | $\{0, 1\}^K$                   | One-hot encoded vector representing active green phase                              | —                                                           |
|                       | $\tau_{\text{green}}$          | $[0.0, 1.0]$                   | Fraction of elapsed time in active green phase                                      | —                                                           |
| **Normalization**     | $w_{\text{pcu}}(v)$            | Dimensionless                             | PCU equivalency factor assigned by vehicle class                                    | MC: 0.3, Car: 1.0, Bus: 2.8, Truck: 2.5                     |
|                       | $C_i^{\text{pcu}}$             | Dimensionless                             | Upper saturation capacity threshold for approach $i$                     | Calculated from lane length                                 |
|                       | $Q_{\text{max}}$               | Vehicles                                  | Halting vehicle threshold for approach queue scaling                                | $50.0\text{ veh}$                                |
|                       | $V_{\text{limit}}$             | $\text{m/s}$                   | Circulatory ring maximum design speed limit                                         | $11.11\text{ m/s}$ ($40\text{ km/h}$) |
| **Actions**           | $a_t$                          | $\{0, \dots, K-1\}$            | Discrete global phase configuration index chosen at $t$                  | —                                                           |
|                       | $\mathbb{I}(\cdot)$            | $\{0, 1\}$                     | Indicator function; equals $1$if$a_t \neq a\_{t-1}$, else $0$ | —                                                           |
| **Reward Terms**      | $R_t$                          | $\mathbb{R}$                   | Scalar reward signal received at decision step $t$                       | —                                                           |
|                       | $W_t$                          | $\text{PCU}\cdot\text{s}$      | System-wide accumulated waiting time weighted by PCU                                | —                                                           |
|                       | $Q_t$                          | Vehicles                                  | Total halting vehicles queued across all inbound arms                               | —                                                           |
|                       | $\Omega_{\text{ring}, t}$      | Dimensionless                             | Quadratic barrier penalty for circulatory congestion                                | —                                                           |
|                       | $\omega_{\text{barrier}}$      | $[0.0, 1.0]$                   | Occupancy threshold beyond which ring penalty activates                             | $0.70$ ($70\%$)                                  |
| **Reward Weights**    | $\alpha$                       | Weight                                    | Multiplier for cumulative PCU delay ($W_t$)                              | $0.05$                                                      |
|                       | $\beta$                        | Weight                                    | Multiplier for inbound queue count ($Q_t$)                               | $0.10$                                                      |
|                       | $\delta$                       | Weight                                    | Multiplier for circulatory barrier penalty ($\Omega_{\text{ring}, t}$)   | $1.50$                                                      |
|                       | $\lambda$                      | Weight                                    | Fixed deduction penalty on phase change (flicker suppression)                       | $2.00$                                                      |
| **Deadlock Tracking** | $P_{\text{deadlock}}$          | Constant                                  | Severe penalty assessed when circulation locks                                      | $500.0$                                                     |
|                       | $\overline{V}_{\text{ring}}$   | $\text{m/s}$                   | Average vehicular speed across all internal ring segments                           | —                                                           |
|                       | $v_{\text{stall}}$             | $\text{m/s}$                   | Velocity threshold defining complete traffic freeze                                 | $0.10\text{ m/s}$                                |
|                       | $t_{\text{stall\_limit}}$      | Seconds ($\text{s}$)           | Stalling duration before gridlock penalty triggers                                  | $60.0\text{ s}$                                  |
|                       | $t_{\text{deadlock\_timeout}}$ | Seconds ($\text{s}$)           | Continuous stall duration triggering early termination                              | $120.0\text{ s}$                                 |

---

## 1. Scope & Control Parameters

- **Agent Architecture:** Single centralized controller observing all $N = 6$ inbound approaches and $M$ internal circulatory segments simultaneously.
- **Control Objective:** Maximize network-wide Passenger Car Unit (PCU) throughput, balance queues across arms, prevent phase flicker, and prevent internal circular ring deadlock.
- **Decision Interval ($\Delta t$):** Configurable interval (default: $5.0\text{ s}$).
- **Safety Constraints:**
- Minimum green duration $t_{\text{min\_green}}$ (default: $20.0\text{ s}$).
- Clearance yellow duration $t_{\text{yellow}}$ (default: $3.0\text{ s}$).

---

## 2. State Space ($\mathcal{S}$)

The observation is a continuous vector $\mathbf{s}_t \in \mathbb{R}^D$ normalized to $[0.0, 1.0]$ via `gymnasium.spaces.Box`:

$\mathbf{s}_t = \big[\mathbf{f}_{\text{inbound}}, \mathbf{f}_{\text{circulatory}}, \mathbf{p}_t, \tau_{\text{green}}\big] \in \mathbb{R}^{3N + 2M + K + 1}$

Where $N = 6$ inbound approaches, $M$ internal ring segments, and $K$ green phase combinations.

### Inbound Approach Features ($\mathbf{f}_{\text{inbound}} \in \mathbb{R}^{3N}$)

For each inbound approach $i \in \{1, \dots, N\}$:

- **PCU Density ($\rho_i \in [0, 1]$):**

$\rho_i = \min\left(1.0, \frac{\sum_{v \in \mathcal{L}_i^{\text{in}}} w_{\text{pcu}}(v)}{C_i^{\text{pcu}}}\right)$

- **Queue Ratio ($q_i \in [0, 1]$):**

$q_i = \min\left(1.0, \frac{\text{HaltingVehicles}_i}{Q_{\text{max}}}\right)$

- **Sublane Lateral Occupancy ($o_i \in [0, 1]$):** Road surface area coverage, capturing lateral packing and motorcycle lane filtering.

### Circulatory Ring Features ($\mathbf{f}_{\text{circulatory}} \in \mathbb{R}^{2M}$)

For each internal circulatory segment $j \in \{1, \dots, M\}$:

- **Ring Segment Occupancy ($\omega_j \in [0, 1]$):** Proportion of segment road surface occupied.
- **Normalized Ring Velocity ($v_j \in [0, 1]$):**

$v_j = \min\left(1.0, \frac{\overline{V}_{\text{ring}, j}}{V_{\text{limit}}}\right)$

### Signal Configuration Features

- **Active Phase One-Hot ($\mathbf{p}_t \in \{0, 1\}^K$):** One-hot encoded vector representing the active green phase.
- **Elapsed Green Ratio ($\tau_{\text{green}} \in [0, 1]$):**

$\tau_{\text{green}} = \min\left(1.0, \frac{t_{\text{elapsed}}}{t_{\text{max\_green}}}\right)$

---

## 3. Action Space ($\mathcal{A}$)

A discrete space representing the selectable phase configurations:

$a_t \in \{0, 1, \dots, K-1\}$

### Phase Transition Logic

- **Hold Phase ($a_t = a_{t-1}$):** Current green phase continues uninterrupted.
- **Switch Phase ($a_t \neq a_{t-1}$):**
- If $t_{\text{elapsed}} < t_{\text{min\_green}}$: Switch command is rejected; current phase is held until the safety threshold is satisfied.
- If $t_{\text{elapsed}} \ge t_{\text{min\_green}}$: Controller inserts yellow transition ($t_{\text{yellow}}$) before switching to phase $a_t$.

---

## 4. Reward Function ($\mathcal{R}$)

A centralized scalar reward balancing delay, queue length, ring spillback prevention, and control smoothness:

$R_t = -\alpha W_t - \beta Q_t - \delta \Omega_{\text{ring}, t} - \lambda \cdot \mathbb{I}(a_t \neq a_{t-1}) - P_{\text{deadlock}}$

| Term                                                           | Mathematical Definition                                                                                                     | Purpose                                                                                                   |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **PCU Delay ($W_t$)**                               | $\sum_{i=1}^{N} \sum_{v \in \mathcal{L}_i^{\text{in}}} w_{\text{pcu}}(v) \cdot \text{WaitingTime}(v)$            | Penalizes total accumulated vehicular delay weighted by vehicle class.                                    |
| **Total Inbound Queue ($Q_t$)**                     | $\sum_{i=1}^{N} \text{HaltingVehicles}_i$                                                                        | Penalizes physical queue buildup across all approach roads.                                               |
| **Circulatory Barrier ($\Omega_{\text{ring}, t}$)** | $\sum_{j=1}^{M} \max\left(0, \omega_j - \omega_{\text{barrier}}\right)^2$                                        | Quadratic penalty activated when any ring segment occupancy exceeds $\omega_{\text{barrier}}$. |
| **Switching Penalty ($\lambda$)**                   | $\lambda \cdot \mathbb{I}(a_t \neq a_{t-1})$                                                                     | Small penalty deducted on phase change to suppress rapid flickering.                                      |
| **Deadlock Penalty ($P_{\text{deadlock}}$)**        | Constant (applied if $\overline{V}_{\text{ring}} < v_{\text{stall}}$ for $> t_{\text{stall\_limit}}$) | Severe penalty if circulating ring reaches complete standstill.                                           |

---

## 5. Episode Lifecycle

- **Reset Protocol:** Sample active route file from the designated split (`TR` for training; `VA`/`TE` for evaluation) via `ScenarioSampler`, reinitialize TraCI simulation, and set signal state to phase 0.
- **Truncation:** Simulation reaches target horizon ($t \ge T_{\text{horizon}}$, e.g., $3600\text{ s}$).
- **Termination:** Episode terminates early if circulating traffic velocity remains below $v_{\text{stall}}$ continuously for $t_{\text{deadlock\_timeout}}$ (e.g., $120\text{ s}$).
