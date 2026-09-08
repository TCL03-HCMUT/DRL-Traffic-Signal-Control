# Centralized Roundabout Traffic Signal MDP ($N=6$ Approaches)

---

## Notation & Symbol Reference Table

| Category              | Symbol                                    | Domain / Units                            | Description / Definition                                                            | Typical Default                                             |
| --------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Topology & Scale**  | $N$                            | $\mathbb{Z}^+$                 | Number of inbound approaches                                                        | $6$                                                         |
|                       | $M$                            | $\mathbb{Z}^+$                 | Number of internal circulatory ring segments                                        | Scenario-dependent                                          |
|                       | $C$                            | $\mathbb{Z}^+$                 | Number of vehicle classes/types represented in queue features                       | Scenario-dependent                                          |
|                       | $K$                            | $\mathbb{Z}^+$                 | Number of distinct selectable green phase stages                                    | Scenario-dependent                                          |
|                       | $\mathcal{L}_i^{\text{in}}$    | Set                                       | Set of lanes belonging to inbound approach $i$                           | —                                                           |
|                       | $v$                            | Entity                                    | Individual vehicle instance currently on the network                                | —                                                           |
| **Timing & Control**  | $\Delta t$                     | Seconds ($\text{s}$)           | Decision time step between consecutive actions                                      | $5.0\text{ s}$                                   |
|                       | $t_{\text{elapsed}}$           | Seconds ($\text{s}$)           | Elapsed time under the current active green phase                                   | —                                                           |
|                       | $t_{\text{min\_green}}$        | Seconds ($\text{s}$)           | Minimum green duration enforced for safety                                          | $10.0\text{ s}$                                  |
|                       | $t_{\text{yellow}}$            | Seconds ($\text{s}$)           | Yellow clearance transition time between phases                                     | $3.0\text{ s}$                                   |
|                       | $t_{\text{max\_green}}$        | Seconds ($\text{s}$)           | Upper threshold for phase duration normalization                                    | $60.0\text{ s}$                                  |
|                       | $T_{\text{horizon}}$           | Seconds ($\text{s}$)           | Episode truncation time limit                                                       | $3600\text{ s}$ (1 hour)                         |
| **State Features**    | $\mathbf{s}_t$                 | $\mathbb{R}^{N(C+1) + M + K + 1}$ | Full normalized observation vector at time step $t$                      | —                                                           |
|                       | $\rho_i$                       | $[0.0, 1.0]$                   | Normalized PCU density on inbound approach $i$                           | —                                                           |
|                       | $\mathbf{q}_i$                 | $[0.0, 1.0]^C$                 | Normalized queue vector on inbound approach $i$, indexed by vehicle type/class | —                                                           |
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
| **Reward Weights**    | $\alpha$                       | Weight                                    | Multiplier for cumulative PCU delay ($W_t$)                              | $0.05$                                                      |
|                       | $\beta$                        | Weight                                    | Multiplier for inbound queue count ($Q_t$)                               | $0.10$                                                      |
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

$\mathbf{s}_t = \big[\mathbf{f}_{\text{inbound}}, \mathbf{f}_{\text{circulatory}}, \mathbf{p}_t, \tau_{\text{green}}\big] \in \mathbb{R}^{N(C+1) + M + K + 1}$

Where $N = 6$ inbound approaches, $C$ vehicle classes/types represented in the queue vector, $M$ internal ring segments, and $K$ green phase combinations.

### Inbound Approach Features ($\mathbf{f}_{\text{inbound}} \in \mathbb{R}^{N(C+1)}$)

For each inbound approach $i \in \{1, \dots, N\}$:

- **PCU Density ($\rho_i \in [0, 1]$):**

$\rho_i = \min\left(1.0, \frac{\sum_{v \in \mathcal{L}_i^{\text{in}}} w_{\text{pcu}}(v)}{C_i^{\text{pcu}}}\right)$

- **Queue Vector ($\mathbf{q}_i \in [0, 1]^C$):** Per-vehicle-type queue ratios on approach $i$, where component $c$ is the normalized number of halting vehicles of class $c$.

$\mathbf{q}_i[c] = \min\left(1.0, \frac{\text{HaltingVehicles}_{i,c}}{Q_{\text{max}, c}}\right)$

### Circulatory Ring Features ($\mathbf{f}_{\text{circulatory}} \in \mathbb{R}^{M}$)

For each internal circulatory segment $j \in \{1, \dots, M\}$:

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

A centralized scalar reward balancing delay, queue-length vector penalties, and control smoothness, with a decaying flicker penalty term:

$R_t = -\alpha W_t - \beta \|\mathbf{Q}_t\|_1 - P^{\text{flicker}}_t - P_{\text{deadlock}}$

where

$F_t = \lambda_f F_{t-1} + \mathbb{I}(a_t \neq a_{t-1})$

$P^{\text{flicker}}_t = w_f \cdot \frac{F_t}{F_{\text{norm}}}$

and $F_{\text{norm}} = \frac{1}{1 - \lambda_f}$.

| Term                                                           | Mathematical Definition                                                                                                     | Purpose                                                                                                   |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **PCU Delay ($W_t$)**                               | $\sum_{i=1}^{N} \sum_{v \in \mathcal{L}_i^{\text{in}}} w_{\text{pcu}}(v) \cdot \text{WaitingTime}(v)$            | Penalizes total accumulated vehicular delay weighted by vehicle class.                                    |
| **Total Inbound Queue ($Q_t$)**                     | $\mathbf{Q}_t = [\text{HaltingVehicles}_{1,1}, \dots, \text{HaltingVehicles}_{N,C}]$                             | Penalizes the vector of physical queue buildup by vehicle type across all inbound approaches.             |
| **Flicker Penalty ($P^{\text{flicker}}_t$)**       | $w_f \cdot \frac{F_t}{F_{\text{norm}}}$, where $F_t = \lambda_f F_{t-1} + \mathbb{I}(a_t \neq a_{t-1})$     | Penalizes repeated rapid phase switching while allowing recent switch history to decay over time.       |
| **Deadlock Penalty ($P_{\text{deadlock}}$)**        | Constant (applied if $\overline{V}_{\text{ring}} < v_{\text{stall}}$ for $> t_{\text{stall\_limit}}$) | Severe penalty if circulating ring reaches complete standstill.                                           |

---

## 5. Episode Lifecycle

- **Reset Protocol:** Sample active route file from the designated split (`TR` for training; `VA`/`TE` for evaluation) via `ScenarioSampler`, reinitialize TraCI simulation, and set signal state to phase 0.
- **Truncation:** Simulation reaches target horizon ($t \ge T_{\text{horizon}}$, e.g., $3600\text{ s}$).
- **Termination:** Episode terminates early if circulating traffic velocity remains below $v_{\text{stall}}$ continuously for $t_{\text{deadlock\_timeout}}$ (e.g., $120\text{ s}$).
