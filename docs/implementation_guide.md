# Traffic DRL Implementation Guide

This document is the working implementation guide for the project:

> Adaptive traffic-signal control with deep reinforcement learning in SUMO for mixed traffic.

It translates the project plan and the existing Python scaffold into an executable development order. It explains:

- what the complete system must do;
- which student owns each part;
- which files and functions must be implemented;
- what each function receives and returns;
- when each function should be implemented;
- what must be tested before the next task starts;
- how to avoid train/validation/test leakage.

The guide assumes the following ownership:

| Owner | Main responsibility                                                                     |
| ----- | --------------------------------------------------------------------------------------- |
| SV1   | SUMO network, vehicle types, route generation, calibration, emissions and simulation QA |
| SV2   | Gymnasium environment, observations, rewards, wrappers, DQN/PPO and checkpoints         |
| SV3   | Manifest, survey data, baselines, metrics, evaluation, statistics and reporting         |

Every owner is responsible for implementation, tests, documentation and a short walkthrough for another member.

---

## 1. Current Repository and Important Constraint

The repository is currently an API scaffold. Most core functions intentionally raise `NotImplementedError`. The implementation should follow the existing module boundaries instead of creating a second architecture.

Important existing areas:

```
src/traffic_drl/contracts.py                 Shared typed contracts
src/traffic_drl/environment/                 SUMO-RL environment and MDP logic
src/traffic_drl/train/                       Sampling, DQN, PPO and checkpoints
src/traffic_drl/baselines/                   Fixed-time and max-pressure controllers
src/traffic_drl/evaluation/                  Rollouts, parsers, metrics and plots
scenarios/generators/                        Route/scenario generation
configs/                                     YAML experiment configuration
sumo/                                        Network, additional XML and SUMO configs
tests/                                       Automated tests
```

The repository contains a small dummy four-arm intersection. The project plan describes the NgA Bay Ly Thai To target as a larger real-world intersection/roundabout. These are not automatically interchangeable.

Before calibration begins, the team must record one decision in `docs/decisions.md`:

1. Use the dummy network only for DEV-00 and create/update a target network for the real experiment; or
2. Use the dummy network as the Phase 1 experimental network and clearly label the result as a development experiment.

Do not claim real-world calibration until the network, demand and vehicle behavior have been compared with survey data.

---

## 2. System Execution Flow

The final execution flow is:

```
OSM/network and survey assumptions
        |
        v
SUMO network + vehicle types + route files
        |
        v
scenario_manifest.csv
        |
        v
ScenarioManifest -> ScenarioSampler
        |
        v
one ScenarioRecord and one route file
        |
        v
create_sumo_env(route_file, seed)
        |
        v
observation + action constraints + reward
        |
        v
Gymnasium wrappers -> DummyVecEnv -> VecNormalize
        |
        +------------------+
        |                  |
        v                  v
DQN/PPO training     Fixed-time/actuated baseline
        |                  |
        +--------+---------+
                 v
       VA selection and model freeze
                 |
                 v
       matched TE benchmark
                 |
                 v
tripinfo/emissions -> EpisodeMetrics -> CI -> report/plots
```

### Non-negotiable execution rules

1. A SUMO environment receives exactly one route file.
2. Never pass a comma-separated list of routes and expect SUMO-RL to choose one randomly.
3. Select one `ScenarioRecord` before creating the environment.
4. Close the old environment before creating the next scenario environment.
5. Training may use only `TR` scenarios.
6. Validation may select checkpoints but must not update model weights or normalization statistics.
7. Test scenarios remain locked until the model and evaluation bundle are frozen.
8. Every controller must use the same route file and matched seeds in a benchmark comparison.
9. Pilot results are for pipeline verification only. They are not final scientific evidence.
10. Every run records scenario ID, controller, seed, configuration path, code commit, SUMO version, Python version and output paths.

---

## 3. Shared Contracts

The contracts in `src/traffic_drl/contracts.py` are the shared vocabulary. Do not replace them with unrelated local dictionaries unless a compatibility adapter is necessary.

### 3.1 `ScenarioRecord`

`ScenarioRecord` represents one row in the scenario manifest.

Required fields:

```
scenario_id: str
split: str                 DEV, TR, VA or TE
route_file: Path           exactly one .rou.xml file
demand_seed: int
sumo_seed: int
num_seconds: int
metadata: dict[str, Any]  demand, turning ratio, vehicle mix, behavior, checksum
```

A scenario ID and route file must be stable after the split is locked. Paths are relative to the repository root in files and resolved to `Path` objects at the boundary.

### 3.2 `EpisodeMetrics`

Every completed controller episode must produce one `EpisodeMetrics` record.

Primary metrics:

- `average_waiting_time`
- `average_queue_length`
- `time_loss`

Constraint metrics:

- `throughput`
- `phase_switch_rate`
- `min_green_violations`

Additional metrics:

- `travel_time`
- `spillback`
- `recovery_time`
- `fuel`
- `co2`
- `nox`
- `inference_latency`

Missing primary metrics must raise a clear error. Do not silently convert missing values to zero.

### 3.3 `EnvironmentFactory`

The factory receives a scenario and a seed:

```python
factory(scenario: ScenarioRecord, seed: int) -> gym.Env
```

It must create a fresh environment bound to that scenario's one route file. It must not reuse a running SUMO process from another scenario.

### 3.4 `Controller`

All learned and heuristic controllers must expose:

```python
predict(observation, deterministic=True) -> action
reset() -> None
```

This allows the benchmark runner to evaluate fixed-time, max-pressure/actuated, DQN and PPO using one rollout interface.

---

## 4. Decisions Required Before Implementation

The following values are currently placeholders or inconsistent across planning documents. Record the final values in `docs/decisions.md` and update `docs/mdp_descriptions.md` before official training.

| Decision                              | Required final value                                            |
| ------------------------------------- | --------------------------------------------------------------- |
| Target junction and controlled signal | Real SUMO TLS ID, or documented metering scope                  |
| Approach order                        | Stable A1-A7 or network-specific order                          |
| Decision interval                     | For example, 5 seconds                                          |
| Yellow clearance                      | Measured or documented value                                    |
| Minimum green                         | One value used by environment and baselines                     |
| State feature order                   | Exact vector order and dimension                                |
| State bounds                          | Maximum queue, speed, waiting and phase duration                |
| Action mapping                        | Hold phase, switch phase or direct legal phase index            |
| Queue definition                      | Vehicle count, PCU, lane queue or detector threshold            |
| Throughput unit                       | Completed vehicles per simulated hour, or another explicit unit |
| Incomplete vehicles                   | Excluded, separately reported or treated by a documented policy |
| Emission units                        | SUMO output units and any conversion                            |
| Reward weights                        | Stored in YAML and included in run metadata                     |

Do not start long training while any of these decisions are unresolved.

---

## 5. File and Configuration Contracts

### 5.1 Scenario manifest

Create:

```
scenarios/scenario_manifest.csv
```

Required columns:

```
scenario_id,split,route_file,demand_seed,sumo_seed,num_seconds
```

Recommended metadata columns:

```
demand_profile,turning_ratio,vehicle_mix,behavior_profile,checksum
```

Example:

```csv
scenario_id,split,route_file,demand_seed,sumo_seed,num_seconds,demand_profile,turning_ratio,vehicle_mix,behavior_profile,checksum
DEV-00,DEV,scenarios/dummy/single-intersection-vhvh.rou.xml,5,5,600,low,balanced,mixed,pilot,
TR-01,TR,scenarios/train/tr_01_low.rou.xml,101,201,900,low,balanced,mixed,typical,
VA-01,VA,scenarios/validation/va_01.rou.xml,301,401,900,medium,new_seed,mixed,typical,
TE-01,TE,scenarios/test/te_01.rou.xml,501,601,900,held_out,replay,mixed,observed,<sha256>
```

### 5.2 Environment YAML

Create files under `configs/env/`. The existing configuration contract contains:

```yaml
network:
  net_file: sumo/net/example.net.xml
  route_files:
    - scenarios/train/tr_01_low.rou.xml
  additional_files:
    - sumo/additional/vehicle_types.add.xml

traffic_light:
  ts_id: junction_0
  single_agent: true

timing:
  num_seconds: 900
  delta_time: 5
  min_green: 30
  yellow_time: 3

sumo_options:
  use_gui: false
  lateral_resolution: 0.4
  additional_sumo_cmd: ""
  tripinfo_output: outputs/tripinfo/{run_id}/tripinfo.xml
  emission_output: outputs/tripinfo/{run_id}/emissions.xml

observation_bounds:
  max_queue_per_lane: 100
  max_time_in_phase: 300
```

The configuration loader must validate required fields before passing selected values into SUMO-RL. Do not pass the complete YAML mapping directly into an SB3 constructor.

### 5.3 Training YAML

Create files under `configs/train/`. It must identify:

- experiment name and seed;
- allowed split, which must be `TR` for official training;
- environment configuration;
- reward name and weights;
- normalization settings;
- DQN/PPO hyperparameters;
- total timesteps;
- checkpoint frequency;
- log location.

### 5.4 Evaluation YAML

Create files under `configs/evaluation/`. It must identify:

- split (`VA` or unlocked `TE`);
- manifest;
- scenario IDs or selection rule;
- evaluation seeds;
- model checkpoint;
- VecNormalize statistics;
- deterministic inference setting;
- output directory;
- metrics to collect;
- checksum file for locked test artifacts.

---

## 6. Detailed Work Allocation: SV1

SV1 owns all simulation inputs and must make the SUMO system reliable before SV2 runs long training.

### 6.1 Week 1: network, vehicle types and DEV-00

#### Network tasks

1. Preserve raw OSM data under `data/osm_raw/`.
2. Record download date, bounding box and OpenStreetMap attribution.
3. Generate or clean the SUMO network under `sumo/net/`.
4. Inspect lanes, directions, connections, TLS and edge IDs with NETEDIT.
5. Create `data/processed/approach_map.csv` mapping A1-A7 to actual edge/lane IDs.
6. Record every manual network edit.
7. Confirm whether the target is a signalized intersection, roundabout, entry metering system or another controllable system.

#### Vehicle-type tasks

Create `sumo/additional/vehicle_types.add.xml` containing at least:

- motorcycle;
- passenger car/taxi;
- bus/minibus;
- truck/goods vehicle.

Each class should support cautious, typical and assertive behavior profiles where possible. Keep physical vehicle properties separate from driver behavior parameters.

Important motorcycle/Sublane parameters include:

```
length, width, minGap, tau, accel, decel, sigma,
speedFactor, laneChangeModel, minGapLat, maxSpeedLat,
latAlignment, lcPushy, lcImpatience, lcStrategic, lcCooperative
```

Initial values are hypotheses. Store calibrated values and assumptions separately.

#### DEV-00 output

By the end of week 1, SV1 must provide:

- a network that opens in SUMO-GUI;
- one valid route file;
- a vehicle type file;
- a documented approach map;
- no invalid route/type references;
- a short scenario that can run without an RL policy.

### 6.2 Week 2: route generation functions

#### `generate_random_trips()`

File: `scenarios/generators/random_trips_gen.py`

Purpose:

- call SUMO's `randomTrips.py`;
- create one route file;
- preserve network and vehicle-type configuration;
- record the generation seed;
- validate the generated output.

Required behavior:

1. Create the output parent directory.
2. Build a subprocess command using explicit arguments.
3. Pass `network_file`, `begin`, `end`, `period` and `seed`.
4. Treat `period` as a generation interval, not as a guaranteed per-lane flow.
5. Include optional additional arguments only after validation.
6. Raise a clear error if SUMO or `randomTrips.py` fails.
7. Call route validation before returning.

Return:

```python
Path(output_route_file)
```

#### `count_realized_flow()`

Purpose: count the vehicles actually written into a route file by approach.

Required behavior:

- parse XML with `xml.etree.ElementTree`;
- resolve each vehicle's route;
- identify the first public edge;
- group vehicles by approach ID;
- return integer counts;
- report unknown approaches rather than silently dropping them.

Return example:

```python
{"A1": 120, "A2": 98, "A3": 0}
```

#### `validate_route_file()`

Required checks:

- XML is well formed;
- every route has an ID;
- every route has valid edges;
- every vehicle has a unique ID;
- every vehicle references an existing route;
- every vehicle type is defined or allowed;
- departure times are numeric and in the expected range;
- required route/type references are not missing.

Return an empty list for a valid file. Return descriptive error strings for invalid files.

This function must not run an RL policy.

### 6.3 Week 2: custom scenario generation

#### `generate_custom_flow()`

File: `scenarios/generators/custom_flow_gen.py`

Purpose: create deterministic scenarios for balanced demand, asymmetric flow, surges and platoons.

Inputs must be checked:

- demand is non-negative;
- turning ratios are valid and sum to the expected total per approach;
- vehicle mix is non-negative and sums to one within tolerance;
- `begin < end`;
- flow pattern is supported;
- seed is recorded.

The function must create valid `<route>`, `<vehicle>` and `<vType>` references and return the output path.

#### `generate_scenario_set()`

Purpose: generate every scenario in one allowed split from the manifest.

Required behavior:

- read the manifest once;
- select only the requested split;
- reject TE generation unless an explicit unlock flag/policy allows it;
- preserve scenario IDs and seeds;
- write routes below the requested output root;
- validate every generated route;
- return all generated paths.

#### `write_scenario_report()`

Write structured JSON or CSV containing:

- scenario ID;
- route file;
- generation seed;
- vehicle counts by approach;
- vehicle counts by type;
- validation errors;
- relevant metadata.

### 6.4 Week 3: calibration and SUMO outputs

SV1 must calibrate in this order:

1. demand and turning ratios;
2. longitudinal behavior: speed, headway, queue discharge, `tau`, `minGap`, acceleration, deceleration and `sigma`;
3. lateral behavior: motorcycle parallel movement, lateral position, lane changes, `minGapLat`, `lcPushy` and `lcImpatience`;
4. validation on a time period/seed not used for calibration;
5. ablation: single vehicle type, multiple vehicle types without Sublane, multiple vehicle types with Sublane.

Create and validate:

```
sumo/additional/vehicle_types.add.xml
sumo/additional/detectors.add.xml
sumo/additional/tls_programs.add.xml   only if supported by the real network
sumo/cfg/*.sumocfg
```

The SUMO configuration must write:

```
outputs/tripinfo/<run_id>/tripinfo.xml
outputs/tripinfo/<run_id>/emissions.xml
```

SV1 must document the exact emissions XML schema and units for SV3.

### 6.5 Week 4: training-input quality assurance

**Objective:** make every training route reliable enough for repeated DQN episodes.

SV1 must QA TR-01 through TR-07 by checking the network references, route edges, departure times, vehicle types, approach counts and vehicle-class counts. Run short SUMO episodes and record teleporting, invalid routes, stuck vehicles, missing outputs and output-path collisions. Correct only simulation-input defects; do not alter the training policy.

Implementation details:

- Extend `validate_route_file()` to report missing route edges, unknown vehicle types, duplicate vehicle IDs and invalid departure times.
- Use `count_realized_flow()` for every TR route and compare the result with the demand recorded in the manifest. Return a report rather than silently correcting the route.
- Add a small route/config smoke command around `generate_random_trips()` and `generate_custom_flow()` so a failed SUMO command includes the exact command, seed and output path.
- Record teleporting and invalid-route warnings in a structured QA result that SV3 can read.

The required output is a route QA report and a verified list of training route files for SV2. The report must include the command used, route path, demand seed, SUMO seed, realized flow and any remaining warning. SV3 reviews the demand totals and SV2 reviews whether the routes can be consumed by `create_sumo_env()`.

### 6.6 Week 5: domain and mixed-traffic QA

**Objective:** verify that the training scenarios cover the intended calibrated domain.

SV1 must compare requested demand with realized demand for each approach and verify the vehicle mix for TR-01 through TR-07. Test the cautious, typical and assertive profiles and the selected Sublane parameters. Confirm that changing the profile changes behavior in a controlled way rather than producing invalid or unstable traffic.

Run emissions on representative low, medium and high-demand scenarios. Record whether fuel, CO2 and NOx are present, their units, the number of records and the runtime overhead. Deliver a domain-coverage report, vehicle-mix report and emissions sample to SV2 and SV3.

Implementation details:

- Use `generate_scenario_set()` to enumerate only TR-01 through TR-07 for the training-domain check.
- Use `count_realized_flow()` to calculate realized demand by approach and parse vehicle `type` attributes to calculate class proportions.
- Validate `vehicle_mix` and `behavior_profiles` before passing them to `generate_custom_flow()`; reject proportions that do not sum to one within the configured tolerance.
- Keep emissions parsing in `parse_emissions()` owned by SV3, but provide SV3 with a known-good XML sample and the exact SUMO units.

### 6.7 Week 6: measurement validation for reward and metrics

**Objective:** confirm that all quantities used by the reward and evaluation are available and correctly interpreted.

SV1 must check queue, speed, waiting time, discharge headway, throughput and emissions measurements against SUMO output and the approach map. Confirm the difference between generated demand and completed vehicles. Run sensitivity checks for vehicle dimensions, `tau`, `minGap`, acceleration, deceleration and Sublane parameters.

The output is a short calibration-validation note containing definitions, units, known limitations and recommended normalization ranges. SV1 must not change the DQN model; changes to simulation inputs must be recorded in `docs/decisions.md` and reviewed by SV2 and SV3.

Implementation details:

- Define the source and unit of each quantity that will enter `queue_loss_reward()`, `wait_difference_reward()` or `combined_reward()`.
- Check that detector/lane IDs in `sumo/additional/detectors.add.xml` exist in the network before a run starts.
- Produce one machine-readable measurement summary for `collect_episode_metrics()` and one human-readable calibration note.
- If a measurement is unavailable, mark it as unavailable in metadata; do not substitute a fabricated zero.

### 6.8 Week 7: emissions by vehicle class

**Objective:** provide reliable environmental data without blocking the DQN deliverable.

SV1 must collect emissions by scenario and vehicle class on TR routes. Verify how SUMO identifies motorcycle, passenger car, bus and truck records, and document any limitation when class-level attribution is unavailable. Compare total emissions and per-vehicle emissions using the same completed-vehicle policy as the traffic metrics.

The required output is an emissions dataset and a unit-conversion note for SV3. If emissions significantly slow training, SV1 must measure the overhead and recommend whether emissions should be collected during training, evaluation only or in a separate batch.

Implementation details:

- Ensure each run has unique `tripinfo` and emissions paths under `outputs/tripinfo/<run_id>/`.
- Verify that `parse_emissions()` can distinguish `fuel`, `CO2` and `NOx` attributes from the actual SUMO output.
- Report both total and per-completed-vehicle values when the completed vehicle count is available.
- Do not make emissions a training reward term until the schema and units have been reviewed by SV2 and SV3.

### 6.9 Week 8: test-artifact preparation without policy access

**Objective:** prepare TE files while preserving the held-out nature of the test set.

SV1 may parse and syntax-check TE route files, verify network references, validate vehicle types and calculate checksums. SV1 must not run DQN, PPO or any policy on TE, and must not use TE behavior to tune demand, vehicle profiles or reward parameters.

Prepare the commands for TE-01 through TE-06 and the stress-test TE-03 runs, but keep the policy execution step disabled until SV3 records the formal unlock. The output is a TE consistency report and a reproducible batch command set.

Implementation details:

- Run `validate_route_file()` and the SUMO configuration parser only; do not call `evaluate_controller()` or any model `predict()` method.
- Calculate and store a checksum for every TE route, network file and additional XML file.
- Make the batch command accept an explicit scenario ID and seed so SV3 can later reproduce exactly one run.
- Record a hard failure when a TE route references a file or vehicle type outside the frozen bundle.

### 6.10 Week 9: simulation freeze and ablation

**Objective:** freeze the simulation inputs used by final evaluation.

SV1 must run the planned ablations on TR and VA: one vehicle type, multiple vehicle types without Sublane and multiple vehicle types with Sublane. Report the effect on queue, speed, discharge behavior, vehicle mix and emissions. Do not use TE for this comparison.

After the ablation decision, freeze the network, vehicle-type files, additional files, route files, SUMO configuration and detector configuration. Compute checksums and give the exact frozen bundle to SV2 and SV3. Any later change requires an incident record and a new approval.

Implementation details:

- Generate separate route/config variants for single-vType, multi-vType/no-Sublane and multi-vType/Sublane conditions; do not overwrite the baseline files.
- Use the same scenario IDs, demand seeds and SUMO seeds across ablations.
- Store the ablation condition in manifest metadata so `ScenarioRecord` and evaluation logs preserve it.
- Use a checksum command or helper that can be rerun before TE evaluation and reports changed files explicitly.

### 6.11 Week 10: matched TE-01 to TE-04 simulations

**Objective:** generate immutable raw simulation outputs for the first final benchmark.

SV1 must run Fixed-time, Actuated or Max-pressure, DQN and the proposed PPO/controller using the same route file, demand seed, SUMO seed, duration and network for every comparison. Preserve tripinfo, emissions, detector outputs, SUMO logs and process exit codes under a run-specific directory.

The output is an immutable benchmark run manifest. It must identify controller, scenario, seed, configuration, commit and output paths. SV1 must report failed or incomplete simulations instead of deleting them. SV3 checks the matching matrix and SV2 checks technical failures.

Implementation details:

- Create one run directory per controller/scenario/seed combination; never allow two controllers to write the same SUMO output file.
- Pass one `ScenarioRecord` and one route file to the environment factory for each run.
- Preserve SUMO stdout/stderr, tripinfo, emissions and detector outputs even when the process exits unsuccessfully.
- Add the run directory and exit status to the benchmark manifest consumed by `evaluate_manifest()`.

### 6.12 Week 11: robustness and stress-test simulation

**Objective:** evaluate vehicle-mix shift, replay data and overload behavior.

Run TE-05 and TE-06 when the required independent data exists. Repeat TE-03 over the agreed seeds and record spillback, gridlock risk, queue recovery and throughput collapse. Export trajectories and heatmap-ready data where possible.

The output is a stress-test package containing raw files, recovery summaries and a list of abnormal runs. Reruns must use the frozen bundle and retain the original failed output and reason for rerun.

Implementation details:

- Use the TE scenario's declared `num_seconds`, route file and seeds without changing demand after the model is frozen.
- Extract queue/spillback and recovery traces in a structured format that can be plotted by `plot_queue_heatmap()` or the equivalent plotting helper.
- Mark gridlock, teleporting, missing tripinfo and premature SUMO termination as separate failure categories.
- Keep the original run ID when a rerun is created by adding a rerun suffix rather than replacing the original directory.

### 6.13 Week 12: simulation figures and technical evidence

**Objective:** produce visual evidence directly from saved simulation data.

SV1 must create the network/approach diagram, calibration figures, queue heatmaps, representative trajectories and failure snapshots. Every figure must be generated from saved CSV/XML output or a documented processed derivative. Do not manually change plotted values.

Deliver figure source data, captions and a short explanation of what each figure demonstrates to SV3. SV3 verifies that the figures agree with the statistical tables.

Implementation details:

- Export approach/lane geometry from the frozen network for the network diagram; do not redraw it by hand with different IDs.
- Provide CSV or JSON inputs for the comparison and heatmap functions in `src/traffic_drl/evaluation/plots.py`.
- Include scenario ID, controller and seed in figure source data so every visual can be traced to a run.
- Generate figures again from the archived raw results before submission to confirm they are reproducible.

### 6.14 Week 13: simulation documentation and archive

**Objective:** make the simulation side reproducible by another team member.

SV1 must finalize instructions for SUMO installation, network generation, route generation, vehicle types, emissions and data assumptions. Remove local machine paths, preserve raw OSM and survey data separately from processed data, and archive the final frozen simulation bundle with checksums.

Another member must run the shortest SUMO/DEV-00 command sequence from the documentation. Any missing prerequisite or ambiguous path must be fixed before submission.

Implementation details:

- Verify that `build_network.sh`, scenario-generation commands and SUMO config paths do not assume a developer-specific working directory.
- Record the SUMO version, Python version and required environment variables used by the final run.
- Include one known-good DEV-00 route, network and additional-file bundle in the reproducibility archive.
- Confirm that the final archive contains raw inputs separately from generated outputs.

### 6.15 December buffer: simulation recovery and archiving

During 01/12-06/12, SV1 may rerun failed seeds or episodes, add missing metrics and repair critical SUMO, route or configuration defects. Each rerun must state the original failure, unchanged controller/model bundle and replacement output path.

During 07/12-13/12, SV1 completes technical appendices, verifies backups and checks that the final network, route, additional XML and raw-output archives can be opened independently.

For every buffer repair, SV1 must identify the affected function or SUMO file, preserve the failed artifact, rerun the smallest affected validation command and send the result to SV3 for approval.

### 6.16 SV1 acceptance checklist

- [ ] Raw OSM is preserved and attributed.
- [ ] Cleaned network starts in SUMO.
- [ ] Approach map is reviewed by SV3.
- [ ] Vehicle types reference valid SUMO classes.
- [ ] Route files pass XML and semantic validation.
- [ ] Realized flow is reported for every generated scenario.
- [ ] DEV-00, TR and VA routes run without unexpected teleporting.
- [ ] TE routes are syntax-checked but not used by a policy before freeze.
- [ ] Tripinfo and emissions files are produced with documented units.

---

## 7. Detailed Work Allocation: SV2

SV2 owns the RL-facing software. The environment must be correct before model tuning begins.

### 7.1 Week 1: MDP and observation contract

Update `docs/mdp_descriptions.md` with the final state/action/reward definitions.

#### State contract

The observation must have a fixed order. A recommended structure is:

```
for each approach/lane:
    queue or density
    PCU or vehicle count
    mean speed
    accumulated waiting time
phase one-hot encoding
elapsed time in current phase
```

The actual vector must be documented as a formula and dimension. All values must be finite `float32` values and must fit the declared Gymnasium space.

#### Action contract

Use a discrete action space. The action may mean:

- hold the current phase; or
- switch to the next legal phase.

The environment, not the policy, must enforce:

- minimum green time;
- yellow clearance;
- legal phase transitions;
- no unsafe direct jump between incompatible phases.

### 7.2 Observation functions

File: `src/traffic_drl/environment/custom_observations.py`

#### `MixedTrafficObservation.__call__()`

Must:

1. read data from the SUMO-RL `TrafficSignal`;
2. use the stable `approach_ids` order;
3. calculate the documented per-approach features;
4. append phase and elapsed-time features in the documented order;
5. clip or normalize according to the configuration;
6. return a fixed-shape finite numeric vector.

#### `MixedTrafficObservation.observation_space()`

Must return a Gymnasium space whose shape, dtype and bounds exactly match `__call__()`.

#### `QueueObservation.__call__()`

The DEV-00 pilot observation should remain minimal and stable. It should include queue/occupancy and current phase features only if those values are available from the dummy environment.

### 7.3 Environment factories

File: `src/traffic_drl/environment/make_env.py`

#### `create_sumo_env()`

Responsibilities:

- map validated configuration values to SUMO-RL;
- pass exactly one route file;
- configure network, TLS, timing, observation and reward functions;
- pass `fixed_ts=True` for the fixed-time baseline when requested;
- pass the seed explicitly;
- select GUI or headless mode;
- create output directories before SUMO starts;
- configure tripinfo and emissions output;
- return a Gymnasium-compatible environment.

Do not hard-code a developer home directory. Do not pass a comma-separated route list.

#### `make_dev_environment()`

Select the `DEV-00` record from the manifest and call `create_sumo_env()` with the pilot settings.

#### `make_vectorized_environment()`

Required order:

```
single Gymnasium environment
    -> DummyVecEnv
    -> optional VecNormalize
```

Training mode may update normalization statistics. Evaluation mode must load saved statistics and set:

```python
vec_env.training = False
vec_env.norm_reward = False
```

#### `close_environment()`

Close the outer wrapper and ensure the underlying SUMO process is terminated. It must be safe to call after a failed episode.

### 7.4 Reward functions

File: `src/traffic_drl/environment/custom_rewards.py`

All reward functions must return one finite scalar `float`.

#### `queue_loss_reward()`

Return a reward based on the change in cumulative queue loss. Lower queue loss should produce a better reward. Define whether the value is a difference, negative absolute value or normalized difference in `docs/mdp_descriptions.md`.

#### `pressure_reward()`

Calculate incoming versus outgoing pressure using the same lane/approach mapping used by the observation. Normalize it so the scale is stable between scenarios.

#### `wait_difference_reward()`

Reward reductions in cumulative waiting time. Do not use a value that makes the reward improve while waiting-time metrics become clearly worse.

#### `emissions_penalty()`

Use only after SV1 confirms the emissions data and units. Return a non-positive penalty after normalization.

#### `combined_reward()`

Combine normalized components using configuration weights. The `context` may contain:

- previous action;
- current action;
- previous metric values;
- normalization scales;
- phase-switch status.

Apply the switch penalty only when a real phase switch occurs. Record all weights in the training YAML and run metadata.

### 7.5 Wrappers

File: `src/traffic_drl/environment/wrappers.py`

#### `MultiScenarioWrapper.reset()`

The existing class cannot safely replace a route inside an already-created SUMO-RL environment. The preferred architecture is to select a scenario before creating the environment through an `EnvironmentFactory`.

If this wrapper is retained, it must receive a factory capable of closing and recreating the underlying environment. It must never merely change `active_scenario` while leaving SUMO attached to the old route.

#### `MultiScenarioWrapper.step()`

Delegate to the active environment and attach scenario ID, split and seed to `info`.

#### `MetricsInfoWrapper.reset()`

Initialize per-episode counters and return the normal Gymnasium pair:

```python
observation, info
```

#### `MetricsInfoWrapper.step()`

Return the five-element Gymnasium result:

```python
observation, reward, terminated, truncated, info
```

Attach current and final metrics to `info` without changing the observation or reward contract.

#### `wrap_environment()`

Apply wrappers in a documented, stable order and preserve the declared action/observation spaces.

### 7.6 Scenario sampler

File: `src/traffic_drl/train/scenario_sampler.py`

#### `ScenarioSampler.sample()`

Select exactly one record from the configured split using a reproducible random generator. Reject invalid split names and reject TE unless test access is explicitly unlocked.

#### `ScenarioSampler.sample_route()`

Return the selected route as one string path. Never join multiple route paths with commas.

#### `ScenarioSampler.set_split()`

Validate the split and update it explicitly. Changing to TE must require an explicit freeze/unlock policy.

### 7.7 DQN implementation

File: `src/traffic_drl/train/train_dqn.py`

#### `build_dqn_model()`

Construct SB3 DQN from selected configuration values only. Configure:

- policy;
- learning rate or schedule;
- buffer size;
- learning starts;
- batch size;
- gamma;
- target update interval;
- exploration schedule;
- seed;
- tensorboard path.

The model must receive a compatible Gymnasium or vectorized environment.

#### `train_dqn()`

Call `model.learn()` with the requested timesteps and callback. Return the trained model. Preserve `reset_num_timesteps=False` when resuming.

#### `run_dqn_pilot()`

Run only on DEV-00. It must:

1. create the environment;
2. run a short reset/step smoke test;
3. train the requested pilot budget;
4. save a checkpoint;
5. reload the checkpoint;
6. run inference with legal actions;
7. verify rewards and losses are finite;
8. return the model.

The pilot must not be used as the final result.

### 7.8 Learning-rate schedules

File: `src/traffic_drl/train/lr_schedules.py`

#### `linear_schedule()`

Return a callable accepting SB3 remaining progress in `[0, 1]` and interpolating from `initial_value` to `final_value`.

#### `cosine_schedule()`

Return a callable using cosine interpolation between the initial and final values. Validate non-negative rates and keep the return value finite.

### 7.9 Checkpoints and callbacks

A checkpoint bundle should contain:

```
model.zip
replay_buffer.pkl             DQN when continuing training
vecnormalize.pkl
resume_info.json
config.yaml or config copy
```

`resume_info.json` must include:

- timestep;
- model class;
- seed;
- split;
- config path or copied config;
- git commit;
- SUMO/Python versions when available;
- extra run metadata.

`RobustCheckpointCallback` must save complete bundles periodically and at training end. `TrafficMetricsCallback` must log the same metric names used by evaluation. `Phase1PilotCallback` may reuse the robust checkpoint logic.

### 7.10 Week 4: official TR-only DQN pipeline

**Objective:** replace the pilot workflow with a repeatable training workflow that samples only allowed training scenarios.

SV2 must implement `make_vectorized_environment()` and connect `ScenarioSampler` to fresh one-route environments. Implement `build_dqn_model()` and `train_dqn()`, load configuration values explicitly and fit `VecNormalize` only on TR. Complete the checkpoint callback so every saved bundle contains model, normalizer, resume metadata and the relevant configuration.

The output is one command that starts a short official TR-only training run. It must log scenario ID, split, demand seed, SUMO seed, model seed and configuration path for each episode. SV1 reviews SUMO options and SV3 audits that no VA or TE record was sampled.

Implementation details:

- `make_vectorized_environment()` must create one `DummyVecEnv` around an environment bound to one selected route, then optionally apply `VecNormalize`.
- `build_dqn_model()` must map only approved YAML keys to SB3 `DQN`; it must not pass the whole configuration mapping to the constructor.
- `train_dqn()` must preserve `reset_num_timesteps=False` when resuming and must forward the checkpoint/metrics callback.
- `ScenarioSampler.sample()` and the environment factory must expose the selected `ScenarioRecord` in `info` and training logs.

### 7.11 Week 5: DQN domain coverage and training monitoring

**Objective:** train DQN across the calibrated TR domain and detect learning or simulation failures early.

SV2 must continue DQN training over TR-01 through TR-07, implement `linear_schedule()` and `cosine_schedule()` if required by the selected configuration, and complete `TrafficMetricsCallback`. Monitor reward together with waiting time, queue length, time loss, throughput and phase-switch rate. A reward increase with a clear traffic regression must be recorded as a reward-design problem, not accepted as success.

The output is a training curve, callback log, schedule test and failure log. SV1 reviews traffic behavior and SV3 reviews scenario coverage and metric names.

Implementation details:

- `linear_schedule()` must map SB3 remaining progress from one to zero without returning a negative learning rate.
- `cosine_schedule()` must return a finite value at both endpoints and remain within the configured rate range.
- `TrafficMetricsCallback._on_step()` must log the same names used by `EpisodeMetrics`, including waiting, queue, time loss, throughput and phase switches.
- Add a training-time check that rejects NaN rewards, observations or losses and records the scenario/seed that produced them.

### 7.12 Week 6: reward v1, checkpoint reliability and VA selection

**Objective:** produce the first defensible DQN checkpoint selected only with validation data.

SV2 must finalize `combined_reward()` and its switch penalty using the measurement definitions supplied by SV1. Validate `save_checkpoint()` and `load_checkpoint()` including replay buffer, VecNormalize, configuration and resume metadata. Run reward v1 on TR and evaluate candidate checkpoints on VA with `training=False` and `norm_reward=False`.

The output is M2: the best DQN checkpoint selected by VA-01/VA-02, together with its immutable normalizer, configuration, seed and commit metadata. SV3 makes the selection decision; SV1 reviews the environment and reward inputs.

Implementation details:

- `combined_reward()` must use the normalized measurement scales agreed with SV1 and apply `switch_penalty` only when the phase actually changes.
- `save_checkpoint()` must write model, replay buffer when requested, VecNormalize statistics and `ResumeInfo` metadata as one bundle.
- `load_checkpoint()` must restore VecNormalize before loading the model and must support `training=False` and `norm_reward=False` for VA.
- Run one short reset/step after loading a checkpoint before accepting it as M2.

### 7.13 Week 7: PPO implementation after DQN stability

**Objective:** add PPO without endangering the minimum DQN deliverable.

Only after DQN can train, save, reload and evaluate, SV2 may implement `build_ppo_model()`, `train_ppo()` and the PPO checkpoint adapters. PPO must use the same environment contract, TR-only sampler, observation order, safety constraints and logging fields as DQN.

The output is a PPO pilot or checkpoint and a short DQN/PPO stability comparison. If PPO is unstable or exceeds the time budget, SV2 must stop PPO work and preserve DQN as the final minimum model.

Implementation details:

- `build_ppo_model()` must use the same observation/action spaces, seed policy, callbacks and log metadata as DQN.
- `train_ppo()` must accept the same callback and resume semantics as `train_dqn()`.
- `save_ppo_checkpoint()` and `load_ppo_checkpoint()` must preserve the normalizer and configuration required for deterministic evaluation.
- PPO training must construct environments from TR records only; VA is for evaluation and TE remains inaccessible.

### 7.14 Week 8: VA-only tuning and inference checks

**Objective:** choose model settings without inspecting held-out test performance.

SV2 may run only the pre-declared tuning budget using TR for learning and VA for selection. Test checkpoint reload, deterministic prediction, legal actions, observation normalization and inference latency. Attempt LSTM/RecurrentPPO only if DQN/PPO artifacts and the evaluation schedule are already safe.

The output is a candidate-model table, selected hyperparameters, inference smoke-test log and a statement that TE was not accessed. SV3 audits the tuning records and SV1 confirms that no simulation assumptions changed.

Implementation details:

- Use `evaluate_manifest()` with `split="VA"` for candidate comparison and keep `VecNormalize.training=False` during every evaluation.
- Call `model.predict(observation, deterministic=True)` during inference checks and validate the returned action against `action_space`.
- Record each candidate's config path, model seed, training timesteps and validation seeds; do not select by a single episode.
- If LSTM/RecurrentPPO is attempted, isolate it in its own config and checkpoint directory so it cannot overwrite DQN/PPO artifacts.

### 7.15 Week 9: model and inference freeze

**Objective:** create the immutable learned-controller package before TE is opened.

SV2 must select the final model using VA, freeze the checkpoint, normalizer, configuration, inference script and policy settings, and record the git commit and software versions. Compute checksums for the model bundle. After SV3 records TE unlock, do not change model weights, reward weights, normalizer or hyperparameters.

The output is a complete evaluation bundle that another member can load without retraining. SV1 checks the input compatibility and SV3 signs the freeze.

Implementation details:

- Use `CheckpointBundle` to verify the model, normalizer, resume metadata and configuration paths exist.
- Store a checksum or freeze marker covering the model zip, normalizer, config, inference script and code commit.
- Load the frozen bundle in a fresh process and perform one deterministic DEV/VA smoke rollout before SV3 unlocks TE.
- After unlock, reject training-mode loads and reject any configuration whose checksum differs from the signed bundle.

### 7.16 Week 10: deterministic inference on TE-01 to TE-04

**Objective:** run the frozen controller under final benchmark conditions.

SV2 must load the final model and VecNormalize in evaluation mode, run deterministic inference and record action latency, episode termination, exceptions and technical failures. The controller must use the same route, seed and duration as every baseline.

Only technical runner defects may be repaired. A repair must not alter the model, reward, normalization statistics or scenario files. Give SV3 the inference logs and SV1 the SUMO process/failure details.

Implementation details:

- `evaluate_controller()` must call `reset()` on the controller, reset the environment with the requested seed and use deterministic prediction.
- Measure inference time around each `predict()` call and store an episode summary, not only a console print.
- Ensure the environment is closed in a `finally` path after success or failure.
- Attach controller name, scenario ID and seed before constructing the final `EpisodeMetrics` record.

### 7.17 Week 11: failure, recovery and latency analysis

**Objective:** describe how the frozen controller behaves under robustness and stress scenarios.

SV2 must analyze TE-05/TE-06 and repeated TE-03 runs for failure modes, recovery time, phase oscillation and inference latency. Do not hide failed episodes and do not tune the model in response to them. A rerun is allowed only when the failure is technical and is documented with the original output.

The output is a failure taxonomy, latency summary and technical incident log for SV3.

Implementation details:

- Separate policy failures, SUMO/process failures, parser failures and incomplete-vehicle cases.
- Compare recovery time and phase-switch behavior using the same metrics as the final benchmark.
- Do not call `learn()`, update normalization statistics or overwrite the frozen checkpoint during analysis.
- A rerun must reference the original run ID and use the same `ScenarioRecord` and seed.

### 7.18 Week 12: methods and reproducibility documentation

**Objective:** document the learned-controller implementation clearly enough to reproduce it.

SV2 must write the MDP, observation, action safety, reward, DQN/PPO, checkpointing, normalization and inference sections. Clean configuration paths and include short commands for pilot, training, validation, checkpoint loading and deterministic inference.

The output is the methods draft and reproducible configuration bundle. SV1 reviews SUMO-specific statements and SV3 checks that all claims match the saved artifacts.

Implementation details:

- Document the exact feature order implemented by `MixedTrafficObservation.__call__()` and the action mapping enforced by the environment.
- Document reward weights, normalization mode, model hyperparameters, training split and checkpoint filenames.
- Include the commands that call `run_dqn_pilot()`, `train_dqn()`, `train_ppo()` and the evaluation runner.
- State which features and metrics are measured directly by SUMO and which are derived by project code.

### 7.19 Week 13: clean-environment reproduction and submission package

**Objective:** verify that the software can be used by a second member from the README.

SV2 must finalize `README.md`, requirements/config examples, pilot and inference commands, and the final model artifact. Run the complete short pipeline from a clean environment: setup, DEV-00, baseline-compatible inference, metric output and checkpoint loading.

The output is the code/config/model archive and demo command. SV3 verifies the artifact inventory and another member runs the documented command sequence.

Implementation details:

- Run a clean import and a short DEV-00 inference test from the README without relying on a local absolute path.
- Verify that `load_dqn_checkpoint()` or `load_ppo_checkpoint()` can restore the model with the saved environment and normalizer.
- Include `resume_info.json`, configuration copies and the git commit in the final archive.
- Record any unavailable GPU or SUMO feature as an explicit environment prerequisite.

### 7.20 December buffer: technical recovery and archive

During 01/12-06/12, SV2 may repair critical training/evaluation pipeline defects or restore a valid checkpoint. The frozen model must not be changed merely to improve final results. Every technical rerun requires an incident note and updated artifact references if applicable.

During 07/12-13/12, SV2 completes the code/config/checkpoint archive and verifies the final command sequence, normalizer and resume metadata.

For every buffer repair, SV2 must run the narrowest affected smoke test first, preserve the frozen model unless the issue is a documented artifact corruption, and send the rerun metadata to SV3.

### 7.21 SV2 acceptance checklist

- [ ] State order and dimension are written down.
- [ ] Observation values match `observation_space()`.
- [ ] Actions cannot violate minimum green or yellow clearance.
- [ ] Reset/step/close work with random actions.
- [ ] Rewards are finite and have the expected sign.
- [ ] One route is attached to one environment.
- [ ] DEV-00 pilot saves and reloads successfully.
- [ ] Official training sampler accepts only TR.
- [ ] VA does not update model or normalizer.
- [ ] Checkpoint bundle contains all required companion files.
- [ ] DQN is complete before PPO/LSTM work begins.

---

## 8. Detailed Work Allocation: SV3

SV3 owns the experiment definition and must protect the validity of comparisons.

### 8.1 Manifest and scenario catalog

File: `src/traffic_drl/environment/scenario_factory.py`

#### `ScenarioManifest.from_csv()`

Use `csv.DictReader` and validate:

- required header columns;
- UTF-8 input;
- duplicate scenario IDs;
- split values;
- integer seeds and duration;
- route file existence;
- checksum format when supplied;
- TE lock policy.

Preserve optional columns in `metadata`.

#### `records_for_split()`

Return records in manifest order for exactly one requested split.

#### `get()`

Return one record by exact ID and raise a clear error if the ID is unknown or duplicated.

#### `load_manifest()`

Load and return `ScenarioManifest` using the shared parser.

#### `select_scenario()`

Select by exact ID when provided. Otherwise select reproducibly using the supplied seed. Validate that the selected record belongs to the requested split.

#### `generate_selected_scenario()`

Select one record, invoke the appropriate generator and return the generated route path. It must preserve manifest ID and generation metadata.

#### `main()`

Provide a command-line entry point for selecting/generating one scenario or a split. Return zero on success and a non-zero exit code on validation failure.

### 8.2 Survey and leakage policy

SV3 must maintain:

```
data/survey_raw/
data/processed/
docs/decisions.md
docs/assumptions.md
```

Survey records should include:

- session/date/time;
- approach and movement;
- vehicle class;
- 15-minute interval;
- queue estimate;
- phase timing;
- weather/abnormal events;
- confidence level;
- calibration/validation/test purpose.

If surveying is impossible, label all assumed demand and timing values explicitly and use sensitivity analysis. Do not describe the model as field-calibrated.

### 8.3 Fixed-time baseline

File: `src/traffic_drl/baselines/fixed_time.py`

#### `create_fixed_time_environment()`

Call the same environment factory/configuration used by DRL, with `fixed_ts=True`. Use the same route, network, duration and seed as the competing controller.

#### `run_fixed_time_episode()`

Run until termination/truncation, collect final info and parsed output files, and return one `EpisodeMetrics` record.

#### `evaluate_fixed_time()`

Run every route/seed combination and optionally serialize results. The output schema must match DQN/PPO evaluation exactly.

### 8.4 Actuated or max-pressure baseline

File: `src/traffic_drl/baselines/max_pressure.py`

Use this module only if max-pressure is the agreed heuristic baseline. If the project requires a real actuated controller, implement it behind the same `Controller` interface and document the distinction.

#### `MaxPressureController.predict()`

Read the observation, calculate pressure for legal phase choices and return a legal action. Respect minimum green and yellow timing as enforced by the environment.

#### `MaxPressureController.reset()`

Reset phase and timing state before every episode.

#### `run_max_pressure_episode()`

Use the same rollout and metric collection procedure as fixed-time and learned controllers.

### 8.5 Tripinfo and emissions parsers

File: `src/traffic_drl/evaluation/parse_tripinfo.py`

#### `parse_tripinfo()`

Use `xml.etree.ElementTree`. For every `<tripinfo>` element:

- parse `waitingTime`;
- parse `timeLoss`;
- parse `duration`;
- count completed vehicles;
- compute means;
- compute throughput using the documented episode-hour denominator.

Do not silently treat missing attributes as zero. Decide and document how incomplete vehicles are represented.

#### `parse_emissions()`

Support the exact SUMO emissions schema produced by SV1. Sum fuel, CO2 and NOx using documented units. Handle capitalization differences deliberately, not accidentally.

#### `merge_tripinfo_metrics()`

Combine traffic and environmental metrics only after controller, scenario ID and seed are attached. The current signature may need an explicit metadata extension or a caller-side construction step. Do not invent phase-switch or min-green values if they were not measured.

### 8.6 Metric functions

File: `src/traffic_drl/evaluation/metrics.py`

#### `standard_metric_names()`

Return primary, constraint and additional metric names in a stable order used by CSV, JSON and plots.

#### `collect_episode_metrics()`

Read one final `info` mapping. Require all primary metrics and required constraints. Convert values to the correct numeric types and attach controller/scenario/seed.

#### `confidence_interval()`

Calculate a 95% confidence interval across independent evaluation seeds. Handle one observation explicitly and document the behavior.

#### `aggregate_metrics()`

Group records by controller and scenario. For each metric calculate:

- sample count;
- mean;
- standard deviation;
- confidence interval.

Do not mix scenarios before producing per-scenario summaries.

#### `interpret_results()`

Compare each controller with the chosen baseline. Report:

- absolute metric values;
- percentage improvement where lower is better;
- throughput regressions;
- phase-switch regressions;
- minimum-green violations;
- missing or failed episodes.

A lower reward alone is not evidence of a better traffic controller.

### 8.7 Evaluation functions

File: `src/traffic_drl/evaluation/evaluate_benchmark.py`

#### `evaluate_controller()`

For every episode:

1. reset controller state;
2. reset environment with the requested seed;
3. request a deterministic action when evaluating learned policies;
4. step until terminated or truncated;
5. record inference latency;
6. collect final metrics;
7. return one `EpisodeMetrics` record.

#### `evaluate_manifest()`

Select only the requested split. For each scenario and evaluation seed, create a fresh environment. The same scenario/seed matrix must be used for every controller.

#### `save_evaluation_results()`

Write structured CSV or JSON. CSV contains one row per episode. JSON contains a list of objects with the same field names. Never serialize Python `repr()` output.

#### `run_benchmark()`

Read evaluation configuration and run:

1. fixed-time;
2. actuated or max-pressure;
3. DQN;
4. optional PPO/proposed model.

Before TE execution, verify:

- model freeze marker/checksum exists;
- normalizer is loaded with `training=False`;
- test manifest checksum matches;
- deterministic inference is enabled;
- no training callback is active.

### 8.8 Plots

File: `src/traffic_drl/evaluation/plots.py`

Implement plotting only after the metric schema is stable. Recommended outputs:

- controller comparison chart;
- learning curve;
- queue heatmap by approach/time;
- emissions comparison;
- per-scenario confidence interval plot.

Plots must be generated from saved metric files, not hand-edited values.

### 8.9 Week 4: smoke testing and validation runner

**Objective:** create an independent validation path before official training becomes expensive.

SV3 must implement `check_environment()` and `run_smoke_test()` in `src/traffic_drl/evaluation/test.py`. The smoke test must check reset, random legal actions, finite rewards, correct termination flags and clean environment closure. SV3 must also begin `evaluate_controller()` and define the evaluation seed matrix and separate VA configuration.

The output is an automated smoke-test result, a VA runner draft and a seed matrix. SV2 checks the Gymnasium/SB3 contract and SV1 checks whether the reported metrics have a valid traffic interpretation.

Implementation details:

- `check_environment()` must call the SB3 environment checker and fail on a reset/step/space mismatch.
- `run_smoke_test()` must use legal random actions, reject non-finite rewards and return `SmokeTestResult` with the final `info` mapping.
- `evaluate_controller()` must reset the controller and environment for each episode and attach controller, scenario and seed metadata.
- The VA seed matrix must be stored in configuration rather than generated implicitly during evaluation.

### 8.10 Week 5: baseline completion and matched inputs

**Objective:** make all controllers comparable using the same experiment inputs.

SV3 must complete `run_fixed_time_episode()` and `evaluate_fixed_time()`. Implement or finalize the actuated/max-pressure baseline and ensure it follows the same `Controller` interface and safety rules. Every baseline run must use the same network, route file, duration and seed that will be used by DQN and PPO.

The output is a baseline runner and matched DEV/TR/VA baseline results. SV1 reviews controller meaning and traffic behavior; SV2 reviews environment and interface compatibility.

Implementation details:

- `create_fixed_time_environment()` must call the same environment construction path as DRL with `fixed_ts=True` and one selected route.
- `run_fixed_time_episode()` must collect final `info`, tripinfo and emissions and convert them to `EpisodeMetrics`.
- `evaluate_fixed_time()` must iterate over the same route/seed matrix later used by DQN and PPO.
- `MaxPressureController.predict()` must return a legal action and `reset()` must clear phase/timing state before each episode.

### 8.11 Week 6: parsers, metric schema and VA checkpoint selection

**Objective:** turn SUMO output and environment information into reliable typed metrics.

SV3 must implement `parse_tripinfo()`, `parse_emissions()`, `merge_tripinfo_metrics()`, `standard_metric_names()`, `collect_episode_metrics()`, `confidence_interval()` and `aggregate_metrics()`. Required primary metrics must be rejected when missing rather than replaced with zero. Throughput, incomplete vehicles and emissions units must follow the decisions recorded with SV1.

Run the parser tests and evaluate DQN candidates on VA-01/VA-02. Select M2 only from validation results and record the selection rule, candidates and seed set. TE must remain locked.

Implementation details:

- `parse_tripinfo()` must parse numeric `waitingTime`, `timeLoss` and `duration` attributes with `ElementTree`, count completed vehicles and apply the documented throughput denominator.
- `parse_emissions()` must sum the exact fuel, CO2 and NOx attributes produced by SV1 and fail clearly on missing required data.
- `collect_episode_metrics()` must reject missing primary metrics instead of inventing zeros.
- `aggregate_metrics()` must group by controller and scenario; `confidence_interval()` must operate over independent evaluation seeds.

### 8.12 Week 7: VA comparison and ablation planning

**Objective:** compare candidate controllers without using held-out test results.

SV3 must extend `evaluate_manifest()` for VA and compare DQN, PPO when available, fixed-time and actuated/max-pressure. Build the ablation matrix for reward, algorithm, vehicle type and Sublane settings. Check that the same scenario/seed matrix is used across controllers and that validation does not update model or normalizer state.

The output is a VA comparison report, an ablation plan and a leakage audit. SV2 uses the report for model decisions and SV1 validates the traffic interpretation.

Implementation details:

- `evaluate_manifest()` must select only `VA`, create a fresh environment for each scenario/seed and never update model or normalizer state.
- Use `save_evaluation_results()` to write one structured CSV/JSON record per episode, including failed-run metadata where applicable.
- Run the same controller list and seed matrix for every candidate so the ablation comparison is paired.
- Store the ablation condition in the scenario or run metadata instead of encoding it only in a filename.

### 8.13 Week 8: evaluation protocol and test preparation

**Objective:** define how the final benchmark will be run before TE results are visible.

SV3 must check TR and VA coverage, complete the first version of `interpret_results()`, define early-stopping and failure-handling rules, and prepare the final evaluation YAML. Include the model checkpoint, normalizer, manifest, evaluation seeds, controller names, metrics and output directory.

The output is a pre-declared VA selection protocol and benchmark configuration. SV3 must not open TE or inspect policy results on TE. SV2 checks artifact paths and SV1 checks scenario labels.

Implementation details:

- `interpret_results()` must define which metric direction is better and must flag throughput, phase-switch and minimum-green regressions.
- The evaluation YAML must identify manifest, split, scenario IDs, seeds, checkpoint, normalizer, deterministic mode and output directory.
- Validate that every scenario in the VA selection protocol belongs to VA and that no TE result path is referenced.
- Record the checkpoint-selection rule before any TE policy rollout is permitted.

### 8.14 Week 9: evaluation-bundle freeze and TE unlock

**Objective:** protect the final test from post-hoc model selection.

SV3 must complete the guard conditions in `run_benchmark()`: verify model and input checksums, verify the frozen normalizer, require deterministic inference, reject an active training callback and refuse TE when the freeze marker is missing. Compare final candidates on TR+VA only.

After SV1 and SV2 approve the model and simulation bundle, record the evaluation-bundle checksum and TE unlock decision. From this point onward, no reward, hyperparameter, route, model or normalizer change is allowed.

Implementation details:

- `run_benchmark()` must refuse TE when the freeze marker, model checksum, normalizer or manifest checksum is missing.
- Verify that evaluation uses `training=False`, `norm_reward=False` and deterministic prediction.
- Check that no training callback or optimizer operation is active in the benchmark process.
- Store the approval decision, checksums and unlock timestamp in a structured evaluation record.

### 8.15 Week 10: TE-01 to TE-04 benchmark management

**Objective:** supervise the first final benchmark and preserve a complete evidence trail.

SV3 must execute `evaluate_manifest()` and `save_evaluation_results()` for TE-01 through TE-04. Verify that every controller has exactly the same scenario/seed combinations and that policy inference is deterministic. Inspect output completeness, failed episodes, duplicate runs and missing metric fields.

The output is structured CSV/JSON results, a benchmark manifest and a completeness report. Failed episodes must remain visible in the report rather than being silently removed.

Implementation details:

- `evaluate_manifest()` must run Fixed-time, Actuated/Max-pressure, DQN and optional PPO against the identical scenario/seed pairs.
- `save_evaluation_results()` must use `csv` or `json`, preserve field names from `EpisodeMetrics` and write to a configurable output path.
- Validate duplicate controller/scenario/seed records and report missing pairs before aggregation.
- Keep raw output paths in metadata so a metric can be traced back to its SUMO files.

### 8.16 Week 11: stress-test parsing and exception analysis

**Objective:** complete the metric dataset for robustness and overload scenarios.

SV3 must parse TE-05/TE-06 and repeated TE-03 outputs. Calculate average waiting time, queue length, time loss, throughput, phase-switch rate, minimum-green violations, travel time, spillback/recovery and emissions. Identify outliers, incomplete runs and parser exceptions.

The output is the complete metric dataset and exception report. SV1 checks physical meaning and SV2 checks episode IDs, seeds and controller metadata.

Implementation details:

- Run `parse_tripinfo()` and `parse_emissions()` for every completed output directory and preserve parser errors with the run ID.
- Use `collect_episode_metrics()` to normalize environment info and parsed XML values into one schema.
- Separate missing data, incomplete vehicles, SUMO failures and controller failures in the exception report.
- Do not remove outliers without recording the rule and the affected scenario/seed.

### 8.17 Week 12: statistics, tables and report draft

**Objective:** convert the final records into auditable statistical evidence.

SV3 must compute mean, standard deviation, 95% confidence intervals and percentage changes against fixed-time. Report both per-scenario values and aggregate values. Use `interpret_results()` to flag throughput regressions, excess phase switching, minimum-green violations and cases where a reward improvement does not correspond to traffic improvement.

Generate tables, comparison plots, confidence-interval plots and the results/discussion draft. Every number must trace to a saved run, scenario, seed and commit.

Implementation details:

- Use `aggregate_metrics()` for per-controller/per-scenario summaries and `confidence_interval()` for seed-level uncertainty.
- Use `interpret_results()` to calculate percentage changes against fixed-time while retaining absolute values.
- Feed `plots.py` from saved CSV/JSON results rather than manually entered numbers.
- Include the number of seeds, failed episodes and confidence-interval method in every result caption.

### 8.18 Week 13: report, archive and presentation

**Objective:** deliver a defensible final research package.

SV3 must merge the report, slides and tables; check metric definitions, citations, limitations, calibration claims and test-set policy. Archive raw results, processed metrics, plots, evaluation configuration and checksums. Run the final presentation rehearsal and confirm that the claims are supported by the benchmark.

The output is the final report, slide deck, benchmark archive and submission checklist. SV1 and SV2 perform cross-review outside their primary areas.

Implementation details:

- Re-run the result-loading and aggregation command from the archived evaluation directory.
- Check that all tables use the same metric names and units as `standard_metric_names()`.
- Link every headline claim to a scenario-level result, baseline comparison and limitation statement.
- Archive evaluation YAML, checksums, raw output index, processed metrics and generated figures together.

### 8.19 December buffer: result verification and closure

During 01/12-06/12, SV3 checks missing or failed results, approves only necessary reruns, adds missing metrics and records every incident. SV1 performs simulation reruns and SV2 repairs technical pipeline issues.

During 07/12-13/12, SV3 rechecks tables and plots, finalizes the README and appendices, creates the final backup and writes the project closure record. No new algorithm or test-set analysis may be introduced.

For every buffer correction, SV3 must preserve the previous result, record the reason for the change, rerun the affected parser/statistical check and update the result index without changing the frozen model-selection decision.

### 8.20 SV3 acceptance checklist

- [ ] Manifest validation catches duplicate IDs and missing routes.
- [ ] Train, validation and test splits are disjoint.
- [ ] TE files are locked before model selection.
- [ ] Fixed-time and heuristic baselines use matched routes/seeds.
- [ ] Tripinfo and emissions parsers reject malformed required data.
- [ ] Metrics include primary and constraint fields.
- [ ] Confidence intervals are calculated over independent seeds.
- [ ] Results are saved in CSV/JSON.
- [ ] Benchmark refuses to run unlocked TE evaluation.
- [ ] Report includes per-scenario and aggregate results.

---

## 9. Weekly Work Allocation

The weekly matrix below is the primary work-allocation view. Each student should first read their column for the current week, then use the detailed tables below it for the exact functions, files and acceptance checks.

The dates follow the 13-week plan beginning on 31/08/2026. Every week has one shared objective, one accountable owner for each workstream and a completion gate. A task is complete only when its output exists and the assigned reviewer has checked it.

### 9.1 Master weekly allocation matrix

| Week                  | Shared objective                               | SV1 - Simulation and data                                             | SV2 - Environment and DRL                                                              | SV3 - Baselines and evaluation                                  | Completion gate                                             |
| --------------------- | ---------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| 1                     | Freeze project contracts and define DEV-00     | Inspect network, map approaches, identify TLS and draft vehicle types | Freeze MDP, observation shape, action mapping and fake environment                     | Define manifest, survey form, splits, seeds and metrics         | Network/control scope and MDP contract approved             |
| 2                     | Connect SUMO to Gymnasium and run the pilot    | Implement route generation/validation and create DEV-00, TR-01, TR-02 | Implement environment factory, observations, rewards, wrappers and DQN pilot           | Implement manifest loading and fixed-time DEV-00 baseline       | Reset/step/close, baseline and checkpoint reload work       |
| 3                     | Calibrate and lock scenario splits             | Calibrate traffic/vehicle behavior and generate TR/VA/TE routes       | Implement sampler and package calibrated DQN pilot                                     | Validate manifest, create checksums and lock TE                 | Second member reproduces DEV-00 and baseline; TE is unused  |
| 4                     | Start repeatable TR-only training              | QA route flow, vehicle classes, teleporting and output paths          | Implement vectorization, TR sampler, DQN construction/training and checkpoint callback | Implement smoke test, initial evaluation runner and seed matrix | Training logs scenario ID and all seeds; only TR is sampled |
| 5                     | Cover the training domain and finish baselines | Validate demand, vehicle-mix, behavior profiles and emissions         | Continue DQN training, schedules and traffic callbacks                                 | Complete fixed-time and actuated/max-pressure evaluation        | All controllers emit the same metric schema                 |
| 6                     | Select the first DQN checkpoint using VA       | Confirm reward/metric measurements and simulation sensitivity         | Finalize reward v1, checkpoint loading and frozen VA normalization                     | Implement XML parsers, metric collection, aggregation and CI    | M2 is selected from VA only; TE remains locked              |
| 7                     | Add emissions evidence and optional PPO        | Collect emissions by scenario and vehicle class                       | Implement/train PPO only after DQN is stable                                           | Compare DQN/PPO/baselines on VA and start ablation matrix       | PPO can be dropped without affecting DQN deliverable        |
| 8                     | Tune only on VA and prepare test artifacts     | Syntax-check and checksum TE files without policy execution           | Run declared VA-only tuning and inference checks                                       | Complete VA selection protocol and benchmark configuration      | Selection criteria are written before TE results            |
| 9                     | Freeze the evaluation bundle and unlock TE     | Freeze simulation inputs and compute artifact checksums               | Freeze model, normalizer, config and inference script                                  | Add benchmark guards, sign bundle and unlock TE                 | No model, reward, route or normalizer changes after unlock  |
| 10                    | Run deterministic TE-01 to TE-04 benchmark     | Run matched simulations and preserve raw SUMO outputs                 | Run frozen deterministic inference and latency logging                                 | Execute benchmark runner and completeness checks                | Failed episodes are recorded, not hidden                    |
| 11                    | Run robustness and stress tests                | Run TE-05/TE-06 and repeated TE-03 stress tests                       | Analyze recovery/failures and fix only technical runner issues                         | Parse all outputs and produce the complete exception report     | Reruns use the same frozen bundle and are documented        |
| 12                    | Produce auditable statistics and report draft  | Create network, calibration and queue figures                         | Write MDP, reward, model and reproducibility methods                                   | Compute CI, comparisons, tables, plots and discussion           | Every number traces to scenario, seed and commit            |
| 13                    | Reproduce, document and submit                 | Finalize SUMO/network/data instructions                               | Finalize README, configs, checkpoints and demo                                         | Finalize report, slides, archive and submission checklist       | A second member reproduces the short pipeline               |
| Buffer 1: 01/12-06/12 | Repair missing or invalid existing results     | Rerun failed simulations and missing seeds                            | Repair critical pipeline/checkpoint defects                                            | Check completeness and approve reruns                           | No new algorithm or scope expansion                         |
| Buffer 2: 07/12-13/12 | Archive and finalize deliverables              | Complete technical appendices and backups                             | Complete code/config archive                                                           | Recheck tables, README, slides and closure record               | Final archive is verified and backed up                     |

### 9.2 Weekly operating rules

For every week, use this sequence:

1. **Start of week:** each student confirms the inputs needed from the previous week.
2. **During the week:** each student works only in their owned files and records decisions in `docs/decisions.md`.
3. **Before the weekly meeting:** the owner attaches the output path, command used, seed and commit hash.
4. **Weekly review:** the assigned reviewer reruns the smallest useful check and records pass/fail plus an issue note.
5. **Gate decision:** the group either accepts the week, records a blocker and recovery action, or moves only the non-blocked work forward.

The detailed weekly sections below define the implementation-level actions behind this matrix.

### Week 1: project contracts and development scenario

**Objective:** agree on what is being simulated and make a small DEV-00 case runnable on paper and in the repository.

| Owner | Detailed work                                                                                                                                                                                                                                                    | Required output                                                                                                         | Reviewer                                                                              |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| SV1   | Inspect the existing dummy network; decide whether it is DEV-only; map A1-A7 or the actual approach IDs; record TLS/junction IDs, lanes and controlled links; preserve OSM/network provenance; draft vehicle types for motorcycle, passenger car, bus and truck. | `data/processed/approach_map.csv`, network inspection note, initial `vehicle_types.add.xml`, DEV-00 route/config draft. | SV2 checks that the proposed observation/action data can be read from SUMO.           |
| SV2   | Freeze the first MDP draft; define observation feature order, dtype, shape and bounds; define discrete actions; implement the first observation-space methods and a lightweight fake DEV environment; prepare SB3 `check_env` test structure.                    | Updated `docs/mdp_descriptions.md`, observation contract, fake environment, initial `tests/test_env_gym.py`.            | SV1 checks SUMO assumptions; SV3 checks that features can become reported metrics.    |
| SV3   | Define manifest columns, split policy, seed policy, survey form, metric names and baseline names; create the first scenario table with DEV/TR/VA/TE rows; document privacy and field-survey rules.                                                               | `scenarios/scenario_manifest.csv` draft, survey form, metric table, `docs/decisions.md` draft.                          | SV1 checks scenario meaning; SV2 checks that fields can be passed to the environment. |

**End-of-week meeting:** approve target network/control scope, approach ordering, `delta_time`, yellow time, minimum green, state/action shape and the meaning of throughput.

**Gate:** no official implementation or long training starts until these decisions are recorded.

### Week 2: route generation, environment integration and pilot

**Objective:** connect a real SUMO episode to Gymnasium and prove that the whole pipeline can run for a short time.

| Owner | Detailed work                                                                                                                                                                                                                                                     | Required output                                                                                   | Reviewer                                                                             |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| SV1   | Implement `generate_random_trips()`, `count_realized_flow()` and `validate_route_file()`; implement the first `generate_custom_flow()` path; produce DEV-00, TR-01 and TR-02; run route and SUMO config checks; complete at least one survey session if possible. | Valid route files, route validation reports, realized-flow reports and raw/processed survey data. | SV3 checks flow totals and scenario metadata; SV2 checks route/config compatibility. |
| SV2   | Implement `create_sumo_env()`, `make_dev_environment()`, `close_environment()`, observation methods, first reward functions and wrappers; connect SUMO-RL/TraCI; run random actions; implement and run `run_dqn_pilot()`.                                         | Working reset/step/close loop, smoke-test log, pilot checkpoint, reload test and no-NaN report.   | SV1 checks SUMO lifecycle; SV3 checks that the pilot is labelled development-only.   |
| SV3   | Implement `ScenarioManifest.from_csv()`, `records_for_split()`, `get()`, `load_manifest()` and initial scenario selection; implement fixed-time environment/episode; review survey data and audit that pilot does not access TE.                                  | Validated manifest, fixed-time DEV-00 result, survey review note and leakage audit.               | SV2 checks runner compatibility; SV1 checks route and demand meaning.                |

**End-of-week meeting:** run DEV-00 from a clean command sequence, compare random-action and fixed-time output, reload the DQN checkpoint and inspect one trajectory.

**Gate:** reset, step, random action, fixed-time baseline, DQN pilot and checkpoint reload all work.

### Week 3: calibration, scenario generation and split lock

**Objective:** create the first complete scenario set and freeze the rules that protect the test set.

| Owner | Detailed work                                                                                                                                                                                                                                                               | Required output                                                                                                         | Reviewer                                                                                     |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| SV1   | Calibrate demand, turning ratios, speed, headway and queue discharge; tune initial vehicle behavior and Sublane settings; implement `generate_scenario_set()` and `write_scenario_report()`; generate TR-01 to TR-07, VA-01/VA-02 and TE-01 to TE-06; syntax-check TE only. | Calibration note, generated routes, realized-flow tables, vehicle-mix report and SUMO tripinfo/emissions output sample. | SV3 checks calibration/validation separation; SV2 checks observation and reward assumptions. |
| SV2   | Update environment mapping after calibration; implement `ScenarioSampler.sample()`, `sample_route()` and `set_split()`; complete DQN pilot checks; package the first calibrated DQN run and improve checkpoint metadata.                                                    | TR-only sampler, calibrated DEV/TR checkpoint, replay/normalizer/resume artifacts and updated environment contract.     | SV1 checks route selection; SV3 checks that sampler cannot access TE.                        |
| SV3   | Complete manifest validation and checksums; implement `select_scenario()` and `generate_selected_scenario()`; run fixed-time on matched DEV/TR/VA inputs; finish parser/metric sanity fixtures; lock TE manifest.                                                           | Locked manifest, checksum file, baseline dataset, parser fixtures and leakage audit.                                    | SV1 checks route IDs; SV2 checks metric availability.                                        |

**End-of-week meeting:** verify that a second member can run DEV-00, fixed-time and DQN using the README. Review every assumption without survey evidence.

**Gate:** TE is locked and has not been used for reward, checkpoint or hyperparameter selection.

### Week 4: training pipeline and TR sampler

**Objective:** replace the pilot path with a repeatable official training path.

| Owner | Detailed work                                                                                                                                                                                                             | Required output                                                                                  | Reviewer                                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| SV1   | QA all TR routes by approach, lane, vehicle class and departure count; check teleporting, invalid routes and output file collisions; benchmark one short batch run.                                                       | TR QA report and corrected route/config files.                                                   | SV3 checks realized demand; SV2 checks runtime and detector compatibility. |
| SV2   | Implement `make_vectorized_environment()`; connect `ScenarioSampler` to fresh one-route environments; implement `build_dqn_model()` and `train_dqn()`; fit VecNormalize only on TR; improve checkpoint callback metadata. | Official TR-only DQN command, vectorized environment, normalizer and periodic checkpoint bundle. | SV1 checks SUMO options; SV3 audits split and seed logs.                   |
| SV3   | Implement `check_environment()` and `run_smoke_test()`; implement the first `evaluate_controller()` path; prepare separate VA environment settings; define evaluation seed matrix.                                        | Automated smoke test, VA runner draft, seed matrix and audit log format.                         | SV2 checks Gym/SB3 behavior; SV1 checks metric interpretation.             |

**Gate:** an official training run logs scenario ID, demand seed and SUMO seed for every episode and samples only TR.

### Week 5: domain randomization and baseline completion

**Objective:** cover the calibrated training domain without leaking validation or test information.

| Owner | Detailed work                                                                                                                                                                                                       | Required output                                                     | Reviewer                                                                         |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| SV1   | Validate TR-01 to TR-07 demand ranges; test vehicle-mix and behavior-profile randomization; compare generated versus realized flow; verify emissions output on representative scenarios.                            | Domain-coverage table, flow deviation report and emissions sample.  | SV3 checks scenario distribution; SV2 checks environment stability.              |
| SV2   | Continue DQN training over TR; implement `linear_schedule()` and `cosine_schedule()` if required; add `TrafficMetricsCallback`; monitor reward, queue, waiting and phase switches; investigate NaN or stalled runs. | DQN training curves, callback logs, schedule tests and failure log. | SV1 checks traffic behavior; SV3 checks whether reward hides metric regressions. |
| SV3   | Complete `evaluate_fixed_time()` and `run_fixed_time_episode()`; implement or finalize actuated/max-pressure baseline; verify all controllers use the same route and seed; review benchmark output schema.          | Baseline runner and matched DEV/TR/VA baseline results.             | SV1 checks controller meaning; SV2 checks interface compatibility.               |

**Gate:** every controller produces the same `EpisodeMetrics` fields and the TR domain is represented in training logs.

### Week 6: reward v1, VA evaluation and DQN milestone

**Objective:** select the first DQN checkpoint using validation only.

| Owner | Detailed work                                                                                                                                                                                                           | Required output                                                                            | Reviewer                                                           |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| SV1   | Confirm queue, speed, headway and emissions measurements used by reward and metrics; run sensitivity checks on vType/Sublane settings; fix only simulation or data defects.                                             | Calibration validation note and confirmed measurement definitions.                         | SV3 checks reported units; SV2 checks reward inputs.               |
| SV2   | Finalize `combined_reward()` and switch penalty; complete `save_checkpoint()`/`load_checkpoint()` validation; run DQN with reward v1; evaluate without updating VecNormalize; preserve the best checkpoint by VA score. | M2 DQN checkpoint, frozen VA normalizer, resume metadata and training/evaluation commands. | SV1 checks traffic behavior; SV3 selects checkpoint using VA only. |
| SV3   | Implement `parse_tripinfo()`, `parse_emissions()`, `merge_tripinfo_metrics()`, `standard_metric_names()` and `collect_episode_metrics()`; implement `confidence_interval()` and first `aggregate_metrics()`.            | Parser tests, metric CSV/JSON fixture, VA summaries and checkpoint-selection record.       | SV1 checks XML units; SV2 checks info fields.                      |

**Gate:** M2 is selected only from VA-01/VA-02. TE remains unopened.

### Week 7: PPO branch and emissions integration

**Objective:** extend the stable DQN pipeline without risking the minimum deliverable.

| Owner | Detailed work                                                                                                                                              | Required output                                                     | Reviewer                                                            |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| SV1   | Collect emissions by scenario and vehicle class on TR; verify fuel/CO2/NOx aggregation; prepare plots or tables showing mixed-traffic effects.             | Emissions dataset and unit conversion note.                         | SV3 checks aggregation; SV2 checks runtime overhead.                |
| SV2   | Implement `build_ppo_model()`, `train_ppo()`, save/load adapters and PPO configuration; train PPO only on TR; continue DQN best run if needed.             | PPO pilot/checkpoint, comparison of DQN and PPO training stability. | SV1 checks environment compatibility; SV3 checks no VA/TE training. |
| SV3   | Extend `evaluate_manifest()` for VA; compare DQN/PPO/fixed-time/actuated on VA; begin reward and algorithm ablation matrix; audit that TE is still locked. | VA comparison table, ablation plan and leakage audit.               | SV2 validates checkpoint loading; SV1 validates traffic metrics.    |

**Gate:** PPO may be dropped if it threatens DQN completion or does not produce a stable VA result.

### Week 8: controlled tuning and test preparation

**Objective:** make limited decisions from VA and prepare test artifacts without looking at policy results on TE.

| Owner | Detailed work                                                                                                                                                                                  | Required output                                                             | Reviewer                                                                        |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| SV1   | Validate TE route XML, network references, checksums and metadata without running policies; optimize SUMO settings if TR training is too slow; prepare stress-test commands.                   | TE syntax/consistency report and performance note.                          | SV3 confirms no TE rollout occurred; SV2 checks runtime changes.                |
| SV2   | Run a small, pre-declared hyperparameter search over DQN/PPO using VA only; try LSTM only if budget remains; test checkpoint reload and deterministic inference.                               | Candidate model table, selected hyperparameters and inference smoke result. | SV3 confirms search used only VA; SV1 checks no simulation assumptions changed. |
| SV3   | Check coverage of TR and VA; complete `interpret_results()` draft; verify early stopping rules, confidence intervals and failure handling; prepare final evaluation config without opening TE. | Evaluation config draft, VA selection protocol and final audit checklist.   | SV2 checks artifact paths; SV1 checks scenario labels.                          |

**Gate:** all model-selection criteria are written before TE results are available.

### Week 9: ablation, model freeze and TE unlock

**Objective:** produce an immutable evaluation bundle.

| Owner | Detailed work                                                                                                                                                                 | Required output                                                                | Reviewer                                                              |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| SV1   | Run vType/no-Sublane/Sublane ablations on TR+VA; freeze network, demand and vehicle-type files; compute checksums for evaluation inputs.                                      | Ablation results, frozen simulation bundle and checksums.                      | SV3 checks ablation split; SV2 checks compatibility with final model. |
| SV2   | Select final DQN/PPO model using VA; freeze checkpoint, configuration, normalizer and inference script; record git commit and environment versions; do not tune after freeze. | Final model bundle, normalizer, config, resume metadata and inference package. | SV1 checks environment inputs; SV3 signs model freeze.                |
| SV3   | Complete `run_benchmark()` guard conditions; compare candidate models on TR+VA; store evaluation-bundle checksum; formally unlock TE only after all signatures/checks pass.   | Signed evaluation bundle, locked benchmark config and TE unlock record.        | SV1 and SV2 both approve.                                             |

**Gate:** no hyperparameter, reward, route or normalizer change is allowed after TE unlock.

### Week 10: deterministic benchmark on TE-01 to TE-04

**Objective:** run the first final benchmark with exactly matched inputs.

| Owner | Detailed work                                                                                                                                                  | Required output                                            | Reviewer                                                       |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| SV1   | Run batch simulations for TE-01 to TE-04 across every controller and required seed; preserve raw tripinfo/emissions/logs; record failures and SUMO exit codes. | Immutable raw benchmark outputs and run manifest.          | SV3 checks route/seed matching; SV2 checks technical failures. |
| SV2   | Load final model and VecNormalize with evaluation mode; run deterministic inference; record action latency and episode failures; fix only runner defects.      | Inference logs, latency data and technical incident notes. | SV3 confirms no model change; SV1 checks SUMO process cleanup. |
| SV3   | Execute `evaluate_manifest()` and `save_evaluation_results()`; check every controller has the same scenario/seed matrix; inspect per-scenario metrics.         | Raw results CSV/JSON and completeness report.              | SV1 reviews traffic values; SV2 reviews schema.                |

**Gate:** incomplete or failed episodes are reported, not silently removed.

### Week 11: TE-05, TE-06 and stress testing

**Objective:** evaluate robustness to vehicle-mix shift, replay data and overload.

| Owner | Detailed work                                                                                                                                                          | Required output                                                | Reviewer                                                         |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| SV1   | Run TE-05/TE-06 where available; repeat TE-03 over multiple seeds; produce trajectories, queue/spillback observations and recovery data.                               | Stress-test raw data, trajectory files and recovery summaries. | SV3 checks test isolation; SV2 checks technical reproducibility. |
| SV2   | Analyze failure and recovery behavior; measure inference latency; fix only technical execution errors; never alter the frozen model.                                   | Failure taxonomy, latency summary and incident log.            | SV3 approves whether a rerun is technically justified.           |
| SV3   | Parse all tripinfo/emissions; compute waiting, timeLoss, queue, throughput, phase switches, min-green violations and emissions; identify outliers and missing records. | Complete metric dataset and exception report.                  | SV1 checks physical interpretation; SV2 checks episode metadata. |

**Gate:** all reruns must use the same frozen model and manifest, with the reason recorded.

### Week 12: statistics, plots and report draft

**Objective:** turn raw benchmark outputs into auditable evidence.

| Owner | Detailed work                                                                                                                                                     | Required output                                                              | Reviewer                                    |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------- |
| SV1   | Create SUMO/network figures, approach diagrams, calibration visuals, queue heatmaps and representative failure snapshots.                                         | Figure inputs and technical figure captions.                                 | SV3 checks that figures use raw saved data. |
| SV2   | Document MDP, observation, action safety, reward, DQN/PPO settings, checkpointing and inference procedure; clean configuration paths.                             | Methods section draft and reproducible configuration bundle.                 | SV1 checks SUMO details; SV3 checks claims. |
| SV3   | Compute mean, SD, 95% CI and percentage change versus fixed-time; run `interpret_results()`; produce per-scenario and aggregate tables; write results/discussion. | Final metric tables, plots, statistical report draft and limitation section. | SV1 and SV2 cross-review.                   |

**Gate:** every reported number can be traced to a saved run, scenario, seed and commit.

### Week 13: reproduction, documentation and submission

**Objective:** make the project reproducible from a clean environment and package the final deliverables.

| Owner | Detailed work                                                                                                                                                      | Required output                                                         | Reviewer                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------- |
| SV1   | Finalize SUMO setup instructions, network input documentation, route-generation instructions and data assumptions; remove accidental local paths.                  | SUMO setup guide, input archive and final simulation README section.    | SV2 runs the instructions from a clean environment. |
| SV2   | Finalize README, requirements/config examples, pilot and inference commands; test checkpoint loading and demo inference; run the complete short pipeline.          | Reproducible code/config bundle, demo command and final model artifact. | SV3 verifies artifact list and metadata.            |
| SV3   | Merge report, slides and tables; check citations, metric definitions, limitations and claims; archive raw results and checksums; run final presentation rehearsal. | Final report, slides, benchmark archive and submission checklist.       | Entire team signs off.                              |

**Final gate:** a second member can reproduce at least DEV-00, one baseline episode, one deterministic model episode and the metric aggregation from the README.

### December buffer: 01/12/2026-13/12/2026

Use the buffer only for work that protects or completes the existing result:

| Period      | Allowed work                                                                                                                                | Owner                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 01/12-06/12 | Rerun failed seeds/episodes; add missing metrics; repair critical SUMO, parser or batch-runner defects; restore from checkpoint when valid. | SV1 runs simulations, SV2 repairs pipeline, SV3 checks completeness. |
| 07/12-13/12 | Recheck tables and plots; finalize README, appendices and slides; create final backups and closure record.                                  | SV3 coordinates, SV1/SV2 complete technical documentation.           |

Do not add a new algorithm, expand the scenario scope or tune the frozen model during the buffer. Prioritize failures that invalidate results, missing baseline/test seeds, incorrect tables/plots and only then formatting changes.

---

## 10. Test and Validation Strategy

### 10.1 Unit tests

Add focused tests for:

- malformed manifest rows;
- duplicate scenario IDs;
- invalid split;
- missing route files;
- deterministic scenario sampling;
- route XML/type validation;
- realized flow counting;
- vehicle mix validation;
- observation shape and dtype;
- observation bounds;
- reward sign and finite values;
- switch penalty behavior;
- metric collection with missing fields;
- tripinfo/emission parsing;
- confidence intervals;
- CSV/JSON round-trip;
- checkpoint companion file paths.

### 10.2 Environment contract test

`tests/test_env_gym.py` should create a lightweight DEV-00 environment and verify:

```python
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Check that:

- the observation belongs to `observation_space`;
- the action belongs to `action_space`;
- reward is finite;
- the five return values have the correct types;
- the episode can close without leaving SUMO running.

### 10.3 Network test

`tests/test_net.py` should verify:

- network file exists;
- expected junction/TLS exists;
- approach edges exist;
- no required route edge is missing;
- vehicle types referenced by routes exist;
- no critical lane is disconnected.

### 10.4 Smoke test

`src/traffic_drl/evaluation/test.py`:

#### `check_environment()`

Run Stable-Baselines3's environment checker against the lightweight DEV environment. If SUMO is unavailable, the test may use a clearly marked fake fixture, but the real SUMO smoke test must still run before training.

#### `run_smoke_test()`

Run reset and random legal actions for a small number of steps. Return `SmokeTestResult` with:

- number of steps;
- termination flags;
- rewards;
- final info.

Reject NaN/inf rewards and malformed transitions.

### 10.5 Integration order

Run validation in this order:

```
1. pytest unit tests
2. XML and manifest validation
3. SUMO route/config validation
4. DEV-00 random-action smoke test
5. fixed-time DEV-00 episode
6. DQN pilot and checkpoint reload
7. short TR training
8. VA evaluation with frozen normalizer
9. only after freeze: TE benchmark
```

Suggested commands after the implementation is in place:

```bash
python -m pytest -q
python -m traffic_drl.evaluation.test
python -m scenarios.generators.scenario_factory --help
```

Use the project's configured virtual environment and ensure `SUMO_HOME` is available before real SUMO tests.

---

## 11. Definition of Done

The implementation is complete only when all of the following are true:

### Data and simulation

- [ ] Network, vehicle types and route files are reproducible.
- [ ] Vehicle mix and behavior assumptions are documented.
- [ ] Realized flow is measured rather than inferred only from `randomTrips.py --period`.
- [ ] Tripinfo and emissions outputs are available and parseable.

### Environment and learning

- [ ] Observation and action contracts are frozen.
- [ ] Minimum green and yellow clearance are enforced.
- [ ] DQN pilot runs and reloads.
- [ ] Official DQN trains only on TR.
- [ ] VecNormalize is frozen for VA/TE.
- [ ] Checkpoint bundles contain model, normalizer, config and resume metadata.

### Evaluation

- [ ] Fixed-time and actuated/heuristic baselines run on matched inputs.
- [ ] TE is locked until model freeze.
- [ ] Every episode has a typed metric record.
- [ ] Results include per-scenario values, means, standard deviations and 95% confidence intervals.
- [ ] Throughput and safety constraints are reported alongside reward and delay.
- [ ] Failed episodes and stress-test failures are included rather than hidden.

### Reproducibility

- [ ] A clean environment can reproduce DEV-00.
- [ ] A second team member can run the README commands.
- [ ] Every result identifies commit, seed, scenario, controller and configuration.
- [ ] Raw data is preserved separately from processed data.
- [ ] The final evaluation bundle has a checksum.

---

## 12. Scope Control

The minimum acceptable scientific system is:

```
mixed-traffic SUMO network
- DEV/TR/VA/TE scenarios
- fixed-time baseline
- DQN controller
- traffic metrics
- reproducible checkpoint and benchmark
```

PPO, LSTM, broad hyperparameter searches and additional algorithms are secondary. If time becomes limited, keep the following in this order:

1. network and route validity;
2. MDP/environment contract;
3. fixed-time baseline;
4. DQN end-to-end training;
5. matched validation/test evaluation;
6. metrics and report;
7. PPO;
8. LSTM and extra ablations.

Do not remove baseline comparisons, test seeds or core metrics to make room for advanced models.
