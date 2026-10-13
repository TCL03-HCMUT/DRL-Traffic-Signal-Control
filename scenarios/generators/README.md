# Scenario generation

This folder contains the scripts used to turn the project’s scenario definitions into runnable SUMO route files and configuration files. The workflow is driven by JSONC manifests under `scenarios/json_definitions`, then converted into XML output under `scenarios/sumo_definitions`.

## What is generated

The generator pipeline produces:

- `.rou.xml` route files for each scenario
- `.sumocfg` files that reference the network, route file, and additional vehicle definitions
- `vtypes.add.xml`, which contains the reusable SUMO vehicle type definitions for cars, motorcycles, buses, trucks, and the emergency vehicle

The default entry point is `scenarios/generators/scenario_factory.py`.

## Input files

The generation pipeline reads these files:

- `scenarios/json_definitions/definition.jsonc`  
  Vehicle profiles (`vType` definitions): dimensions, acceleration, deceleration, max speed, color, lane behavior, etc.
- `scenarios/json_definitions/routes.jsonc`  
  Predefined edge sequences for each turning movement, such as `A1_to_A2`, `A3_to_A4`, etc.
- `scenarios/json_definitions/scenario_manifest.jsonc`  
  Scenario parameters: demand by approach, turning ratios, and vehicle mix.
- `sumo/net/official/roundabout.net.xml`  
  The SUMO road network used for all generated scenarios.

## Generation flow

`scenario_factory.py` follows this order:

1. Reads the scenario manifest to choose a scenario.
2. Converts `definition.jsonc` into a SUMO `additional` file with `<vType>` entries.
3. Builds an XML route file using `routes.jsonc` and the selected demand/turning ratios.
4. Writes a `.sumocfg` file that links the network, route file, and additional definitions.
5. Saves all outputs into `scenarios/sumo_definitions/`.

The actual flow construction happens in `custom_flow_gen.py`:

- Each demand value is treated as an hourly arrival rate on an approach.
- For each movement from that approach, the script multiplies the hourly demand by the movement ratio.
- The result becomes a `flow` element with `vehsPerHour` in the SUMO route file.

Example:

- Approach `A1` demand = `4200` vehicles/hour
- Movement `A1_to_A2` ratio = `0.38`
- Flow rate = `4200 * 0.38 = 1596` vehicles/hour

## Generate a single scenario

From the project root, run:

```bash
python scenarios/generators/scenario_factory.py --scenario 01_morning_peak_inbound
```

This creates:

- `scenarios/sumo_definitions/01_morning_peak_inbound.rou.xml`
- `scenarios/sumo_definitions/01_morning_peak_inbound.sumocfg`
- `scenarios/sumo_definitions/vtypes.add.xml`

To generate a different scenario, replace the scenario ID with any key from `scenario_manifest.jsonc`, for example:

```bash
python scenarios/generators/scenario_factory.py --scenario 02_evening_rush_outbound
```

## Generate a selected scenario

You can generate a specific scenario by passing its ID to the helper script:

```bash
./scripts/generate_all_scenarios.sh 01_morning_peak_inbound
```

```bat
scripts\generate_all_scenarios.bat 01_morning_peak_inbound
```

If no scenario ID is passed, the script prints the available scenario names from the manifest and exits.

## Generate scenarios from a CSV file

The project also includes a CSV-based selection file at `scenarios/json_definitions/selected_scenario.csv`. To generate every scenario listed there, pass the CSV path:

```bash
./scripts/generate_all_scenarios.sh --csv scenarios/json_definitions/selected_scenario.csv
```

```bat
scripts\generate_all_scenarios.bat --csv scenarios\json_definitions\selected_scenario.csv
```

This reads the `scenario_id` column and invokes the generator for each entry in order.

## Generate all scenarios

The manifest contains multiple predefined scenarios. You can loop through them with a small shell or Python snippet:

```bash
for scenario in 01_morning_peak_inbound 02_evening_rush_outbound 03_midday_balanced_lunch 04_afternoon_school_commute 05_late_night_freeflow; do
  ./scripts/generate_all_scenarios.sh "$scenario"
done
```

On Windows:

```bat
for %%S in (01_morning_peak_inbound 02_evening_rush_outbound 03_midday_balanced_lunch 04_afternoon_school_commute 05_late_night_freeflow) do (
    scripts\generate_all_scenarios.bat %%S
)
```

## Customizing the scenario

Edit the scenario definition files to change traffic behavior:

### 1. Demand by approach

In `scenario_manifest.jsonc`, update the `demand_by_approach` object to change how many vehicles arrive from each approach.

Example:

```json
"demand_by_approach": {
  "A1": 4200,
  "A2": 1800,
  "A3": 2600,
  "A4": 2100,
  "A6": 2400
}
```

### 2. Turning ratios

Adjust the `turning_ratios` values to change where vehicles go after entering the roundabout.

Example:

```json
"A1_to_A2": 0.38,
"A1_to_A5": 0.42
```

The ratios for a single source approach should sum to approximately 1.

### 3. Vehicle mix

Update `vehicle_mix` to change the proportion of passenger cars, motorcycles, buses, trucks, and emergency vehicles.

Example:

```json
"vehicle_mix": {
  "gas_motorbike": 0.65,
  "electric_motorbike": 0.24,
  "car": 0.04,
  "MPV": 0.045,
  "bus": 0.015,
  "truck": 0.008,
  "ambulance": 0.002
}
```

These values are used as probabilities for the SUMO `vTypeDistribution` in the generated route file.

## Validation and usage

After generation, you can run the scenario in SUMO with the generated `.sumocfg` file:

```bash
sumo -c scenarios/sumo_definitions/01_morning_peak_inbound.sumocfg
```

If you want to inspect or debug the route generation logic, the helper script `random_trips_gen.py` is also included, although the project’s main scenario builder uses the manifest-driven generator described above.

## Notes

- The generator expects the repo root to be the working directory when you launch the script.
- The JSON files are stored as `.jsonc` (JSON with comments), so they can include comments for documentation without breaking parsing.
- `json_comment_remove.py` strips comments before the data is loaded into Python.

This makes the scenario generation process easy to tune without editing raw XML by hand.
