"""Manifest-driven scenario selection and route generation system."""
from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Any, Sequence, Mapping

from custom_flow_gen import generate_custom_flow
from json_comment_remove import load_jsonc

import os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_sumocfg(
    net_file: Path | str,
    route_file: Path | str,
    add_file: Path | str,
    output_cfg: Path | str,
    project_root: Path,
    begin: int = 0,
    end: int = 3600,
    seed: int = 42
) -> Path:
    out_path = Path(output_cfg).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the directory where the .sumocfg will live
    cfg_dir = out_path.parent
    
    # Calculate paths strictly relative to the .sumocfg folder
    rel_net_path = os.path.relpath(Path(net_file).resolve(), cfg_dir)
    rel_route_path = os.path.relpath(Path(route_file).resolve(), cfg_dir)
    rel_add_path = os.path.relpath(Path(add_file).resolve(), cfg_dir)

    config = ET.Element("configuration")
    
    input_elem = ET.SubElement(config, "input")
    # .replace('\\', '/') ensures paths work perfectly in SUMO on Windows
    ET.SubElement(input_elem, "net-file", {"value": str(rel_net_path).replace('\\', '/')})
    ET.SubElement(input_elem, "route-files", {"value": str(rel_route_path).replace('\\', '/')})
    ET.SubElement(input_elem, "additional-files", {"value": str(rel_add_path).replace('\\', '/')})
        
    time_elem = ET.SubElement(config, "time")
    ET.SubElement(time_elem, "begin", {"value": str(begin)})
    ET.SubElement(time_elem, "end", {"value": str(end)})
    ET.SubElement(time_elem, "step-length", {"value": "0.1"})
    
    proc_elem = ET.SubElement(config, "processing")
    ET.SubElement(proc_elem, "seed", {"value": str(seed)})
    ET.SubElement(proc_elem, "lateral-resolution", {"value": "1.2"})
    ET.SubElement(proc_elem, "lanechange.duration", {"value": "0"})
    
    xml_str = minidom.parseString(ET.tostring(config)).toprettyxml(indent="    ")
    out_path.write_text(xml_str, encoding="utf-8")
    return out_path

from custom_flow_gen import generate_custom_flow, convert_definitions_to_sumo

def load_scenario_manifest(manifest_path: Path | str) -> dict[str, dict[str, Any]]:
    """Load a per-scenario manifest with vehicle mix, demand and turning ratios."""
    manifest = load_jsonc(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("Scenario manifest must be a dictionary keyed by scenario_id.")

    required_keys = {"demand_by_approach", "turning_ratios", "vehicle_mix"}
    for scenario_id, scenario_data in manifest.items():
        if not isinstance(scenario_data, dict):
            raise TypeError(f"Scenario '{scenario_id}' must map to a dictionary.")
        missing = sorted(required_keys - set(scenario_data.keys()))
        if missing:
            raise KeyError(
                f"Scenario '{scenario_id}' is missing required keys: {missing}"
            )
    return manifest


def build_random_scenario(
    project_root: Path,
    net_file_path: Path,
    scenario_id: str,
    demand_by_approach: Mapping[str, float],
    turning_ratios: Mapping[str, float],
    vehicle_mix: Mapping[str, float],
    seed: int = 42
) -> Path:
    base_dir = project_root / "scenarios"
    json_dir = base_dir / "json_definitions"
    sumo_dir = base_dir / "sumo_definitions"
    
    # 1. Convert JSON definition to separate SUMO additional file
    add_output = sumo_dir / "vtypes.add.xml"
    convert_definitions_to_sumo(
        json_def_path=json_dir / "definition.jsonc",
        output_add_path=add_output
    )
    
    # 2. Build route from the scenario-specific data
    route_output = sumo_dir / f"{scenario_id}.rou.xml"
    cfg_output = sumo_dir / f"{scenario_id}.sumocfg"
    
    generate_custom_flow(
        scenario_id=scenario_id,
        output_route_file=route_output,
        demand_by_approach=demand_by_approach,
        turning_ratios=turning_ratios,
        vehicle_mix=vehicle_mix,
        begin=0,
        end=3600,
        seed=seed
    )
    
    # 3. Generate SUMOCFG linking net, route, and the separate definitions file
    generate_sumocfg(
        net_file=net_file_path,
        route_file=route_output,
        add_file=add_output,
        output_cfg=cfg_output,
        project_root=project_root,
        begin=0,
        end=3600,
        seed=seed
    )
    
    print(f"Generated scenario: {scenario_id} at {sumo_dir}")
    return cfg_output

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a SUMO scenario from a manifest.")
    parser.add_argument("--scenario", default=None, help="Scenario id to generate from the manifest.")
    args = parser.parse_args(argv)

    # Resolves to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    net_path = project_root / "sumo" / "net" / "official" / "roundabout.net.xml"
    manifest_path = project_root / "scenarios" / "json_definitions" / "scenario_manifest.jsonc"
    
    # Check if network file exists before generating
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found at: {net_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found at: {manifest_path}")

    manifest = load_scenario_manifest(manifest_path)
    if not manifest:
        raise ValueError(f"Scenario manifest is empty: {manifest_path}")

    scenario_id = args.scenario or next(iter(manifest))
    if scenario_id not in manifest:
        valid = ", ".join(manifest.keys())
        raise KeyError(f"Unknown scenario '{scenario_id}'. Valid options: {valid}")
    scenario_data = manifest[scenario_id]

    build_random_scenario(
        project_root=project_root,
        net_file_path=net_path,
        scenario_id=scenario_id,
        demand_by_approach=scenario_data["demand_by_approach"],
        turning_ratios=scenario_data["turning_ratios"],
        vehicle_mix=scenario_data["vehicle_mix"],
        seed=42
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())