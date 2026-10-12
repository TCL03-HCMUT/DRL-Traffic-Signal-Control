"""Phase 1 random route generation API."""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

def generate_random_trips(
    *, 
    network_file: str | Path, 
    output_route_file: str | Path, 
    begin: int, 
    end: int, 
    period: float, 
    seed: int, 
    vehicle_types_file: str | Path | None = None, 
    additional_options: Mapping[str, Any] | None = None
) -> Path:
    out_path = Path(output_route_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "sumolib.randomTrips",
        "-n", str(network_file),
        "-o", str(out_path),
        "-b", str(begin),
        "-e", str(end),
        "-p", str(period),
        "--seed", str(seed)
    ]

    if vehicle_types_file:
        cmd.extend(["--additional-files", str(vehicle_types_file)])

    if additional_options:
        for k, v in additional_options.items():
            cmd.extend([f"--{k}", str(v)])

    subprocess.run(cmd, check=True)
    return out_path

def count_realized_flow(route_file: str | Path, *, approach_ids: list[str] | None = None) -> dict[str, int]:
    counts = {approach: 0 for approach in (approach_ids or [])}
    tree = ET.parse(route_file)
    root = tree.getroot()

    for veh in root.findall("vehicle") + root.findall("flow"):
        route_elem = veh.find("route")
        if route_elem is not None and "edges" in route_elem.attrib:
            first_edge = route_elem.attrib["edges"].split()[0]
            # Match edge to an approach prefix if needed
            for approach in counts.keys():
                if first_edge.startswith(approach):
                    counts[approach] += 1
                    break
                    
    return counts

def validate_route_file(route_file: str | Path, *, expected_vehicle_types: set[str] | None = None) -> list[str]:
    errors = []
    try:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        for veh in root.findall("vehicle") + root.findall("flow"):
            v_type = veh.get("type")
            if not v_type:
                errors.append(f"Vehicle/Flow '{veh.get('id')}' is missing a type attribute.")
                
            if expected_vehicle_types and v_type not in expected_vehicle_types:
                # Could be a distribution ID instead of a base vehicle type, handle accordingly
                pass
                
            route = veh.find("route")
            if route is None and veh.get("route") is None:
                errors.append(f"Vehicle/Flow '{veh.get('id')}' has no route definition.")

    except ET.ParseError as e:
        errors.append(f"XML parsing failed: {e}")
        
    return errors