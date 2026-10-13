import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Any, Mapping

# Assuming json_comment_remove is in the same directory
from json_comment_remove import load_jsonc

def convert_definitions_to_sumo(
    json_def_path: Path | str,
    output_add_path: Path | str
) -> Path:
    """Converts definition.jsonc into a reusable SUMO additional file (<vType> definitions)."""
    out_path = Path(output_add_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("additional", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd"
    })

    v_data = load_jsonc(json_def_path)
    for v_id, attrs in v_data.items():
        xml_attrs = {k: str(v) for k, v in attrs.items()}
        xml_attrs["id"] = v_id
        
        v_elem = ET.SubElement(root, "vType", xml_attrs)
        
        # Bluelight parameters for the ambulance
        if v_id == "hyundai_solati_ambulance":
            ET.SubElement(v_elem, "param", {"key": "has.bluelight.device", "value": "true"})
            ET.SubElement(v_elem, "param", {"key": "device.bluelight.reactiondist", "value": "40.0"})

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    out_path.write_text(xml_str, encoding="utf-8")
    return out_path


def generate_custom_flow(
    *,
    scenario_id: str,
    output_route_file: str | Path,
    demand_by_approach: Mapping[str, float],
    turning_ratios: Mapping[str, float],
    vehicle_mix: Mapping[str, float],
    begin: int,
    end: int,
    seed: int,
    flow_pattern: str = "uniform",
    behavior_profiles: Mapping[str, str] | None = None
) -> Path:
    out_path = Path(output_route_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("routes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
    })

    base_dir = Path(__file__).resolve().parent.parent / "json_definitions"

    # 1. Distribution references the external vTypes defined in vtypes.add.xml
    # SUMO expects space-separated strings for existing vTypes and probabilities
    vtypes_str = " ".join(vehicle_mix.keys())
    probs_str = " ".join(str(prob) for prob in vehicle_mix.values())
    
    ET.SubElement(root, "vTypeDistribution", {
        "id": scenario_id,
        "vTypes": vtypes_str,
        "probabilities": probs_str
    })

    # 2. Load Routes from routes.jsonc
    route_data = load_jsonc(base_dir / "routes.jsonc")
    for r_id, edge_list in route_data.get("routes", {}).items():
        ET.SubElement(root, "route", {"id": r_id, "edges": " ".join(edge_list)})

    # 3. Generate Flows
    for approach, vehs_per_hour in demand_by_approach.items():
        for mov_id, ratio in turning_ratios.items():
            if mov_id.startswith(approach):
                flow_rate = vehs_per_hour * ratio
                if flow_rate > 0:
                    ET.SubElement(root, "flow", {
                        "id": f"flow_{mov_id}_{scenario_id}",
                        "type": scenario_id,
                        "route": f"r_{mov_id}",
                        "begin": str(begin),
                        "end": str(end),
                        "vehsPerHour": f"{flow_rate:.1f}",
                        "departLane": "best",
                        "departPos": "base",
                        "departPosLat": "random",
                        "departSpeed": "last"
                    })

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    out_path.write_text(xml_str, encoding="utf-8")
    return out_path