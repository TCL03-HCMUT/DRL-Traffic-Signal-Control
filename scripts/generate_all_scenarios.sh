#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_CMD=()
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "Python interpreter not found in PATH." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Available scenarios:" >&2
  "${PYTHON_CMD[@]}" - "$PROJECT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
manifest_path = project_root / "scenarios" / "json_definitions" / "scenario_manifest.jsonc"
text = manifest_path.read_text(encoding="utf-8")
text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))
manifest = json.loads(text)
for key in manifest:
    print(key)
PY
  exit 1
fi

CSV_PATH=""
SCENARIO_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv|-c)
      if [[ $# -lt 2 ]]; then
        echo "Missing CSV path after $1" >&2
        exit 1
      fi
      CSV_PATH="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [scenario_id] | --csv path/to/scenarios.csv" >&2
      exit 0
      ;;
    *)
      if [[ -n "$SCENARIO_ID" ]]; then
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      SCENARIO_ID="$1"
      shift
      ;;
  esac
done

if [[ -n "$CSV_PATH" ]]; then
  if [[ ! -f "$CSV_PATH" ]]; then
    CSV_PATH="$PROJECT_ROOT/$CSV_PATH"
  fi
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV file not found: $CSV_PATH" >&2
    exit 1
  fi

  "${PYTHON_CMD[@]}" - "$CSV_PATH" "$PROJECT_ROOT" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

csv_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()

with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    scenario_ids = []
    for row in reader:
        sid = (row.get("scenario_id") or "").strip()
        if sid:
            scenario_ids.append(sid)

if not scenario_ids:
    raise SystemExit(f"No scenario_id values found in {csv_path}")

for scenario_id in scenario_ids:
    print(f"Generating scenario: {scenario_id}")
    subprocess.run([
        sys.executable,
        str(project_root / "scenarios" / "generators" / "scenario_factory.py"),
        "--scenario",
        scenario_id,
    ], check=True)
PY
  exit 0
fi

if [[ -z "$SCENARIO_ID" ]]; then
  echo "Usage: $0 [scenario_id] | --csv path/to/scenarios.csv" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" "$PROJECT_ROOT/scenarios/generators/scenario_factory.py" --scenario "$SCENARIO_ID"