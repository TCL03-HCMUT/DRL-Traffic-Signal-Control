@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."

if "%~1"=="" (
    echo Available scenarios:
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import json, pathlib; p = pathlib.Path('scenarios/json_definitions/scenario_manifest.jsonc'); s = p.read_text(encoding='utf-8'); s = '\n'.join(line for line in s.splitlines() if not line.strip().startswith('//')); d = json.loads(s); print('\n'.join(d.keys()))"
    ) else (
        python -c "import json, pathlib; p = pathlib.Path('scenarios/json_definitions/scenario_manifest.jsonc'); s = p.read_text(encoding='utf-8'); s = '\n'.join(line for line in s.splitlines() if not line.strip().startswith('//')); d = json.loads(s); print('\n'.join(d.keys()))"
    )
    popd
    exit /b 1
)

if /I "%~1"=="--csv" (
    if "%~2"=="" (
        echo Missing CSV path after --csv
        popd
        exit /b 1
    )
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import csv, subprocess, sys; from pathlib import Path; p = Path(r'%~2'); rows = list(csv.DictReader(p.open('r', encoding='utf-8', newline=''))); ids = [row.get('scenario_id','').strip() for row in rows if (row.get('scenario_id') or '').strip()]; assert ids, 'No scenario_id values found in ' + str(p); [print('Generating scenario:', sid) or subprocess.run([sys.executable, str(Path.cwd()/'scenarios'/'generators'/'scenario_factory.py'), '--scenario', sid], check=True) for sid in ids]"
    ) else (
        python -c "import csv, subprocess, sys; from pathlib import Path; p = Path(r'%~2'); rows = list(csv.DictReader(p.open('r', encoding='utf-8', newline=''))); ids = [row.get('scenario_id','').strip() for row in rows if (row.get('scenario_id') or '').strip()]; assert ids, 'No scenario_id values found in ' + str(p); [print('Generating scenario:', sid) or subprocess.run([sys.executable, str(Path.cwd()/'scenarios'/'generators'/'scenario_factory.py'), '--scenario', sid], check=True) for sid in ids]"
    )
    if errorlevel 1 (
        echo Failed to generate scenarios from %~2
        popd
        exit /b 1
    )
    popd
    exit /b 0
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 scenarios\generators\scenario_factory.py --scenario "%~1"
) else (
    python scenarios\generators\scenario_factory.py --scenario "%~1"
)

if errorlevel 1 (
    echo Failed to generate scenario %~1
    popd
    exit /b 1
)

popd
exit /b 0