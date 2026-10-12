import re
import json
from pathlib import Path

def load_jsonc(file_path: str | Path) -> dict:
    """Strip JavaScript-style comments and load standard JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Remove single-line // comments and multi-line /* */ comments
    cleaned = re.sub(r"//.*?$|/\*.*?\*/", "", content, flags=re.MULTILINE | re.DOTALL)
    return json.loads(cleaned)