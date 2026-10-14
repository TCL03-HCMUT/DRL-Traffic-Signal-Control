"""
Check the notebook for basic validity and that the function calls are syntactically correct.
"""
import json
import ast
import sys
from pathlib import Path

def check_notebook(notebook_path):
    print(f"Checking notebook: {notebook_path.name}")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  Failed to load notebook as JSON: {e}")
        return False

    # Check that it has cells
    if 'cells' not in nb:
        print("  No 'cells' key in notebook.")
        return False

    print(f"  Found {len(nb['cells'])} cells.")

    # We'll check each code cell for basic Python syntax
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        cleaned_lines = []
        for line in cell.get('source', []):
            stripped = line.strip()
            if stripped.startswith('!') or stripped.startswith('%'):
                cleaned_lines.append(f"# {line}")
            else:
                cleaned_lines.append(line)
        source = ''.join(cleaned_lines)
        if not source.strip():
            continue
        try:
            # Parse the source to check for syntax errors
            ast.parse(source)
        except SyntaxError as e:
            print(f"  Syntax error in cell {i}: {e}")
            print(f"    Cell source: {source[:200]}...")
            return False
        except Exception as e:
            # Other errors (like missing imports) are okay for this check
            pass

    print("  All code cells are syntactically valid.")
    return True

def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "demo_orchestration.ipynb"
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        return False

    success = check_notebook(notebook_path)
    return success

if __name__ == "__main__":
    if main():
        print("\nNotebook check passed.")
        sys.exit(0)
    else:
        print("\nNotebook check failed.")
        sys.exit(1)