import json
import re

# Read file
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Find all occurrences of quote mismatches
# Pattern: something that looks like a string array element with wrong ending

# Fix 1: Replace any control characters that are not allowed in JSON strings
# JSON strings cannot contain literal control chars (0x00-0x1F) except escaped
# Replace actual control chars with spaces or remove them

# First, let's identify all problematic patterns
lines = content.split("\n")
problems = []

for i, line in enumerate(lines):
    # Check for unbalanced quotes in JSON string literals
    stripped = line.strip()
    if stripped.startswith('"') and stripped.endswith(","):
        # This is a JSON string array element
        quote_count = stripped.count('"')
        if quote_count % 2 != 0:
            problems.append((i + 1, line, "unbalanced quotes"))

    # Check for single quote at end (should be double quote)
    if stripped.endswith("','") or stripped.endswith("',\\n',"):
        problems.append((i + 1, line, "single quote ending"))

print(f"Found {len(problems)} problematic lines:")
for line_no, line, issue in problems[:10]:
    print(f"  Line {line_no}: {issue}")
    print(f"    {repr(line)}")

# Now let's try to fix the remaining issues
# Create a new clean version
print("\nCreating clean version...")

# Strategy: recreate the notebook from scratch with valid JSON
# Keep only the cells that are valid

# Find all cell boundaries
cell_starts = []
cell_ends = []
for match in re.finditer(r'"cell_type":\s*"(\w+)"', content):
    cell_starts.append(match.start())

# Try to parse what we can
# Split into sections and try to salvage valid parts
cells = []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if '"cell_type": "code"' in line or '"cell_type": "markdown"' in line:
        # Start of a cell
        cell_start = i
        # Find end of cell (next cell_type or end of cells array)
        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip()
            if '"cell_type":' in next_line and j > i + 1:
                break
            if next_line == "]" and '"source":' in lines[j - 1]:
                break
            j += 1
        cell_end = j

        # Extract cell content
        cell_lines = lines[cell_start:cell_end]
        cell_content = "\n".join(cell_lines)

        # Try to parse this cell
        try:
            # Wrap in braces to make it valid JSON
            test_json = "{" + cell_content + "}"
            cell_data = json.loads(test_json)
            cells.append(cell_data)
            print(
                f"  Successfully parsed cell {len(cells)} (lines {cell_start + 1}-{cell_end})"
            )
        except json.JSONDecodeError as e:
            print(f"  Failed to parse cell at lines {cell_start + 1}-{cell_end}: {e}")

        i = j
    else:
        i += 1

print(f"\nTotal valid cells: {len(cells)}")

if cells:
    # Create new notebook
    new_notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.8.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Write
    with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
        json.dump(new_notebook, f, indent=1, ensure_ascii=False)

    print("\nSuccessfully created clean notebook!")
else:
    print("\nNo valid cells found - need different approach")
