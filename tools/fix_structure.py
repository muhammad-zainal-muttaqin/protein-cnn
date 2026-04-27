import json
import re

# Read file
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"Line 367 index: {366}")
print()

# Check JSON structure around line 367
print("Lines 365-370:")
for i in range(364, min(371, len(lines))):
    marker = " <-- ERROR LINE" if i == 366 else ""
    print(f"{i + 1}: {repr(lines[i])}{marker}")

print()
print("Checking JSON structure...")

# Try to manually construct valid JSON by fixing common issues
fixed_lines = []
for i, line in enumerate(lines):
    # Ensure proper line endings
    if not line.endswith("\n"):
        line = line + "\n"
    fixed_lines.append(line)

# Write and test
fixed_content = "".join(fixed_lines)
with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
    f.write(fixed_content)

try:
    data = json.loads(fixed_content)
    print("SUCCESS! JSON is valid!")
    print(f"Cells: {len(data['cells'])}")
except json.JSONDecodeError as e:
    print(f"ERROR: {e}")
    print(f"Line: {e.lineno}, Col: {e.colno}")

    # Show problematic line
    if e.lineno <= len(lines):
        print(f"Content: {repr(lines[e.lineno - 1])}")
