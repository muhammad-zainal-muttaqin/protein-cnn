#!/usr/bin/env python3
import json
import re

# Read file
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Find the problematic section
print("Searching for control characters...")

# Check every character
for i, char in enumerate(content):
    code = ord(char)
    if code == 9:  # Tab
        print(f"Tab at position {i}: {repr(content[max(0, i - 30) : i + 30])}")
    elif code == 10 or code == 13:  # Newline/CR
        pass  # These are OK between JSON elements
    elif code < 32:  # Other control characters
        print(
            f"Control char {code} at position {i}: {repr(content[max(0, i - 30) : i + 30])}"
        )

# Try to fix by replacing tabs in string literals
# Pattern: find tabs that are inside quotes (not at start of line)
print("\nAttempting to fix...")

# Method 1: Replace all tabs
fixed = content.replace(chr(9), "    ")

# Try to parse
try:
    data = json.loads(fixed)
    print("\nSUCCESS! JSON is now valid after replacing tabs")
    with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
        f.write(fixed)
    print("File saved!")
except json.JSONDecodeError as e:
    print(f"\nStill has error: {e}")
    print(f"Position: {e.pos}")
    print(f"Character: {repr(fixed[e.pos]) if e.pos < len(fixed) else 'EOF'}")

    # Show more context
    start = max(0, e.pos - 50)
    end = min(len(fixed), e.pos + 50)
    print(f"Context: {repr(fixed[start:end])}")
