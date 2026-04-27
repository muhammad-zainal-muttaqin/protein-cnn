with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Fix line 367 (index 366)
line = lines[366]
print("Before:", repr(line))

# The line should end with ",' but it ends with ',
# Replace the incorrect ending
if "','\n" in line:
    line = line.replace("','\n", '",\n')
    print("Fixed single quote to double quote")

print("After:", repr(line))
lines[366] = line

# Write back
with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("File updated!")

# Test JSON
import json

with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("SUCCESS! JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"ERROR: {e}")
