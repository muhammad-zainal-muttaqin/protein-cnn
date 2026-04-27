with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find lines ending with "" that are not escaped
issues = []
for i, line in enumerate(lines):
    stripped = line.rstrip()
    # Check for "" at the end
    if stripped.endswith('""') and not stripped.endswith('\\""'):
        issues.append((i + 1, line))

print(f"Found {len(issues)} lines with double quote issues:")
for line_no, line in issues:
    print(f"  Line {line_no}: {repr(line)}")
