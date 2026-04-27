import json

# Read file as binary to preserve exact bytes
with open("cnn_tutorial_indonesia.ipynb", "rb") as f:
    data = f.read()

# Find and display any byte < 32 that's not CR, LF, or escaped tab
print("Finding control characters...")
for i, b in enumerate(data):
    if b < 32:
        if b not in [10, 13]:  # Not LF or CR
            print(f"Found control byte {b} (0x{b:02x}) at position {i}")
            print(f"Context: {data[max(0, i - 30) : i + 30]}")
            print()

# Try reading as text and looking for tabs
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Count tabs
if chr(9) in text:
    print(f"Found {text.count(chr(9))} tab characters in text")
    # Find positions
    idx = 0
    count = 0
    while True:
        idx = text.find(chr(9), idx)
        if idx == -1:
            break
        count += 1
        if count <= 5:
            print(f"  Tab at position {idx}: {repr(text[max(0, idx - 20) : idx + 20])}")
        idx += 1
else:
    print("No tab characters found")

# Now let's try to understand what character is at position 14831
print(
    f"\nCharacter at position 14831: {repr(text[14831]) if 14831 < len(text) else 'OUT OF RANGE'}"
)
print(f"Bytes around 14831: {data[14825:14835]}")

# Let's check line 367 specifically
lines = text.split("\n")
if len(lines) > 366:
    line367 = lines[366]
    print(f"\nLine 367 length: {len(line367)}")
    print(f"Line 367: {repr(line367)}")

    # Check each character
    for i, c in enumerate(line367):
        if ord(c) < 32:
            print(f"  Control char at column {i}: {repr(c)} (ord={ord(c)})")
