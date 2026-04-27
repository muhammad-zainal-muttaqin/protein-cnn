import json
import re

# Read the file
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Replace actual tab characters (ASCII 9) with 4 spaces
fixed_content = content.replace("\t", "    ")

# Also handle any other control characters that might be problematic
fixed_content = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", fixed_content)

# Write back
with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
    f.write(fixed_content)

# Validate
with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("SUCCESS! File JSON sekarang valid!")
        print(f"Jumlah cells: {len(data['cells'])}")
        print("File siap dibuka di Jupyter/Colab")
    except json.JSONDecodeError as e:
        print(f"Masih ada error: {e}")
        print(f"Line: {e.lineno}, Column: {e.colno}")
