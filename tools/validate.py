import json

with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8") as f:
    try:
        data = json.load(f)
        print("SUCCESS! JSON is valid!")
        print("Cells:", len(data["cells"]))
        print("File ready to use!")
    except json.JSONDecodeError as e:
        print("ERROR:", e)
        print("Line:", e.lineno)
        print("Column:", e.colno)
