import json

# Read the existing notebook
try:
    with open(
        "cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8", errors="ignore"
    ) as f:
        content = f.read()

    # Try to find where cells start
    cell_start = content.find('"cells":')
    if cell_start > 0:
        print(f"Found cells at position {cell_start}")

        # Find the problematic cell (cell with 'TAMPILKAN RINGKASAN MODEL')
        problem_idx = content.find("TAMPILKAN RINGKASAN MODEL")
        if problem_idx > 0:
            print(f"Found problematic content at position {problem_idx}")

            # Show context
            context_start = max(0, problem_idx - 200)
            context_end = min(len(content), problem_idx + 200)
            print(f"Context: {repr(content[context_start:context_end])}")

            # Find the start and end of this cell
            cell_type_start = content.rfind('"cell_type": "code"', 0, problem_idx)
            next_cell = content.find('"cell_type":', problem_idx)

            if cell_type_start > 0 and next_cell > 0:
                print(f"Cell starts at {cell_type_start}")
                print(f"Next cell at {next_cell}")

                # Remove the problematic cell
                fixed_content = content[:cell_type_start] + content[next_cell:]

                # Write back
                with open("cnn_tutorial_indonesia.ipynb", "w", encoding="utf-8") as f:
                    f.write(fixed_content)

                print("Removed problematic cell!")

                # Test
                with open("cnn_tutorial_indonesia.ipynb", "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        print(f"SUCCESS! Notebook has {len(data['cells'])} cells")
                    except json.JSONDecodeError as e:
                        print(f"Still has error: {e}")

except Exception as e:
    print(f"Error: {e}")
