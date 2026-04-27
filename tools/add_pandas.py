import json

# Read the original notebook
with open("CNN_Learning_Module_Commented.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Create new cell with pandas visualization
pandas_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "# =============================================================================\n# VISUALISASI DATA DALAM BENTUK TABEL - Menggunakan Pandas\n# KOMENTAR: Shape saja tidak cukup, kita perlu lihat data sebenarnya!\n# =============================================================================\n\n# Install pandas kalau belum ada\n# !pip install pandas\n\nimport pandas as pd\n\n# =============================================================================\n# 1. LIHAT RAW DATA SEBELUM RESHAPE\n# =============================================================================\n\nprint(\"=\" * 70)\nprint(\"1. RAW DATA SEBELUM RESHAPE (Flattened)\")\nprint(\"=\" * 70)\n\n# Ambil protein pertama dari raw data (belum di-reshape)\nsample_raw = train_raw[0]\n\n# Buat DataFrame untuk 100 elemen pertama\ndf_raw = pd.DataFrame({\n    'Index': range(100),\n    'Value': sample_raw[:100],\n    'Feature_Type': ['One-Hot AA']*21 + ['Q3_Label']*3 + ['Reserved']*7 + \n                    ['Extra']*4 + ['Profile']*21 + ['Mask']*1 + ['More...']*43\n})\n\nprint(f\"\\nShape raw data: {sample_raw.shape}\")\nprint(f\"Total elemen: {len(sample_raw):,}\")\nprint(\"\\n100 elemen pertama:\")\nprint(df_raw.to_string(index=False))\nprint(\"\\n... (masih ada 39,800 elemen lagi)\"\n",
}

# Insert new cell after index 3 (LOAD DATA cell)
notebook["cells"].insert(4, pandas_cell)

# Write back
with open("CNN_Learning_Module_With_Pandas.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook dengan visualisasi pandas berhasil dibuat!")
print("File: CNN_Learning_Module_With_Pandas.ipynb")
