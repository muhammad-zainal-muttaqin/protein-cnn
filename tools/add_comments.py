import json

# Read the original notebook
with open("CNN_Learning_Module.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Enhanced cell 2 with detailed comments
notebook["cells"][2][
    "source"
] = """# =============================================================================
# IMPORT LIBRARY - Library standar untuk ML dan Deep Learning
# =============================================================================

# numpy: Library fundamental untuk array numerik dan matematika matrix
# Digunakan untuk manipulasi data, operasi matematika, dan array processing
import numpy as np

# torch: Framework deep learning dari Meta (Facebook) AI Research
# PyTorch menggunakan dynamic computational graph yang lebih fleksibel
import torch

# torch.nn: Neural network layers dan fungsi (Conv1d, Linear, ReLU, dll)
import torch.nn as nn

# torch.nn.functional: Fungsi tanpa state (activation, loss functions)
import torch.nn.functional as F

# Dataset dan DataLoader: Tools PyTorch untuk data handling dan batching
from torch.utils.data import Dataset, DataLoader

# matplotlib: Library plotting standar (mirip MATLAB)
import matplotlib.pyplot as plt

# Path manipulation: Cross-platform file path handling
from pathlib import Path

# JSON processing: Untuk menyimpan log dan config
import json

# Dataclasses: Membuat class data dengan auto-generated methods
from dataclasses import dataclass

# =============================================================================
# DEVICE SETUP - Menentukan GPU atau CPU
# =============================================================================

# torch.cuda.is_available() cek apakah NVIDIA GPU tersedia
# Kalau ya, gunakan 'cuda' (GPU acceleration)
# Kalau tidak, gunakan 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""

# Enhanced cell 4 (Load data)
notebook["cells"][4][
    "source"
] = """# =============================================================================
# LOAD DATA - Membaca file protein dari disk
# =============================================================================

# Path ke dataset protein
# CullPDB: Dataset training dengan 5365 protein yang strukturnya diketahui
train_path = '/workspace/cullpdb+profile_5926_filtered.npy.gz'

# CB513: Dataset test terpisah untuk evaluasi final
test_path = '/workspace/cb513+profile_split1.npy.gz'

# np.load dengan mmap_mode='r': Memory-mapped file (lazy loading)
# KEUNTUNGAN: Data tidak dimuat ke RAM sekaligus, hemat memori
train_raw = np.load(train_path, mmap_mode='r')
test_raw = np.load(test_path, mmap_mode='r')

print(f"Raw train shape: {train_raw.shape}")  # (5365, 39900)
print(f"Raw test shape: {test_raw.shape}")    # (514, 39900)

# =============================================================================
# RESHAPE DATA - Mengubah bentuk data ke format 3D
# =============================================================================

# reshape(-1, 700, 57):
# -1: Biarkan numpy hitung jumlah samples
# 700: Panjang sequence (amino acids)
# 57: Fitur per posisi (21 one-hot + 3 label + 33 profile/mask)
train_data = train_raw.reshape(-1, 700, 57)
test_data = test_raw.reshape(-1, 700, 57)

print(f"Reshaped train: {train_data.shape}")  # (5365, 700, 57)
print(f"Reshaped test: {test_data.shape}")  # (514, 700, 57)
print(f"Jumlah protein: {train_data.shape[0]}")"""

# Enhanced cell 5 (Understand structure)
notebook["cells"][5][
    "source"
] = """# =============================================================================
# MEMBANDINGKAN STRUKTUR FITUR - Apa saja yang tersedia?
# =============================================================================

print("Struktur fitur per posisi (57 total):")
print("  [:21]     = One-hot amino acid (21 classes)")
print("              Vektor binary: 1 untuk amino acid yang ada, 0 lainnya")
print("  [21:24]   = Q3 labels (3 kelas: Coil, Helix, Sheet)")
print("  [24:31]   = Reserved/Unknown (7 fitur)")
print("  [31:35]   = Extra features (4 fitur: accessibility, entropy)")
print("  [35:56]   = Profile/PSSM (21 nilai konservasi evolutionary)")
print("  [56]      = Mask (0=valid, 1=padding)")

# Visualisasi satu protein
sample = train_data[0]  # Ambil protein pertama
print(f"Sample protein shape: {sample.shape}")  # (700, 57)

# Lihat 5 posisi pertama dari one-hot encoding
print("Amino acid one-hot (first 5 positions):")
print(sample[:5, :21])

# Interpretasi: Setiap baris adalah satu posisi
# Angka 1 menunjukkan amino acid mana (contoh: [1,0,0,...] = Alanine)"""

# Enhanced cell 6 (Data preparation)
notebook["cells"][6][
    "source"
] = '''# =============================================================================
# DATA PREPARATION - Function untuk ekstrak fitur dan label
# =============================================================================

def prepare_data(data, feature_set='baseline42'):
    """
    Extract features dan labels dari raw data.
    
    Args:
        data: Array dengan shape (n_samples, 700, 57)
        feature_set: Pilihan fitur yang akan digunakan
            - 'baseline42': 21 amino acid + 21 profile = 42 fitur
            - 'extended46': 21 amino acid + 4 extra + 21 profile = 46 fitur
    
    Returns:
        features: Input untuk model (n_samples, 700, n_features)
        labels: Target labels (n_samples, 700) dengan nilai 0-7
        mask: Boolean array untuk posisi valid
    """
    
    # Ekstrak komponen fitur
    # [:, :, 0:21] mengambil kolom 0-20 untuk semua samples dan posisi
    aa = np.array(data[:, :, 0:21], dtype=np.float32)      # One-hot amino acid
    profile = np.array(data[:, :, 35:56], dtype=np.float32)  # PSSM profile
    extra = np.array(data[:, :, 31:35], dtype=np.float32)     # Extra features
    
    # Kombinasi fitur sesuai konfigurasi
    if feature_set == 'baseline42':
        # baseline42: Amino acid + Profile = 42 fitur (REKOMENDASI)
        # np.concatenate menggabungkan di axis terakhir (fitur)
        features = np.concatenate([aa, profile], axis=-1)
    elif feature_set == 'extended46':
        # extended46: + Extra features = 46 fitur
        features = np.concatenate([aa, extra, profile], axis=-1)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    # Ekstrak labels Q8 (8 kelas)
    # np.argmax mengubah one-hot (8 kolom) menjadi class index (1 angka 0-7)
    labels_onehot = data[:, :, 22:30]
    labels = np.argmax(labels_onehot, axis=-1).astype(np.int64)
    
    # Mask: True untuk posisi valid (kolom 56 == 0)
    mask = (data[:, :, 56] == 0)
    
    return features, labels, mask

# Apply ke training dan test set
X_train, y_train, mask_train = prepare_data(train_data)
X_test, y_test, mask_test = prepare_data(test_data)

print(f"X_train shape: {X_train.shape}")  # (5365, 700, 42)
print(f"y_train shape: {y_train.shape}")   # (5365, 700)
print(f"Feature dimension: {X_train.shape[-1]}")  # 42
print(f"Number of classes: {np.max(y_train) + 1}")  # 8 (Q8)'''

# Enhanced cell 7 (Visualization)
notebook["cells"][7][
    "source"
] = """# =============================================================================
# VISUALISASI DISTRIBUSI - Analisis karakteristik dataset
# =============================================================================

# Ambil semua label yang valid (bukan padding)
# Boolean indexing: y_train[mask_train] hanya ambil posisi True
labels_flat = y_train[mask_train]

# Hitung frekuensi setiap kelas
# minlength=8 memastikan semua 8 kelas terwakili
label_counts = np.bincount(labels_flat, minlength=8)
label_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']

# Buat figure dengan 2 subplot
plt.figure(figsize=(10, 4))

# Subplot 1: Distribusi label
plt.subplot(1, 2, 1)  # 1 baris, 2 kolom, subplot ke-1
plt.bar(label_names, label_counts, edgecolor='black')
plt.title('Distribusi Label Q8 (Training)')
plt.xlabel('Secondary Structure')
plt.ylabel('Count')

# Tambahkan nilai di atas bar
for i, count in enumerate(label_counts):
    plt.text(i, count + max(label_counts)*0.01, str(count), ha='center')

# Subplot 2: Distribusi panjang protein
plt.subplot(1, 2, 2)  # 1 baris, 2 kolom, subplot ke-2
# mask_train.sum(axis=1) = jumlah posisi valid per protein
lengths = mask_train.sum(axis=1)
plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribusi Panjang Protein')
plt.xlabel('Length')
plt.ylabel('Count')
# Garis vertikal di rata-rata dengan warna merah putus-putus
plt.axvline(lengths.mean(), color='r', linestyle='--', 
            label=f'Mean: {lengths.mean():.1f}')
plt.legend()

plt.tight_layout()  # Atur spacing agar tidak overlap
plt.show()

# Print statistik
print("Statistik dataset:")
print(f"  Jumlah protein: {len(lengths):,}")
print(f"  Rata-rata panjang: {lengths.mean():.1f}")
print(f"  Median panjang: {np.median(lengths):.1f}")
print(f"  Min length: {lengths.min()}")
print(f"  Max length: {lengths.max()}")"""

# Write back
with open("CNN_Learning_Module_Commented.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook dengan komentar berhasil dibuat!")
