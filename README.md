# Protein CNN — Prediksi Secondary Structure Protein pada CullPDB / CB513

Repo ini mendokumentasikan dua jalur eksperimen prediksi **secondary structure protein** menggunakan `CullPDB` sebagai train/validation dan `CB513` sebagai **final holdout test set**:

- **Q8** — 8-class prediction, sweep arsitektur berjenjang dari baseline CNN hingga `Residual Dilated CNN 1D`, **101 run tercatat**
- **Q3** — 3-class prediction (Helix / Sheet / Coil), baseline simpel vs `ResCNN + BiLSTM + Attention` dengan Optuna search

---

## Ringkasan Hasil Terbaik

| Task | Model | Accuracy | F1 (macro) | AUC | Test Loss |
|---|---|---:|---:|---:|---:|
| **Q8** | ResDil CNN 1D (`p4_07`) | 0.6926 | — | — | 0.9242 |
| Q3 baseline | CNN 1D sederhana | **0.8139** | 0.4313 | 0.9484 | 0.0120 |
| **Q3 terbaik** | ResCNN+BiLSTM+Attention | **0.8244** | **0.4544** | **0.9814** | 0.0453 |

---

## Bagian 1 — Q8 Secondary Structure (101 Eksperimen)

### Keputusan Final

| Komponen | Keputusan |
|---|---|
| Dataset train | `cullpdb+profile_5926_filtered.npy.gz` |
| Dataset test | `cb513+profile_split1.npy.gz` |
| Task | masked Q8 prediction |
| Input features | `baseline42` — 21 one-hot + 21 profile = 42 kanal |
| Arsitektur terbaik | `resdil_cnn1d` |
| Recipe terbaik | `baseline42 + ce + no class weighting` |
| Run final terbaik | `p4_07_resdil_b42_ce_none_c320_e24_seed7` |
| Ledger eksperimen | [results/reports/run_ledger.csv](results/reports/run_ledger.csv) |
| Report lengkap | [results/reports/final_report.md](results/reports/final_report.md) |

### Metrik Resmi Q8

| Run | Best Val Q8 | Test Q8 | Test Loss |
|---|---:|---:|---:|
| Baseline CNN 1D | 0.7061 | 0.6695 | 0.9079 |
| Baseline CNN 2D | 0.6795 | 0.6427 | 0.9919 |
| Tuned CNN 1D (Optuna) | 0.7203 | 0.6820 | 0.8967 |
| Tuned CNN 2D (Optuna) | 0.6914 | 0.6526 | 0.9664 |
| Incremental ResDil Step 1 | 0.7268 | 0.6878 | 0.8872 |
| **Best Final ResDil** | **0.7354** | **0.6926** | **0.9242** |

Gain total: **+0.0231** absolute dibanding baseline CNN 1D awal.

### Perjalanan Eksperimen Q8

Setiap titik di grafik berikut adalah satu run. Garis hitam menunjukkan running best.

![Perkembangan test Q8 seluruh run](results/figures/protein_cnn_progress_test_q8.png)

Sumber data: [results/figures/data/protein_cnn_progress_test_q8.csv](results/figures/data/protein_cnn_progress_test_q8.csv)

Beberapa run `extended46` mencapai validation Q8 tinggi tetapi tidak generalize ke CB513 — terlihat jelas pada grafik validation di bawah.

![Perkembangan best validation Q8](results/figures/protein_cnn_progress_best_val_q8.png)

Sumber data: [results/figures/data/protein_cnn_progress_best_val_q8.csv](results/figures/data/protein_cnn_progress_best_val_q8.csv)

#### Dataset & Setup Evaluasi

| Split | Protein | Valid Residues | Mean Length | Median Length |
|---|---:|---:|---:|---:|
| CullPDB total | 5365 | 1154412 | 215.2 | 184 |
| CB513 test | 514 | 84765 | 164.9 | 132 |

- Input shape: `(N, 700, 57)` — fitur `[:, :, 0:21]` + `[:, :, 35:56]` = 42 kanal
- Padding dimask saat hitung loss dan accuracy
- CB513 dijaga ketat sebagai test final (tidak pernah disentuh saat tuning)

#### Fase Riset

| Fase | Deskripsi |
|---|---|
| `research_incremental` | Validasi awal `resdil_cnn1d` |
| `research_stage1` | Sweep feature set, objective, weighting, width (48 run) |
| `research_stage2` | Retrain panjang kandidat terbaik (8 run) |
| `research_stage3_confirm` | Konfirmasi multi-seed kandidat top (12 run) |
| `research_phase4` | Penajaman akhir keluarga ResDil (8 run) |

![Ringkasan performa test per phase](results/figures/protein_cnn_phase_summary.png)

Sumber data: [results/figures/data/protein_cnn_phase_summary.csv](results/figures/data/protein_cnn_phase_summary.csv)

![Perbandingan keluarga model](results/figures/protein_cnn_model_family_summary.png)

Sumber data: [results/figures/data/protein_cnn_model_family_summary.csv](results/figures/data/protein_cnn_model_family_summary.csv)

- `cnn2d` tidak pernah mendekati ceiling keluarga `1D`
- `resdil_cnn1d` mengambil alih posisi terbaik dari `cnn1d` plain

#### Top-5 Run Q8

| Rank | Run | Phase | Best Val Q8 | Test Q8 | Test Loss |
|---:|---|---|---:|---:|---:|
| 1 | `p4_07_resdil_b42_ce_none_c320_e24_seed7` | Phase 4 | 0.7354 | 0.6926 | 0.9242 |
| 2 | `s2_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320` | Stage 2 | 0.7303 | 0.6919 | 0.9327 |
| 3 | `p4_01_resdil_b42_ce_none_c320_e18` | Phase 4 | 0.7303 | 0.6919 | 0.9327 |
| 4 | `s3_02_s1_02_resdil_cnn1d_baseline42_ce_none_c192_seed13` | Stage 3 | 0.7307 | 0.6917 | 0.8843 |
| 5 | `s3_03_s1_04_resdil_cnn1d_baseline42_ce_none_c256_seed21` | Stage 3 | 0.7306 | 0.6913 | 0.9025 |

#### Kandidat Terbaik — Stabilitas Seed

Family yang konsisten sehat: `resdil_cnn1d + baseline42 + ce + no weighting`

![Stabilitas kandidat ResDil](results/figures/protein_cnn_resdil_candidate_stability.png)

Sumber data: [results/figures/data/protein_cnn_resdil_candidate_stability.csv](results/figures/data/protein_cnn_resdil_candidate_stability.csv)

#### Kurva Training Representatif Q8

![Kurva training representatif](results/figures/protein_cnn_training_curves.png)

Sumber data: [results/figures/data/protein_cnn_training_curves.csv](results/figures/data/protein_cnn_training_curves.csv)

#### Loss vs Accuracy

![Hubungan test loss vs test Q8](results/figures/protein_cnn_test_loss_vs_q8.png)

Sumber data: [results/figures/data/protein_cnn_test_loss_vs_q8.csv](results/figures/data/protein_cnn_test_loss_vs_q8.csv)

Loss lintas family objective tidak boleh dibandingkan mentah. `sqrt_inverse` dan varian focal bisa terlihat "loss kecil" tapi kalah di `test_q8`.

---

## Bagian 2 — Q3 Secondary Structure

Task Q3 menyederhanakan label menjadi 3 kelas: **Helix (0) / Sheet (1) / Coil (2)**. Dua model dibandingkan pada dataset yang sama (CullPDB train, CB513 test).

### Model A — CNN 1D Baseline Q3

Model sederhana sebagai referensi. Dilatih 30 epoch, checkpoint terbaik di epoch 12 (best val loss).

| Metric | Nilai |
|---|---:|
| Accuracy | **0.8139** |
| Precision (macro) | 0.4195 |
| Recall (macro) | 0.6177 |
| F1 Score (macro) | 0.4313 |
| AUC (OVR) | 0.9484 |
| Loss | 0.0120 |
| Checkpoint | Epoch 12 / 30 |

Notebook: [`notebooks/results/Protein_1D_Q3.ipynb`](notebooks/results/Protein_1D_Q3.ipynb)

### Model B — ResCNN + BiLSTM + Attention (Optuna)

Model lanjutan dengan arsitektur terbaik dari 18 Optuna trials (TPE, 8 epoch per trial). Trial 15 menghasilkan val macro-F1 terbaik = **0.5040** dan dijadikan konfigurasi final 40 epoch.

#### Konfigurasi Terbaik (Trial 15)

| Hyperparameter | Nilai |
|---|---|
| Filters | 128 |
| Conv blocks | 4 |
| Kernel size | 5 |
| LSTM units | 64 |
| Attention heads | 2 |
| Attention key dim | 24 |
| Dense units | 96 |
| Dropout | 0.2542 |
| Learning rate | 8.55e-04 |
| Focal gamma | 1.726 |
| Label smoothing | 0.0348 |
| Batch size | 8 |

#### Metrik Final pada CB513 (40 Epoch)

| Metric | Nilai |
|---|---:|
| Accuracy | **0.8244** |
| Balanced Accuracy | 0.7017 |
| Precision (macro) | 0.5041 |
| Recall (macro) | 0.7017 |
| F1 Score (macro) | **0.4544** |
| AUC (OVR) | **0.9814** |
| Test Loss | 0.0453 |
| Best Val Macro-F1 | **0.5033** |

#### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — Helix | 1.0000 | 0.8492 | 0.9184 | 340,699 |
| 1 — Sheet | 0.4937 | 0.3482 | 0.4084 | 17,920 |
| 2 — Coil | 0.0185 | 0.9077 | 0.0363 | 1,181 |

Helix hampir sempurna karena mendominasi distribusi (340k vs 1.2k residu Coil). Macro metrics rendah akibat imbalance ekstrem antar kelas.

#### Perbandingan Baseline vs ResCNN+BiLSTM+Attention

| Metric | CNN 1D Baseline | ResCNN+BiLSTM+Att | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8139 | **0.8244** | +0.0105 |
| F1 (macro) | 0.4313 | **0.4544** | +0.0231 |
| AUC (OVR) | 0.9484 | **0.9814** | +0.0330 |
| Loss | 0.0120 | 0.0453 | — |

#### Kurva Training — ResCNN+BiLSTM+Attention Q3

![Kurva training Q3 final](results/training/training_curves.png)

![Kurva training detail final run](results/training/final_train_20260426_131406/training_curves.png)

Artefak lengkap: [`results/training/final_train_20260426_131406/`](results/training/final_train_20260426_131406/)  
Detail Optuna trials: [`logs/TRIAL_RESULTS.md`](logs/TRIAL_RESULTS.md)

---

## Insight Utama

1. **CNN 1D menang jelas atas CNN 2D pada Q8.** Gap tidak berubah dari baseline sampai akhir.
2. **Optuna membantu, tapi tidak cukup sendirian.** Gain signifikan datang dari perubahan arsitektur ke `resdil_cnn1d`.
3. **`baseline42` lebih sehat dari `extended46` untuk generalisasi.** Beberapa run `extended46` overfitting ke validation internal.
4. **`ce + no class weighting` adalah jalur terbaik Q8.** Family ini mendominasi top ledger.
5. **Q3 ResCNN+BiLSTM+Attention mencatat AUC 0.9814 dan accuracy 82.44%**, naik dari baseline Q3 di 81.39% — tapi macro F1 tetap rendah karena imbalance kelas berat (Helix 95%, Coil 0.3%).

---

## Audit Trail

### Q8 Artifacts

- Ledger final: [results/reports/run_ledger.csv](results/reports/run_ledger.csv)
- Report lengkap: [results/reports/final_report.md](results/reports/final_report.md)
- Status terbaru: [results/reports/latest_status.md](results/reports/latest_status.md)
- Summary riset: [results/reports/research_summary.json](results/reports/research_summary.json)
- Summary phase 4: [results/reports/phase4_summary.json](results/reports/phase4_summary.json)
- Baseline CNN 1D: [results/artifacts/cnn1d/report.json](results/artifacts/cnn1d/report.json)
- Baseline CNN 2D: [results/artifacts/cnn2d/report.json](results/artifacts/cnn2d/report.json)
- Tuned CNN 1D: [results/artifacts/optuna_cnn1d/optuna_report.json](results/artifacts/optuna_cnn1d/optuna_report.json)
- Tuned CNN 2D: [results/artifacts/optuna_cnn2d/optuna_report.json](results/artifacts/optuna_cnn2d/optuna_report.json)
- Best final run: [results/artifacts/research_runs/p4_07_resdil_b42_ce_none_c320_e24_seed7/report.json](results/artifacts/research_runs/p4_07_resdil_b42_ce_none_c320_e24_seed7/report.json)

### Q3 Artifacts

- Optuna trial results: [logs/TRIAL_RESULTS.md](logs/TRIAL_RESULTS.md)
- Test metrics JSON: [results/training/test_metrics.json](results/training/test_metrics.json)
- Model weights: [results/training/final_train_20260426_131406/best_weights.weights.h5](results/training/final_train_20260426_131406/best_weights.weights.h5)
- Final model: [results/training/final_train_20260426_131406/final_model.keras](results/training/final_train_20260426_131406/final_model.keras)
- Training history: [results/training/final_train_20260426_131406/final_history.json](results/training/final_train_20260426_131406/final_history.json)
- Baseline notebook: [notebooks/results/Protein_1D_Q3.ipynb](notebooks/results/Protein_1D_Q3.ipynb)
- ResCNN notebook: [notebooks/results/protein_q3_rescnn_bilstm_attention_optuna.ipynb](notebooks/results/protein_q3_rescnn_bilstm_attention_optuna.ipynb)

---

## Cara Reproduksi

**Q8 Baseline CNN 1D:**

```bash
python train.py \
  --train-path /workspace/cullpdb+profile_5926_filtered.npy.gz \
  --test-path /workspace/cb513+profile_split1.npy.gz \
  --model cnn1d \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --output-dir results/artifacts/cnn1d
```

**Q8 Optuna CNN 1D:**

```bash
python tune_optuna.py \
  --train-path /workspace/cullpdb+profile_5926_filtered.npy.gz \
  --test-path /workspace/cb513+profile_split1.npy.gz \
  --model cnn1d \
  --trials 8 \
  --epochs 6 \
  --final-epochs 18 \
  --output-dir results/artifacts/optuna_cnn1d
```

**Q3 ResCNN+BiLSTM+Attention (Optuna search + final training):**

```bash
python scripts/train_q3.py
```

Analisis lengkap, tabel top-10, pembahasan run gagal, dan interpretasi detail ada di [results/reports/final_report.md](results/reports/final_report.md).
