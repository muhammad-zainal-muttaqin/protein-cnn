# Protein CNN — Secondary Structure Prediction on CullPDB / CB513

Repo ini mendokumentasikan eksperimen CNN untuk prediksi secondary structure protein berbasis `CullPDB` sebagai train/validation source dan `CB513` sebagai final test set. Fokus eksperimen sengaja dijaga sederhana dan ketat: mulai dari baseline `CNN 1D` vs `CNN 2D`, lalu lanjut ke hyperparameter tuning pada kedua arsitektur dengan audit trail yang lengkap.

Eksperimen berjalan dalam tiga fase yang sequential: verifikasi dataset → baseline architecture comparison → hyperparameter tuning. Setiap fase menghasilkan keputusan yang di-lock dan dibawa ke fase berikutnya, sehingga progres eksperimen tetap reproducible dan traceable.

Hasil akhir saat ini: model terbaik adalah **tuned `CNN 1D`** dengan **validation Q8 = 0.7203** dan **test Q8 pada CB513 = 0.6820**. `CNN 2D` juga membaik setelah tuning, tetapi tetap tertinggal cukup jelas dengan **test Q8 = 0.6526**. Kesimpulan praktisnya sederhana: untuk problem ini, `CNN 1D` tetap menjadi backbone yang paling masuk akal untuk diteruskan.

---

## Keputusan Final & Angka Resmi

Sebelum masuk ke narasi per fase, ini keputusan dan angka resmi yang sekarang di-lock:

| Komponen | Keputusan |
| --- | --- |
| Dataset train utama | `cullpdb+profile_5926_filtered.npy.gz` |
| Dataset test final | `cb513+profile_split1.npy.gz` |
| Task | masked `Q8` secondary-structure prediction |
| Input features | `42` kanal per residu |
| Arsitektur terbaik | `CNN 1D` |
| Baseline terbaik | `artifacts/cnn1d/report.json` |
| Run tuning terbaik | `artifacts/optuna_cnn1d/optuna_report.json` |
| Ledger seluruh run | `outputs/reports/run_ledger.csv` |

Metrik resmi saat ini:

| Run | Best Val Q8 | Test Q8 on CB513 |
| --- | ---: | ---: |
| Baseline CNN 1D | 0.7061 | 0.6695 |
| Baseline CNN 2D | 0.6795 | 0.6427 |
| Tuned CNN 1D | **0.7203** | **0.6820** |
| Tuned CNN 2D | 0.6914 | 0.6526 |

Status terkini yang paling ringkas tersedia di [outputs/reports/latest_status.md](outputs/reports/latest_status.md).

---

## Format Dataset Yang Dipakai

File input di workspace:

- `/workspace/cullpdb+profile_5926_filtered.npy.gz`
- `/workspace/cb513+profile_split1.npy.gz`

Meskipun ekstensi file berakhiran `.gz`, isi file yang tersedia di environment ini sebenarnya adalah array `.npy` langsung.

Setelah di-reshape, bentuk data menjadi:

- `CullPDB`: `(5365, 700, 57)`
- `CB513`: `(514, 700, 57)`

Interpretasi praktis yang dipakai di repo ini:

- input sequence one-hot: `[:, :, 0:21]`
- label Q8 one-hot: `[:, :, 22:30]`
- profile features kontinu: `[:, :, 35:56]`
- padding mask: `[:, :, 56] == 1`

Fitur yang benar-benar dipakai model:

- `21` kanal amino-acid one-hot
- `21` kanal profile kontinu
- total input = `42` kanal per residu

Catatan penting:

- split validation diambil dari protein-level split internal `CullPDB`
- `CB513` dipakai sebagai test final, bukan validation selama training

---

## Perjalanan Eksperimen

### Phase 0 — Validasi dataset & setup eksperimen

Fase ini memastikan bahwa dataset dapat dibaca dengan benar, reshape ke `(700, 57)` valid, mask padding konsisten, dan split train/validation/test tidak tercampur.

Keputusan dari fase ini:

- eksperimen difokuskan ke task `Q8`
- input memakai `42` fitur, bukan semua `57` kanal mentah
- `CB513` dijaga sebagai final holdout test

Artefak yang relevan:

- [protein_cnn/data.py](protein_cnn/data.py)
- [train.py](train.py)

### Phase 1 — Baseline `CNN 1D` vs `CNN 2D`

Dua baseline pertama dijalankan dengan setup yang sebanding:

- `CNN 1D`: konvolusi sepanjang dimensi sequence
- `CNN 2D`: input diperlakukan sebagai grid `700 x 42`

Hasil baseline:

| Model | Epoch | Best Val Q8 | Test Q8 |
| --- | ---: | ---: | ---: |
| CNN 1D | 5 | 0.7061 | 0.6695 |
| CNN 2D | 5 | 0.6795 | 0.6427 |

Temuan utama:

- `CNN 1D` sudah unggul sejak baseline awal
- `CNN 2D` lebih lambat dan lebih lemah
- gap performa tidak kecil, jadi `CNN 1D` layak dijadikan fokus tuning

Artefak baseline:

- [artifacts/cnn1d/report.json](artifacts/cnn1d/report.json)
- [artifacts/cnn1d/history.jsonl](artifacts/cnn1d/history.jsonl)
- [artifacts/cnn2d/report.json](artifacts/cnn2d/report.json)
- [artifacts/cnn2d/history.jsonl](artifacts/cnn2d/history.jsonl)

### Phase 2 — Hyperparameter tuning

Fase ini menjalankan `Optuna` pada kedua arsitektur dengan budget yang lebih serius:

- search: `8 trials`
- search per trial: `6 epoch`
- retrain final best params: `18 epoch`

#### Tuned CNN 1D

Best params:

- `batch_size = 16`
- `lr = 0.0007645573`
- `weight_decay = 3.715e-06`
- `dropout = 0.1677`
- `width = 192`

Hasil:

- best validation dari search: `0.7121`
- best validation setelah final retrain: `0.7203`
- final test Q8: `0.6820`

Ini berarti tuning `CNN 1D` memberi gain nyata terhadap baseline:

- `+0.0143` absolute pada validation (`0.7061 -> 0.7203`)
- `+0.0125` absolute pada test (`0.6695 -> 0.6820`)

Artefak:

- [artifacts/optuna_cnn1d/optuna_report.json](artifacts/optuna_cnn1d/optuna_report.json)
- [artifacts/optuna_cnn1d/trial_logs.jsonl](artifacts/optuna_cnn1d/trial_logs.jsonl)
- [artifacts/optuna_cnn1d/final_history.jsonl](artifacts/optuna_cnn1d/final_history.jsonl)

#### Tuned CNN 2D

Best params:

- `batch_size = 16`
- `lr = 0.0011606166`
- `weight_decay = 1.647e-05`
- `dropout = 0.1902`

Hasil:

- best validation dari search: `0.6815`
- best validation setelah final retrain: `0.6914`
- final test Q8: `0.6526`

Tuning `CNN 2D` memang meningkatkan hasil baseline, tetapi kenaikannya tidak cukup untuk menyalip `CNN 1D`.

Artefak:

- [artifacts/optuna_cnn2d/optuna_report.json](artifacts/optuna_cnn2d/optuna_report.json)
- [artifacts/optuna_cnn2d/trial_logs.jsonl](artifacts/optuna_cnn2d/trial_logs.jsonl)
- [artifacts/optuna_cnn2d/final_history.jsonl](artifacts/optuna_cnn2d/final_history.jsonl)

---

## Insight Utama

Kesimpulan paling penting dari seluruh perjalanan eksperimen:

1. `CNN 1D` unggul sejak baseline dan tetap unggul setelah tuning.
2. Hyperparameter tuning memang membantu kedua model, tetapi **membantu `CNN 1D` lebih efektif**.
3. `CNN 2D` tidak gagal total, tetapi cost/performance-nya kalah jelas dari `CNN 1D`.
4. Bottleneck eksperimen saat ini bukan lagi “memilih antara 1D vs 2D”, karena jawabannya sudah cukup jelas: **pilih 1D**.
5. Langkah berikutnya yang paling rasional bukan lagi mengganti arsitektur dasar, melainkan memperdalam tuning `CNN 1D`, menambah evaluasi per seed, dan menambah analisis kelas.

Secara praktis:

- jika butuh model terbaik saat ini: pilih `CNN 1D tuned`
- jika butuh pembanding arsitektur: simpan `CNN 2D tuned` sebagai runner-up
- jika butuh fokus eksperimen berikutnya: teruskan hanya `CNN 1D`

---

## Audit Trail

Repo ini sengaja menyimpan jejak eksperimen dalam format yang mudah dibaca ulang.

Sumber audit trail utama:

- Ledger seluruh run: [outputs/reports/run_ledger.csv](outputs/reports/run_ledger.csv)
- Status ringkas terbaru: [outputs/reports/latest_status.md](outputs/reports/latest_status.md)
- Ringkasan baseline: [artifacts/summary.json](artifacts/summary.json)

Log detail yang bisa dianalisis ulang:

- baseline `CNN 1D`: [artifacts/cnn1d/history.jsonl](artifacts/cnn1d/history.jsonl)
- baseline `CNN 2D`: [artifacts/cnn2d/history.jsonl](artifacts/cnn2d/history.jsonl)
- tuning trials `CNN 1D`: [artifacts/optuna_cnn1d/trial_logs.jsonl](artifacts/optuna_cnn1d/trial_logs.jsonl)
- tuning final `CNN 1D`: [artifacts/optuna_cnn1d/final_history.jsonl](artifacts/optuna_cnn1d/final_history.jsonl)
- tuning trials `CNN 2D`: [artifacts/optuna_cnn2d/trial_logs.jsonl](artifacts/optuna_cnn2d/trial_logs.jsonl)
- tuning final `CNN 2D`: [artifacts/optuna_cnn2d/final_history.jsonl](artifacts/optuna_cnn2d/final_history.jsonl)

---

## Cara Menjalankan

Contoh baseline:

```bash
python train.py \
  --train-path /workspace/cullpdb+profile_5926_filtered.npy.gz \
  --test-path /workspace/cb513+profile_split1.npy.gz \
  --model cnn1d \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir artifacts/cnn1d
```

Contoh tuning:

```bash
python tune_optuna.py \
  --train-path /workspace/cullpdb+profile_5926_filtered.npy.gz \
  --test-path /workspace/cb513+profile_split1.npy.gz \
  --model cnn1d \
  --trials 8 \
  --epochs 6 \
  --final-epochs 18 \
  --output-dir artifacts/optuna_cnn1d
```

Untuk pembanding 2D, cukup ganti `--model cnn1d` menjadi `--model cnn2d`.

---

## Kesimpulan

Eksperimen ini sudah menghasilkan satu baseline yang jelas, terukur, dan bisa ditelusuri:

- baseline terbaik: `CNN 1D`
- model terbaik saat ini: `tuned CNN 1D`
- skor test terbaik di `CB513`: **0.6820 Q8**

Repo ini belum berhenti di “model jalan”. Repo ini sekarang sudah punya:

- baseline comparison
- hyperparameter tuning
- final best-run selection
- ledger eksperimen
- log training lengkap
- tautan audit trail yang bisa diikuti ulang

Untuk fase berikutnya, jalur paling kuat adalah:

1. lanjutkan tuning hanya pada `CNN 1D`
2. tambahkan evaluasi multi-seed
3. tambahkan analisis per kelas dan confusion pattern
4. baru setelah itu pertimbangkan eksperimen yang lebih jauh seperti distillation
