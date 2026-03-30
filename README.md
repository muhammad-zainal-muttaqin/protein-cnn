# Protein CNN

Eksperimen baseline untuk prediksi secondary structure protein berbasis dataset `CullPDB` dan `CB513`.

## Tujuan

Project ini menjawab pertanyaan praktis berikut:

1. Apakah `CNN 1D` atau `CNN 2D` lebih masuk akal untuk baseline pada dataset ini?
2. Seperti apa pipeline eksperimen yang bersih jika `CullPDB` dipakai sebagai train/validation dan `CB513` dipakai sebagai test final?
3. Kapan `Optuna` berguna, dan kapan `Knowledge Distillation (KD)` layak dicoba?

## Dataset

- `cullpdb+profile_5926_filtered.npy.gz`: dataset train utama
- `cb513+profile_split1.npy.gz`: dataset test final

Catatan penting:

- Meskipun ekstensi file berakhiran `.gz`, isi file yang tersedia di workspace ini adalah array `.npy` langsung.
- Setiap protein dipadatkan ke panjang maksimum `700` residu.
- Bentuk data asli:
  - `CullPDB`: `(5365, 39900)` lalu di-reshape menjadi `(5365, 700, 57)`
  - `CB513`: `(514, 39900)` lalu di-reshape menjadi `(514, 700, 57)`

Interpretasi praktis yang dipakai di repo ini:

- Fitur input per residu: `42` kanal
  - `21` kanal one-hot sequence: `[:, :, 0:21]`
  - `21` kanal profil kontinu: `[:, :, 35:56]`
- Target Q8 per residu: `[:, :, 22:30]`
- Mask padding: `[:, :, 56] == 1`

## Tahapan Percobaan Yang Masuk Akal

1. Verifikasi format data.
   Pastikan split train/test bersih, shape konsisten, dan padding termask dengan benar.

2. Bangun baseline `CNN 1D`.
   Input diperlakukan sebagai sekuens panjang `700` dengan `42` fitur per posisi.

3. Bangun baseline `CNN 2D`.
   Input diperlakukan sebagai grid `700 x 42`, sehingga model bisa menangkap pola lokal lintas residu dan lintas fitur.

4. Split validation dari `CullPDB`.
   Validation diambil dari protein-level split internal `CullPDB`, bukan dari `CB513`.

5. Evaluasi final di `CB513`.
   Test set hanya dipakai sekali setelah konfigurasi model ditetapkan.

6. Tuning hyperparameter dengan `Optuna`.
   Dilakukan setelah baseline jalan dan metriknya stabil.

7. `Knowledge Distillation`.
   Masuk akal jika sudah ada teacher model yang jelas lebih baik dari baseline.

## Ekspektasi Sebelum Training

- `CNN 1D` biasanya menjadi baseline yang lebih kuat dan lebih stabil untuk masalah sequence labeling seperti ini.
- `CNN 2D` bisa kompetitif, tetapi sering butuh desain kernel yang lebih hati-hati agar benar-benar lebih baik.
- `Optuna` bukan pengganti `KD`.
  - `Optuna` dipakai untuk mencari konfigurasi terbaik.
  - `KD` dipakai untuk mentransfer pengetahuan dari teacher ke student.
- Urutan kerja yang defensible:
  1. baseline `CNN 1D`
  2. baseline `CNN 2D`
  3. tuning `Optuna`
  4. baru pertimbangkan `KD`

## Hasil Yang Diharapkan Dari Repo Ini

Repo ini menghasilkan:

- pipeline loading dataset yang bisa langsung dipakai
- training baseline `CNN 1D` dan `CNN 2D`
- metrik `Q8 accuracy` masked per residu
- ringkasan hasil dalam file JSON

## Cara Menjalankan

Simpan dataset pada path berikut:

- `/workspace/cullpdb+profile_5926_filtered.npy.gz`
- `/workspace/cb513+profile_split1.npy.gz`

Contoh training `CNN 1D`:

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

Contoh training `CNN 2D`:

```bash
python train.py \
  --train-path /workspace/cullpdb+profile_5926_filtered.npy.gz \
  --test-path /workspace/cb513+profile_split1.npy.gz \
  --model cnn2d \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-3 \
  --output-dir artifacts/cnn2d
```

## Hasil Baseline

Training dijalankan pada `2026-03-30` dengan GPU `NVIDIA GeForce RTX 3060` (`torch 2.11.0+cu126`).

Konfigurasi:

- split validation: `20%` dari protein di `CullPDB`
- metrik utama: `Q8 accuracy` masked per residu
- optimizer: `Adam`
- epoch:
  - `CNN 1D`: `5`
  - `CNN 2D`: `5`

Ringkasan hasil:

| Model | Best Val Q8 | Test Q8 on CB513 | Catatan |
| --- | ---: | ---: | --- |
| CNN 1D | 0.7061 | 0.6695 | baseline terbaik saat ini |
| CNN 2D | 0.6795 | 0.6427 | lebih lambat dan lebih rendah |

## Insight

Apa yang kita kerjakan sekarang:

1. memvalidasi bentuk dataset `CullPDB` dan `CB513`
2. membangun baseline `CNN 1D` dan `CNN 2D`
3. melatih kedua model dengan split train/validation yang bersih
4. menguji performa final pada `CB513`
5. menyimpan artefak hasil pada folder `artifacts/`

Interpretasi awal:

- `CNN 1D` unggul cukup jelas dibanding `CNN 2D` pada setup baseline ini.
- Selisih test sekitar `2.68` poin absolut Q8 (`0.6695` vs `0.6427`).
- `CNN 2D` juga jauh lebih mahal secara waktu.
  - `CNN 1D`: sekitar `2-3 detik/epoch` setelah warm-up
  - `CNN 2D`: sekitar `21-22 detik/epoch`
- Untuk langkah berikutnya, pendekatan yang paling masuk akal adalah mempertahankan `CNN 1D` sebagai backbone utama.

## Rekomendasi Eksperimen Lanjutan

Urutan yang masuk akal dari titik ini:

1. Tuning `CNN 1D` dengan `Optuna`.
   Cari `learning rate`, `batch size`, `dropout`, `kernel size`, `hidden channels`, dan `weight decay`.

2. Tambahkan validasi yang lebih disiplin.
   Misalnya beberapa seed atau beberapa split agar hasil tidak terlalu bergantung pada satu split acak.

3. Coba `Knowledge Distillation` hanya jika sudah ada teacher yang lebih baik.
   Saat ini belum ada teacher yang lebih kuat dari baseline `CNN 1D`, jadi `KD` belum menjadi prioritas.

4. Pertimbangkan metrik tambahan.
   Selain `Q8`, bisa tambahkan `Q3` dan confusion matrix per kelas agar lebih mudah membaca kelas mana yang paling sulit.

Kesimpulan praktis untuk pertanyaan awal:

- Jika harus memilih langkah berikutnya sekarang, `Optuna` lebih masuk akal daripada `KD`.
- Jika harus memilih arsitektur baseline sekarang, `CNN 1D` adalah pilihan yang lebih kuat daripada `CNN 2D`.
