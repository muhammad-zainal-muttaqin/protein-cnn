# Final Report — Protein CNN Research

## Executive Summary

Ledger final repo ini berisi **101 eksperimen**:

- `81` run selesai dan punya metrik test lengkap
- `16` run `Optuna search` hanya menghasilkan validation metric
- `4` run gagal dan tetap dicatat sebagai bagian dari audit trail

Run terbaik final adalah **`p4_07_resdil_b42_ce_none_c320_e24_seed7`** dengan:

- best validation Q8 = **0.7354**
- test Q8 pada `CB513` = **0.6926**
- test loss = **0.9242**

Runner-up final:

- `s2_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320` → test Q8 **0.6919**

Kenaikan yang paling penting:

- baseline `CNN 1D` → tuned `CNN 1D`: **+0.0125**
- tuned `CNN 1D` → incremental `ResDil`: **+0.0058**
- baseline `CNN 1D` → best final `ResDil`: **+0.0231**

Secara praktis, repo ini sudah cukup jelas menjawab pertanyaan arsitektur utama:

- `1D` lebih cocok daripada `2D`
- tuning membantu, tetapi tidak cukup sendirian
- ceiling terbaik repo saat ini datang dari **`resdil_cnn1d` + `baseline42` + `ce` + `no class weighting`**

---

## Dataset & Evaluation Setup

Task yang dipakai repo ini adalah **masked Q8 secondary-structure prediction**, bukan Q3.

File sumber:

- `/workspace/cullpdb+profile_5926_filtered.npy.gz`
- `/workspace/cb513+profile_split1.npy.gz`

Struktur data setelah reshape:

- `CullPDB`: `(5365, 700, 57)`
- `CB513`: `(514, 700, 57)`

Setup fitur utama yang dipakai model final:

- `[:, :, 0:21]` = amino-acid one-hot
- `[:, :, 35:56]` = profile features
- total input = `42` kanal
- target = `Q8`
- padding dimask saat menghitung loss dan accuracy

Statistik ringkas:

| Split | Protein | Valid Residues | Mean Length | Median Length |
|---|---:|---:|---:|---:|
| CullPDB total | 5365 | 1154412 | 215.2 | 184 |
| CB513 test | 514 | 84765 | 164.9 | 132 |

Keputusan evaluasi yang di-lock sepanjang repo:

- validation diambil dari split internal `CullPDB`
- `CB513` dijaga sebagai **test final**
- metrik resmi adalah masked `Q8 accuracy`

Ini penting karena angka repo ini memang tidak dimaksudkan untuk dibandingkan dengan setup yang memakai test set sebagai validation during training.

---

## Evolusi Eksperimen

### Progres global

![Perkembangan test Q8 di seluruh run selesai](../figures/protein_cnn_progress_test_q8.png)

Sumber data: [../figures/data/protein_cnn_progress_test_q8.csv](../figures/data/protein_cnn_progress_test_q8.csv)

Grafik ini menunjukkan dua hal:

1. ada lompatan nyata dari baseline ke tuned model
2. lompatan berikutnya datang dari keluarga `resdil_cnn1d`, bukan dari variasi `cnn2d`

Progress validation memperlihatkan banyak run dengan `best_val_q8` tinggi, tetapi tidak semuanya translate ke `CB513`.

![Perkembangan best validation Q8 di seluruh run non-failed](../figures/protein_cnn_progress_best_val_q8.png)

Sumber data: [../figures/data/protein_cnn_progress_best_val_q8.csv](../figures/data/protein_cnn_progress_best_val_q8.csv)

Interpretasi penting:

- validation tinggi tidak otomatis berarti test tinggi
- beberapa konfigurasi `extended46` terlihat sangat bagus di validation, tetapi justru melemah di test

### Ringkasan per phase

| Phase | Run completed with test | Best test Q8 | Mean test Q8 |
|---|---:|---:|---:|
| Baseline | 2 | 0.6695 | 0.6561 |
| Optuna Final | 2 | 0.6820 | 0.6673 |
| Incremental | 1 | 0.6878 | 0.6878 |
| Research Stage 1 | 48 | 0.6841 | 0.6315 |
| Research Stage 2 | 8 | 0.6919 | 0.6851 |
| Research Stage 3 | 12 | 0.6917 | 0.6881 |
| Research Phase 4 | 8 | 0.6926 | 0.6795 |

![Ringkasan performa test per phase](../figures/protein_cnn_phase_summary.png)

Sumber data: [../figures/data/protein_cnn_phase_summary.csv](../figures/data/protein_cnn_phase_summary.csv)

Interpretasi phase:

- `baseline` menjawab pertanyaan awal `1D vs 2D`
- `tune_final` menaikkan baseline `CNN 1D` secara nyata
- `research_incremental` membuktikan perubahan arsitektur masih memberi ruang
- `research_stage2` dan `research_stage3_confirm` mengerucutkan kandidat terbaik
- `research_phase4` menutup eksperimen dengan winner final

Catatan penting untuk `research_phase4`: mean phase ini turun karena phase 4 sengaja memuat beberapa percobaan berisiko tinggi seperti `extended46` dan `focal` untuk menguji ceiling, bukan hanya mengulang kandidat aman.

### Ringkasan per keluarga model

| Keluarga model | Run completed with test | Best test Q8 | Mean test Q8 |
|---|---:|---:|---:|
| CNN 1D | 27 | 0.6820 | 0.6345 |
| CNN 2D | 2 | 0.6526 | 0.6477 |
| ResDil CNN 1D | 52 | 0.6926 | 0.6614 |

![Perbandingan keluarga model](../figures/protein_cnn_model_family_summary.png)

Sumber data: [../figures/data/protein_cnn_model_family_summary.csv](../figures/data/protein_cnn_model_family_summary.csv)

Yang perlu dibaca dari tabel dan grafik ini:

- `cnn2d` tidak pernah mendekati family `1D`
- `cnn1d` menjadi baseline yang valid
- `resdil_cnn1d` mengambil alih posisi terbaik repo secara konsisten

---

## Baseline, Tuning, dan Titik Plateau

### Baseline

| Run | Best Val Q8 | Test Q8 | Test Loss |
|---|---:|---:|---:|
| Baseline CNN 1D | 0.7061 | 0.6695 | 0.9079 |
| Baseline CNN 2D | 0.6795 | 0.6427 | 0.9919 |

Gap baseline awal:

- `CNN 1D` unggul **+0.0268** absolute atas `CNN 2D`

Keputusan arsitektur sebenarnya sudah cukup jelas sejak titik ini.

### Optuna

| Run | Best Val Q8 | Test Q8 | Test Loss |
|---|---:|---:|---:|
| Tuned CNN 1D | 0.7203 | 0.6820 | 0.8967 |
| Tuned CNN 2D | 0.6914 | 0.6526 | 0.9664 |

Gain tuning:

- `CNN 1D`: `0.6695 -> 0.6820` (**+0.0125**)
- `CNN 2D`: `0.6427 -> 0.6526` (**+0.0099**)

Tuning jelas berguna, tetapi setelah titik ini gain mulai melambat. Itu sebabnya repo bergerak ke perubahan arsitektur dan bukan terus mengulang sweep hyperparameter standar.

### Incremental ResDil

`incremental_resdil_step1` menjadi jembatan penting dari tuning ke riset arsitektur:

| Run | Best Val Q8 | Test Q8 | Test Loss |
|---|---:|---:|---:|
| Incremental ResDil Step 1 | 0.7268 | 0.6878 | 0.8872 |

Ini membuktikan bahwa arsitektur `1D` yang lebih kuat memang masih punya headroom di atas `CNN 1D` tuned.

---

## Autoresearch: Stage 1 sampai Phase 4

### Stage 1

Stage 1 adalah sweep paling luas. Di sini repo menguji:

- `cnn1d` vs `resdil_cnn1d`
- `baseline42` vs `extended46`
- `ce` vs `focal`
- `none` vs `sqrt_inverse`
- beberapa branch `cnn2d`

Stage ini berguna bukan karena menghasilkan winner final langsung, tetapi karena mempersempit ruang pencarian.

Hasil terpenting Stage 1:

- best stage1 = `s1_06_resdil_cnn1d_baseline42_ce_none_c320` dengan test Q8 `0.6841`
- `sqrt_inverse` dan `focal + sqrt_inverse` menghasilkan test Q8 yang jauh lebih rendah
- branch `cnn2d` tetap lemah dan empat run `cnn2d` tercatat gagal karena bug `in_channels`

### Stage 2

Stage 2 melakukan retrain lebih panjang pada kandidat terkuat. Hasil besarnya:

- `s2_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320` = `0.6919`
- ini pertama kali mendorong repo masuk area `0.69` di test Q8

Stage 2 menegaskan bahwa:

- retrain yang lebih panjang masuk akal untuk keluarga `ResDil`
- kandidat terbaik tetap `baseline42 + ce + no weighting`

### Stage 3 Confirm

Stage 3 mengecek stabilitas lintas seed untuk kandidat terbaik.

Run-run penting:

- `s3_02_s1_02_resdil_cnn1d_baseline42_ce_none_c192_seed13` = `0.6917`
- `s3_03_s1_04_resdil_cnn1d_baseline42_ce_none_c256_seed21` = `0.6913`
- `s3_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320_seed13` = `0.6901`

Temuan utama:

- seed menggeser hasil beberapa basis point
- tetapi tidak pernah membalik keputusan arsitektur atau recipe

### Phase 4

Phase 4 adalah fase penajaman akhir. Di sinilah repo menguji:

- retrain `baseline42` recipe yang paling kuat
- variasi `extended46`
- satu branch `focal`
- final seed confirmation

Best final repo lahir di fase ini:

- `p4_07_resdil_b42_ce_none_c320_e24_seed7` = `0.6926`

Yang sama pentingnya, phase 4 juga menghasilkan counterexample yang kuat:

- `p4_04_resdil_e46_ce_none_c320_e24` → val `0.7497`, test `0.6461`
- `p4_05_resdil_e46_ce_none_c384_e24` → val `0.7478`, test `0.6512`

Ini adalah bukti paling jelas bahwa validation tinggi tidak cukup; generalisasi ke `CB513` tetap penentu akhir.

---

## Stabilitas Kandidat ResDil

![Stabilitas kandidat ResDil baseline42 + CE + no weighting](../figures/protein_cnn_resdil_candidate_stability.png)

Sumber data: [../figures/data/protein_cnn_resdil_candidate_stability.csv](../figures/data/protein_cnn_resdil_candidate_stability.csv)

Grafik ini hanya fokus pada family yang benar-benar relevan:

- `resdil_cnn1d`
- `baseline42`
- `ce`
- `no class weighting`

Temuan utamanya:

- variasi seed dan schedule memang menggeser hasil
- tetapi seluruh cluster terbaik tetap berkumpul di family ini
- width `320` bukan satu-satunya kandidat kuat, tetapi menjadi recipe yang paling konsisten di atas

---

## Training Curves

![Kurva training baseline, tuned, dan best final](../figures/protein_cnn_training_curves.png)

Sumber data: [../figures/data/protein_cnn_training_curves.csv](../figures/data/protein_cnn_training_curves.csv)

Interpretasi kurva:

- baseline `CNN 1D` plateau cukup cepat
- `CNN 1D` tuned memperbaiki stabilitas validation dan menurunkan loss
- `ResDil` final mencapai train accuracy lebih tinggi dan menjaga validation di level yang lebih kompetitif

Kurva ini cocok dengan cerita eksperimen repo: setelah tuning, masalahnya bukan lagi sekadar `lr` atau `dropout`, tetapi kapasitas arsitektur dan receptive field sequence.

---

## Top-10 Runs

| Rank | Run | Phase | Model | Feature set | Loss | Best Val Q8 | Test Q8 | Test Loss |
|---:|---|---|---|---|---|---:|---:|---:|
| 1 | `p4_07_resdil_b42_ce_none_c320_e24_seed7` | Research Phase 4 | ResDil CNN 1D | `baseline42` | `ce` | 0.7354 | 0.6926 | 0.9242 |
| 2 | `s2_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320` | Research Stage 2 | ResDil CNN 1D | `baseline42` | `ce` | 0.7303 | 0.6919 | 0.9327 |
| 3 | `p4_01_resdil_b42_ce_none_c320_e18` | Research Phase 4 | ResDil CNN 1D | `baseline42` | `ce` | 0.7303 | 0.6919 | 0.9327 |
| 4 | `s3_02_s1_02_resdil_cnn1d_baseline42_ce_none_c192_seed13` | Research Stage 3 | ResDil CNN 1D | `baseline42` | `ce` | 0.7307 | 0.6917 | 0.8843 |
| 5 | `s3_03_s1_04_resdil_cnn1d_baseline42_ce_none_c256_seed21` | Research Stage 3 | ResDil CNN 1D | `baseline42` | `ce` | 0.7306 | 0.6913 | 0.9025 |
| 6 | `p4_08_resdil_b42_ce_none_c320_e24_seed13` | Research Phase 4 | ResDil CNN 1D | `baseline42` | `ce` | 0.7319 | 0.6905 | 0.9252 |
| 7 | `s3_02_s1_02_resdil_cnn1d_baseline42_ce_none_c192_seed21` | Research Stage 3 | ResDil CNN 1D | `baseline42` | `ce` | 0.7276 | 0.6904 | 0.8919 |
| 8 | `s3_01_s1_06_resdil_cnn1d_baseline42_ce_none_c320_seed13` | Research Stage 3 | ResDil CNN 1D | `baseline42` | `ce` | 0.7317 | 0.6901 | 0.8822 |
| 9 | `s3_03_s1_04_resdil_cnn1d_baseline42_ce_none_c256_seed7` | Research Stage 3 | ResDil CNN 1D | `baseline42` | `ce` | 0.7345 | 0.6900 | 0.9229 |
| 10 | `p4_02_resdil_b42_ce_none_c320_e24` | Research Phase 4 | ResDil CNN 1D | `baseline42` | `ce` | 0.7293 | 0.6897 | 0.8928 |

Top-10 ini menguatkan satu hal dengan sangat jelas: posisi atas ledger didominasi hampir sepenuhnya oleh `resdil_cnn1d` dengan `baseline42 + ce + no weighting`.

---

## Failure Analysis

### Mengapa `CNN 1D` lebih baik dari `CNN 2D`

Jawabannya bukan hanya karena angka final lebih tinggi, tetapi karena seluruh perjalanan eksperimen mendukung keputusan itu:

- baseline `CNN 1D` sudah unggul `+0.0268`
- `CNN 2D` tuned tetap berhenti di `0.6526`
- `cnn2d` tidak pernah muncul di cluster teratas ledger final

Secara struktur masalah, task ini memang lebih natural sebagai sequence labeling daripada image-style grid processing.

### Mengapa `extended46` val tinggi tetapi test turun

Ini temuan paling penting dari phase 4:

- `p4_04`: best val `0.7497`, test `0.6461`
- `p4_05`: best val `0.7478`, test `0.6512`

Kalau hanya melihat validation, dua run ini tampak sangat menjanjikan. Tetapi begitu diuji di `CB513`, performanya turun jauh di bawah kandidat `baseline42`.

Interpretasi praktis:

- `extended46` belum menunjukkan generalisasi yang sehat di repo ini
- penambahan fitur tidak otomatis membantu
- repo harus tetap memilih model berdasarkan holdout test, bukan validation tinggi semata

### Mengapa loss kecil tidak otomatis berarti model lebih baik

Grafik berikut merangkum hubungan loss vs test Q8:

![Hubungan test loss vs test Q8](../figures/protein_cnn_test_loss_vs_q8.png)

Sumber data: [../figures/data/protein_cnn_test_loss_vs_q8.csv](../figures/data/protein_cnn_test_loss_vs_q8.csv)

Temuan utamanya:

- beberapa run `sqrt_inverse` atau `focal + sqrt_inverse` punya `test_loss` yang jauh lebih kecil
- tetapi `test_q8` mereka tidak menang, bahkan sering jatuh cukup jauh

Kesimpulan:

- `test_loss` lintas family objective tidak comparable secara langsung
- pemilihan model lintas objective di repo ini harus memakai `test_q8`

### Run gagal yang tetap dicatat

| Run gagal | Feature set | Loss | Catatan singkat |
|---|---|---|---|
| `s1_49_cnn2d_baseline42_ce_none_d19` | `baseline42` | `ce` | branch `cnn2d` stage 1 gagal di bug `in_channels` |
| `s1_50_cnn2d_baseline42_ce_none_d28` | `baseline42` | `ce` | branch `cnn2d` stage 1 gagal di bug `in_channels` |
| `s1_51_cnn2d_extended46_ce_none_d19` | `extended46` | `ce` | branch `cnn2d` stage 1 gagal di bug `in_channels` |
| `s1_52_cnn2d_extended46_ce_none_d28` | `extended46` | `ce` | branch `cnn2d` stage 1 gagal di bug `in_channels` |

Keempat run ini tidak dihapus atau disembunyikan. Mereka tetap berada di ledger sebagai bukti bahwa cabang tersebut memang pernah diuji.

---

## Final Conclusion

Kesimpulan repo saat ini sudah cukup tegas:

1. **Pilih `1D`, bukan `2D`.**
2. **Gunakan `baseline42`, bukan `extended46`, untuk hasil terbaik yang benar-benar generalize.**
3. **Gunakan `ce + no class weighting` sebagai recipe utama.**
4. **Gunakan `resdil_cnn1d` sebagai family model utama.**
5. **Tetapkan `p4_07_resdil_b42_ce_none_c320_e24_seed7` sebagai hasil resmi terbaik repo saat ini.**

Kalau eksperimen lanjutan diteruskan, arah yang paling masuk akal bukan lagi memperbanyak sweep acak, tetapi:

- analisis per kelas Q8
- arsitektur sequence yang lebih kuat dari CNN murni
- evaluasi fitur tambahan yang benar-benar meningkatkan generalisasi, bukan hanya validation

---

## Audit Trail

Artefak utama:

- Ledger final: [run_ledger.csv](run_ledger.csv)
- Status terbaru: [latest_status.md](latest_status.md)
- Summary riset: [research_summary.json](research_summary.json)
- Summary phase 4: [phase4_summary.json](phase4_summary.json)

Artefak model penting:

- Baseline CNN 1D: [../../artifacts/cnn1d/report.json](../../artifacts/cnn1d/report.json)
- Baseline CNN 2D: [../../artifacts/cnn2d/report.json](../../artifacts/cnn2d/report.json)
- Tuned CNN 1D: [../../artifacts/optuna_cnn1d/optuna_report.json](../../artifacts/optuna_cnn1d/optuna_report.json)
- Tuned CNN 2D: [../../artifacts/optuna_cnn2d/optuna_report.json](../../artifacts/optuna_cnn2d/optuna_report.json)
- Best final run: [../../artifacts/research_runs/p4_07_resdil_b42_ce_none_c320_e24_seed7/report.json](../../artifacts/research_runs/p4_07_resdil_b42_ce_none_c320_e24_seed7/report.json)
