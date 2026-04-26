# Q3 ResCNN+BiLSTM+Attention - Hyperparameter Search Results

## Setup
- **GPU**: NVIDIA RTX 3090 (24GB)
- **TF**: 2.19.0 / cuDNN 9.3
- **Data**: CullPDB (5365 seqs) → train/val split 80/20
- **Test**: CB513 (514 seqs)
- **Search**: 20 trials Optuna (TPE sampler), 8 epochs each
- **Final**: 40 epochs with best params

## Completed Trials (18/20)

| Trial | Val Macro-F1 | Filters | Conv | Kernel | LSTM | Attn Heads | Attn Dim | Dense | Dropout | L2 | LR | Focal γ | Label Smooth | Batch |
|-------|-------------|---------|------|--------|------|-----------|---------|-------|---------|------|------|---------|-------------|-------|
| 0 | 0.3924 | 96 | 2 | 5 | 64 | 4 | 32 | 128 | 0.4250 | 4.5e-06 | 2.44e-04 | 2.064 | 0.0101 | 24 |
| 1 | 0.3906 | 160 | 2 | 7 | 128 | 4 | 24 | 64 | 0.2329 | 1.2e-05 | 1.61e-04 | 2.463 | 0.0262 | 24 |
| 2 | 0.4226 | 128 | 3 | 9 | 128 | 4 | 24 | 64 | 0.4293 | 6.5e-06 | 5.84e-04 | 1.556 | 0.0391 | 8 |
| 3 | 0.3988 | 96 | 3 | 7 | 64 | 4 | 48 | 128 | 0.4407 | 4.0e-06 | 1.61e-04 | 2.468 | 0.0025 | 16 |
| 4 | 0.4168 | 96 | 4 | 9 | 96 | 4 | 48 | 96 | 0.3206 | 1.9e-05 | 3.01e-04 | 1.525 | 0.0502 | 16 |
| 5 | 0.4553 | 160 | 4 | 5 | 96 | 4 | 48 | 128 | 0.1877 | 1.6e-06 | 4.38e-04 | 0.840 | 0.0543 | 16 |
| 6 | 0.4700 | 128 | 4 | 5 | 64 | 4 | 24 | 96 | 0.3148 | 1.4e-05 | 6.55e-04 | 1.573 | 0.0259 | 8 |
| 7 | 0.3961 | 160 | 3 | 9 | 128 | 4 | 32 | 128 | 0.2057 | 2.1e-05 | 1.40e-04 | 1.737 | 0.0219 | 24 |
| 8 | 0.4420 | 96 | 3 | 7 | 96 | 4 | 32 | 96 | 0.2702 | 1.9e-06 | 1.42e-03 | 1.320 | 0.0369 | 24 |
| 9 | 0.4616 | 96 | 3 | 7 | 128 | 2 | 24 | 128 | 0.1687 | 3.0e-06 | 1.90e-03 | 0.985 | 0.0303 | 8 |
| 10 | 0.4462 | 128 | 4 | 5 | 64 | 2 | 24 | 96 | 0.3471 | 4.0e-05 | 8.41e-04 | 1.999 | 0.0103 | 8 |
| 11 | 0.4647 | 128 | 4 | 7 | 64 | 2 | 24 | 96 | 0.3500 | 3.0e-06 | 1.98e-03 | 0.834 | 0.0213 | 8 |
| 12 | 0.4399 | 128 | 4 | 5 | 64 | 2 | 24 | 96 | 0.3734 | 9.9e-06 | 1.29e-03 | 1.175 | 0.0180 | 8 |
| 13 | 0.4710 | 128 | 4 | 5 | 64 | 2 | 24 | 96 | 0.2722 | 2.5e-06 | 8.36e-04 | 1.187 | 0.0366 | 8 |
| 14 | 0.4682 | 128 | 4 | 5 | 64 | 2 | 24 | 96 | 0.2760 | 1.2e-06 | 7.88e-04 | 1.242 | 0.0439 | 8 |
| **15** | **0.5040** | **128** | **4** | **5** | **64** | **2** | **24** | **96** | **0.2542** | **4.7e-05** | **8.55e-04** | **1.726** | **0.0348** | **8** |
| 16 | 0.4943 | 128 | 4 | 5 | 64 | 2 | 24 | 96 | 0.2402 | 4.4e-05 | 1.10e-03 | 1.837 | 0.0595 | 8 |
| 17 | 0.4699 | 128 | 3 | 5 | 64 | 2 | 24 | 64 | 0.2319 | 4.6e-05 | 1.15e-03 | 1.867 | 0.0593 | 8 |

## Best Configuration (Trial 15)
```
filters:           128
conv_blocks:       4
kernel_size:       5
lstm_units:        64
attention_heads:   2
attention_key_dim: 24
dense_units:       96
dropout:           0.2542
weight_decay:      4.73e-05
learning_rate:     8.55e-04
focal_gamma:       1.726
label_smoothing:   0.0348
batch_size:        8
```

**Val Macro-F1: 0.5040**

## Final Training (40 epochs, Trial 15 params)

### CB513 Test Results
| Metric | Value |
|--------|-------|
| Accuracy | **82.44%** |
| Balanced Accuracy | 70.17% |
| Precision (macro) | 50.41% |
| Recall (macro) | 70.17% |
| F1 (macro) | 45.44% |
| AUC (OVR) | **0.9814** |
| Test Loss | 0.0453 |
| Best Val Macro-F1 | **0.5033** |

### Per-Class Performance (Helix=0, Sheet=1, Coil=2)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Helix) | 1.0000 | 0.8492 | 0.9184 | 340,699 |
| 1 (Sheet) | 0.4937 | 0.3482 | 0.4084 | 17,920 |
| 2 (Coil) | 0.0185 | 0.9077 | 0.0363 | 1,181 |

### Confusion Matrix
```
[[289310   6290  45099]
 [     0   6240  11680]
 [     0    109   1072]]
```

### Key Observations
- Best architecture uses small kernel (5), deep conv (4 blocks), small LSTM (64 units), few attention heads (2)
- Batch size 8 consistently outperforms 16/24
- Label smoothing of 0.03-0.05 helps
- TPE converges to a narrow region: filters=128, conv_blocks=4, kernel_size=5 around trial 9+
- Class imbalance (340k helix vs 1k coil) strongly affects macro metrics
- Helix prediction is near-perfect; sheet/coil recall is low but AUC remains high (0.98)
