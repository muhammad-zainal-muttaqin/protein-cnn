# -*- coding: utf-8 -*-
"""GPU-cloud Jupyter ready — Q3 protein secondary structure experiment.

Model: residual Conv1D + BiLSTM + self-attention.
Objective: tune for validation macro-F1, then evaluate on CB513.

Run on a fresh machine that only has Jupyter + PyTorch pre-installed.
All other dependencies are installed in Cell 0 below.

Dataset files downloaded automatically from Google Drive:
- cullpdb+profile_5926_filtered.npy.gz  (train/val)
- cb513+profile_split1.npy.gz           (test)
"""

# =============================================================================
# CELL 0 — Install all dependencies  (run once, then restart kernel if asked)
# =============================================================================

# %%
import subprocess
import sys


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


# Core ML stack
_pip("tensorflow[and-cuda]")          # TF + GPU support; falls back to CPU if no CUDA
_pip("scikit-learn")
_pip("optuna")
_pip("matplotlib")

# Dataset download helper
_pip("gdown")

print("All dependencies installed.")

# =============================================================================
# CELL 1 — Download datasets from Google Drive
# =============================================================================

# %%
import os
from pathlib import Path

import gdown

# ── Edit these two IDs if links change ────────────────────────────────────────
# First  link → cullpdb+profile_5926_filtered.npy.gz   (train/val, ~120 MB)
# Second link → cb513+profile_split1.npy.gz             (test, ~12 MB)
GDRIVE_ID_TRAIN = "1LzTB2ifuTO976Mrz8v2qSP3PUNFlymBC"
GDRIVE_ID_TEST  = "1ysmn20jCL2xQ-PY1vT2_11gsfz8MGI7K"
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

TRAIN_FILE = "cullpdb+profile_5926_filtered.npy.gz"
TEST_FILE  = "cb513+profile_split1.npy.gz"

train_path = DATA_DIR / TRAIN_FILE
test_path  = DATA_DIR / TEST_FILE

if not train_path.exists():
    print("Downloading training dataset …")
    gdown.download(id=GDRIVE_ID_TRAIN, output=str(train_path), quiet=False)
else:
    print("Training dataset already present:", train_path)

if not test_path.exists():
    print("Downloading test dataset …")
    gdown.download(id=GDRIVE_ID_TEST, output=str(test_path), quiet=False)
else:
    print("Test dataset already present:", test_path)

print("Data directory contents:", list(DATA_DIR.iterdir()))

# =============================================================================
# CELL 2 — Imports & global settings
# =============================================================================

# %%
from __future__ import annotations

import json
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
    SeparableConv1D,
    SpatialDropout1D,
    TimeDistributed,
)
from tensorflow.keras.regularizers import l2

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = str(DATA_DIR)                  # local ./data folder
OUTPUT_ROOT = "./output"
RUN_NAME    = f"rescnn_bilstm_attention_q3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, RUN_NAME)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Experiment knobs ──────────────────────────────────────────────────────────
SEED          = 42
MAX_LEN       = 700
N_CLASSES     = 3
VAL_FRACTION  = 0.20

# Fast sanity check  → TRIALS=3,  SEARCH_EPOCHS=3,  FINAL_EPOCHS=8
# Serious run        → TRIALS=20, SEARCH_EPOCHS=8,  FINAL_EPOCHS=40
TRIALS        = 20
SEARCH_EPOCHS = 8
FINAL_EPOCHS  = 40

print("TensorFlow version :", tf.__version__)
print("GPU devices        :", tf.config.list_physical_devices("GPU"))
print("Output directory   :", OUTPUT_DIR)

# =============================================================================
# CELL 3 — Reproducibility
# =============================================================================

# %%
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


set_seed(SEED)

run_config = {
    "base_dir":        BASE_DIR,
    "output_root":     OUTPUT_ROOT,
    "output_dir":      OUTPUT_DIR,
    "train_file":      TRAIN_FILE,
    "test_file":       TEST_FILE,
    "seed":            SEED,
    "max_len":         MAX_LEN,
    "n_classes":       N_CLASSES,
    "trials":          TRIALS,
    "search_epochs":   SEARCH_EPOCHS,
    "final_epochs":    FINAL_EPOCHS,
    "val_fraction":    VAL_FRACTION,
    "created_at":      datetime.now().isoformat(timespec="seconds"),
}

with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)

# =============================================================================
# CELL 4 — Data loading
# =============================================================================

# %%
def load_raw_dataset(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = np.load(path, allow_pickle=True)
    return data.reshape(data.shape[0], MAX_LEN, 57)


def extract_q3(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove label channels 21:24 from input features, return (X, y, mask)."""
    y      = dataset[:, :, 21:24]
    x_aa   = dataset[:, :, :21]
    x_other = dataset[:, :, 24:]
    x      = np.concatenate([x_aa, x_other], axis=2)
    mask   = np.sum(x, axis=2) != 0
    return x.astype(np.float32), y.astype(np.float32), mask


cullpdb = load_raw_dataset(os.path.join(BASE_DIR, TRAIN_FILE))
cb513   = load_raw_dataset(os.path.join(BASE_DIR, TEST_FILE))

# Sanity check: cullpdb should be much larger than cb513
print(f"cullpdb shape: {cullpdb.shape}   cb513 shape: {cb513.shape}")
assert cullpdb.shape[0] > cb513.shape[0], (
    "Shapes look swapped — swap GDRIVE_ID_TRAIN / GDRIVE_ID_TEST and re-run Cell 1."
)

rng     = np.random.default_rng(SEED)
indices = rng.permutation(cullpdb.shape[0])
split   = int((1.0 - VAL_FRACTION) * cullpdb.shape[0])

X_train, y_train, mask_train = extract_q3(cullpdb[indices[:split]])
X_val,   y_val,   mask_val   = extract_q3(cullpdb[indices[split:]])
X_test,  y_test,  mask_test  = extract_q3(cb513)

print("X_train:", X_train.shape, "| X_val:", X_val.shape, "| X_test:", X_test.shape)
print("Valid residues  train/val/test:",
      int(mask_train.sum()), int(mask_val.sum()), int(mask_test.sum()))

y_train_labels   = np.argmax(y_train, axis=-1)
flat_train_labels = y_train_labels[mask_train]
class_weights    = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(N_CLASSES),
    y=flat_train_labels,
).astype(np.float32)
print("Balanced class weights (raw):", class_weights)


def make_sample_weight(mask: np.ndarray, y_one_hot: np.ndarray, weights: np.ndarray) -> np.ndarray:
    labels = np.argmax(y_one_hot, axis=-1)
    return mask.astype(np.float32) * weights[labels]


sw_train = make_sample_weight(mask_train, y_train, class_weights)
sw_val   = make_sample_weight(mask_val,   y_val,   class_weights)
sw_test  = mask_test.astype(np.float32)

# =============================================================================
# CELL 5 — Metrics and callbacks
# =============================================================================

# %%
def macro_f1_masked(y_true: np.ndarray, y_pred_prob: np.ndarray, mask: np.ndarray) -> float:
    valid      = mask.astype(bool).reshape(-1)
    true_flat  = y_true.reshape(-1, N_CLASSES)[valid]
    pred_flat  = y_pred_prob.reshape(-1, N_CLASSES)[valid]
    true_label = np.argmax(true_flat, axis=1)
    pred_label = np.argmax(pred_flat, axis=1)
    return float(f1_score(true_label, pred_label, average="macro", zero_division=0))


class ValMacroF1Checkpoint(Callback):
    """Evaluate masked validation macro-F1 and save the best weights."""

    def __init__(
        self,
        x_val: np.ndarray,
        y_val_data: np.ndarray,
        mask_val_data: np.ndarray,
        filepath: str | None = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.x_val        = x_val
        self.y_val_data   = y_val_data
        self.mask_val_data = mask_val_data
        self.filepath     = filepath
        self.batch_size   = batch_size
        self.best         = -1.0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs  = logs or {}
        pred  = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        score = macro_f1_masked(self.y_val_data, pred, self.mask_val_data)
        logs["val_macro_f1"] = score
        print(f" val_macro_f1={score:.4f}")
        if score > self.best:
            self.best = score
            if self.filepath:
                self.model.save_weights(self.filepath)
                print(f" saved best weights → {self.filepath}  ({score:.4f})")

# =============================================================================
# CELL 6 — Loss and model architecture
# =============================================================================

# %%
def categorical_focal_loss(gamma: float = 1.5, label_smoothing: float = 0.0):
    """Focal categorical crossentropy compatible with sample_weight masks."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if label_smoothing > 0:
            y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / N_CLASSES
        else:
            y_true_smooth = y_true
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce             = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred_clipped), axis=-1)
        pt             = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)
        focal_weight   = tf.pow(1.0 - pt, gamma)
        return focal_weight * ce

    return loss


def residual_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    weight_decay: float,
) -> tf.Tensor:
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same", kernel_regularizer=l2(weight_decay))(shortcut)

    x = SeparableConv1D(
        filters, kernel_size, padding="same", dilation_rate=dilation_rate, activation="relu",
        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
    )(x)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(dropout)(x)
    x = SeparableConv1D(
        filters, kernel_size, padding="same", dilation_rate=dilation_rate, activation="relu",
        depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
    )(x)
    x = Add()([shortcut, x])
    return LayerNormalization()(x)


def build_model(
    filters: int           = 128,
    conv_blocks: int       = 3,
    kernel_size: int       = 7,
    lstm_units: int        = 96,
    attention_heads: int   = 4,
    attention_key_dim: int = 32,
    dense_units: int       = 96,
    dropout: float         = 0.30,
    weight_decay: float    = 1e-5,
) -> Model:
    inp = Input(shape=(MAX_LEN, 54), name="features")
    x   = Conv1D(filters, 1, padding="same", activation="relu", kernel_regularizer=l2(weight_decay))(inp)
    x   = LayerNormalization()(x)

    dilation_cycle = [1, 2, 4, 8]
    for block_idx in range(conv_blocks):
        x = residual_conv_block(
            x=x, filters=filters, kernel_size=kernel_size,
            dilation_rate=dilation_cycle[block_idx % len(dilation_cycle)],
            dropout=dropout, weight_decay=weight_decay,
        )

    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=dropout,
             recurrent_dropout=0.0, kernel_regularizer=l2(weight_decay))
    )(x)
    x = LayerNormalization()(x)

    attention = MultiHeadAttention(
        num_heads=attention_heads, key_dim=attention_key_dim, dropout=dropout,
    )(x, x)
    x = Add()([x, attention])
    x = LayerNormalization()(x)

    x   = TimeDistributed(Dense(dense_units, activation="relu", kernel_regularizer=l2(weight_decay)))(x)
    x   = Dropout(dropout)(x)
    out = TimeDistributed(Dense(N_CLASSES, activation="softmax"), name="q3")(x)
    return Model(inp, out)


def compile_model(model: Model, lr: float, gamma: float, label_smoothing: float) -> None:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=categorical_focal_loss(gamma=gamma, label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

# =============================================================================
# CELL 7 — Optuna hyperparameter search
# =============================================================================

# %%
def objective(trial: optuna.Trial) -> float:
    tf.keras.backend.clear_session()
    set_seed(SEED + trial.number)

    params = {
        "filters":           trial.suggest_categorical("filters",           [96, 128, 160]),
        "conv_blocks":       trial.suggest_int(        "conv_blocks",       2, 4),
        "kernel_size":       trial.suggest_categorical("kernel_size",       [5, 7, 9]),
        "lstm_units":        trial.suggest_categorical("lstm_units",        [64, 96, 128]),
        "attention_heads":   trial.suggest_categorical("attention_heads",   [2, 4]),
        "attention_key_dim": trial.suggest_categorical("attention_key_dim", [24, 32, 48]),
        "dense_units":       trial.suggest_categorical("dense_units",       [64, 96, 128]),
        "dropout":           trial.suggest_float(      "dropout",           0.15, 0.45),
        "weight_decay":      trial.suggest_float(      "weight_decay",      1e-6, 5e-5, log=True),
    }
    lr              = trial.suggest_float("lr",              1e-4, 2e-3, log=True)
    gamma           = trial.suggest_float("focal_gamma",     0.8,  2.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0,  0.06)
    batch_size      = trial.suggest_categorical("batch_size", [8, 16, 24])

    model = build_model(**params)
    compile_model(model, lr=lr, gamma=gamma, label_smoothing=label_smoothing)

    f1_callback = ValMacroF1Checkpoint(X_val, y_val, mask_val, filepath=None, batch_size=32)
    callbacks = [
        f1_callback,
        EarlyStopping(monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        validation_data=(X_val, y_val, sw_val),
        epochs=SEARCH_EPOCHS,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    trial.set_user_attr("best_val_macro_f1", f1_callback.best)
    trial.set_user_attr("params_full", {
        **params, "lr": lr, "focal_gamma": gamma,
        "label_smoothing": label_smoothing, "batch_size": batch_size,
    })
    return f1_callback.best


study = optuna.create_study(
    direction="maximize",
    study_name="q3_rescnn_bilstm_attention_macro_f1",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
)
study.optimize(objective, n_trials=TRIALS, gc_after_trial=True)

print("Best trial :", study.best_trial.number)
print("Best val macro-F1 :", study.best_value)
print("Best params :", study.best_params)

try:
    study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)
except Exception as exc:
    print(f"Could not save optuna_trials.csv: {exc}")

with open(os.path.join(OUTPUT_DIR, "optuna_best_params.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_trial":  study.best_trial.number,
            "best_value":  study.best_value,
            "best_params": study.best_params,
            "all_trials": [
                {
                    "number":     t.number,
                    "value":      t.value,
                    "state":      str(t.state),
                    "params":     t.params,
                    "user_attrs": t.user_attrs,
                }
                for t in study.trials
            ],
        },
        f, indent=2,
    )

# =============================================================================
# CELL 8 — Final training with best hyperparameters
# =============================================================================

# %%
tf.keras.backend.clear_session()
set_seed(SEED)

best = study.best_params
model_params = {k: best[k] for k in [
    "filters", "conv_blocks", "kernel_size", "lstm_units",
    "attention_heads", "attention_key_dim", "dense_units", "dropout", "weight_decay",
]}

final_model = build_model(**model_params)
compile_model(final_model, lr=best["lr"], gamma=best["focal_gamma"],
              label_smoothing=best["label_smoothing"])
final_model.summary()

with open(os.path.join(OUTPUT_DIR, "model_architecture.json"), "w", encoding="utf-8") as f:
    f.write(final_model.to_json())

best_weights_path = os.path.join(OUTPUT_DIR, "best_rescnn_bilstm_attention_q3.weights.h5")
final_f1_callback = ValMacroF1Checkpoint(X_val, y_val, mask_val,
                                         filepath=best_weights_path, batch_size=32)
final_callbacks = [
    final_f1_callback,
    EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=False),
    ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
]

history = final_model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    validation_data=(X_val, y_val, sw_val),
    epochs=FINAL_EPOCHS,
    batch_size=best["batch_size"],
    callbacks=final_callbacks,
    verbose=1,
)

final_model.load_weights(best_weights_path)
print("Loaded best final weights — val macro-F1:", final_f1_callback.best)

with open(os.path.join(OUTPUT_DIR, "final_history.json"), "w", encoding="utf-8") as f:
    json.dump(
        {key: [float(v) for v in values] for key, values in history.history.items()},
        f, indent=2,
    )

# =============================================================================
# CELL 9 — Test evaluation on CB513
# =============================================================================

# %%
test_loss, test_acc = final_model.evaluate(
    X_test, y_test, sample_weight=sw_test, batch_size=32, verbose=1,
)

y_pred       = final_model.predict(X_test, batch_size=32, verbose=0)
valid        = mask_test.astype(bool).reshape(-1)
y_true_flat  = y_test.reshape(-1, N_CLASSES)[valid]
y_pred_flat  = y_pred.reshape(-1, N_CLASSES)[valid]
y_true_label = np.argmax(y_true_flat, axis=1)
y_pred_label = np.argmax(y_pred_flat, axis=1)

metrics = {
    "keras_masked_accuracy": float(test_acc),
    "sklearn_masked_accuracy": float(accuracy_score(y_true_label, y_pred_label)),
    "balanced_accuracy":       float(balanced_accuracy_score(y_true_label, y_pred_label)),
    "precision_macro":         float(precision_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "recall_macro":            float(recall_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "f1_macro":                float(f1_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "auc_ovr":                 float(roc_auc_score(y_true_flat, y_pred_flat, multi_class="ovr")),
    "test_loss":               float(test_loss),
    "best_val_macro_f1":       float(final_f1_callback.best),
    "best_params":             best,
}

print("\n=== CB513 TEST METRICS ===")
for key, value in metrics.items():
    if key != "best_params":
        print(f"{key:28s}: {value:.4f}")

print("\nClassification report (Helix=0, Sheet=1, Coil=2):")
report_text = classification_report(y_true_label, y_pred_label, digits=4, zero_division=0)
report_dict = classification_report(y_true_label, y_pred_label, digits=4, zero_division=0, output_dict=True)
print(report_text)

print("Confusion matrix:")
cm = confusion_matrix(y_true_label, y_pred_label)
print(cm)

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)
with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2)

np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
np.save(os.path.join(OUTPUT_DIR, "y_true_labels.npy"),       y_true_label)
np.save(os.path.join(OUTPUT_DIR, "y_pred_labels.npy"),       y_pred_label)
np.save(os.path.join(OUTPUT_DIR, "y_pred_probabilities.npy"), y_pred_flat)

final_model.save(os.path.join(OUTPUT_DIR, "final_rescnn_bilstm_attention_q3.keras"))

# =============================================================================
# CELL 10 — Training curves
# =============================================================================

# %%
hist = history.history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.get("loss", []),     label="train")
plt.plot(hist.get("val_loss", []), label="val")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.get("accuracy", []),     label="train acc")
plt.plot(hist.get("val_accuracy", []), label="val acc")
if "val_macro_f1" in hist:
    plt.plot(hist["val_macro_f1"], label="val_macro_f1")
plt.title("Accuracy / Macro-F1")
plt.legend()

plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=160)
plt.show()

print("\nAll artifacts saved to:", OUTPUT_DIR)
