from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

_cudnn_path = "/venv/main/lib/python3.10/site-packages/nvidia/cudnn/lib"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _cudnn_path not in _ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_path}:/usr/lib/x86_64-linux-gnu:{_ld}"

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from optuna.distributions import (
    CategoricalDistribution, FloatDistribution, IntDistribution,
)
from optuna.trial import create_trial, TrialState
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add, Bidirectional, Conv1D, Dense, Dropout, LayerNormalization,
    LSTM, MultiHeadAttention, SeparableConv1D, SpatialDropout1D, TimeDistributed,
)
from tensorflow.keras.regularizers import l2

SEED = 42
MAX_LEN = 700
N_CLASSES = 3
VAL_FRACTION = 0.20
TRIALS_REMAINING = 2
SEARCH_EPOCHS = 8
FINAL_EPOCHS = 40

BASE_DIR = str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data")
(BASE_DIR if Path(BASE_DIR).exists() else Path("/workspace/data")).mkdir(parents=True, exist_ok=True)
if not Path(BASE_DIR).exists():
    BASE_DIR = "/workspace/data"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_artifacts")
RUN_NAME = f"rescnn_bilstm_attention_q3_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_NAME)
TRAIN_FILE = "cullpdb+profile_5926_filtered.npy.gz"
TEST_FILE = "cb513+profile_split1.npy.gz"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print("Output:", OUTPUT_DIR)
print("GPU:", tf.config.list_physical_devices("GPU"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


set_seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
def load_raw_dataset(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data.reshape(data.shape[0], MAX_LEN, 57)

def extract_q3(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = dataset[:, :, 21:24]
    x = np.concatenate([dataset[:, :, :21], dataset[:, :, 24:]], axis=2)
    mask = np.sum(x, axis=2) != 0
    return x.astype(np.float32), y.astype(np.float32), mask

cullpdb = load_raw_dataset(os.path.join(BASE_DIR, TRAIN_FILE))
cb513 = load_raw_dataset(os.path.join(BASE_DIR, TEST_FILE))
rng = np.random.default_rng(SEED)
indices = rng.permutation(cullpdb.shape[0])
split = int((1.0 - VAL_FRACTION) * cullpdb.shape[0])
X_train, y_train, mask_train = extract_q3(cullpdb[indices[:split]])
X_val, y_val, mask_val = extract_q3(cullpdb[indices[split:]])
X_test, y_test, mask_test = extract_q3(cb513)

y_train_labels = np.argmax(y_train, axis=-1)
flat_train_labels = y_train_labels[mask_train]
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.arange(N_CLASSES), y=flat_train_labels,
).astype(np.float32)

def make_sample_weight(mask, y_one_hot, weights):
    labels = np.argmax(y_one_hot, axis=-1)
    return mask.astype(np.float32) * weights[labels]

sw_train = make_sample_weight(mask_train, y_train, class_weights)
sw_val = make_sample_weight(mask_val, y_val, class_weights)
sw_test = mask_test.astype(np.float32)

print(f"Data: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

# ── Metrics / model / loss ───────────────────────────────────────────────────
def macro_f1_masked(y_true, y_pred_prob, mask):
    valid = mask.astype(bool).reshape(-1)
    true_flat = y_true.reshape(-1, N_CLASSES)[valid]
    pred_flat = y_pred_prob.reshape(-1, N_CLASSES)[valid]
    return float(f1_score(np.argmax(true_flat, 1), np.argmax(pred_flat, 1),
                          average="macro", zero_division=0))

class ValMacroF1Checkpoint(Callback):
    def __init__(self, x_val, y_val_data, mask_val_data, filepath=None, batch_size=32):
        super().__init__()
        self.x_val, self.y_val_data, self.mask_val_data = x_val, y_val_data, mask_val_data
        self.filepath = filepath
        self.batch_size = batch_size
        self.best = -1.0
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        score = macro_f1_masked(self.y_val_data, pred, self.mask_val_data)
        logs["val_macro_f1"] = score
        print(f" val_macro_f1={score:.4f}")
        if score > self.best:
            self.best = score
            if self.filepath:
                self.model.save_weights(self.filepath)
                print(f" saved best -> {self.filepath} ({score:.4f})")

def categorical_focal_loss(gamma=1.5, label_smoothing=0.0):
    def loss(y_true, y_pred):
        if label_smoothing > 0:
            y_true_smooth = y_true * (1.0 - label_smoothing) + label_smoothing / N_CLASSES
        else:
            y_true_smooth = y_true
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred_clipped), axis=-1)
        pt = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)
        return tf.pow(1.0 - pt, gamma) * ce
    return loss

def residual_conv_block(x, filters, kernel_size, dilation_rate, dropout, weight_decay):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same", kernel_regularizer=l2(weight_decay))(shortcut)
    x = SeparableConv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
                        activation="relu", depthwise_regularizer=l2(weight_decay),
                        pointwise_regularizer=l2(weight_decay))(x)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(dropout)(x)
    x = SeparableConv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate,
                        activation="relu", depthwise_regularizer=l2(weight_decay),
                        pointwise_regularizer=l2(weight_decay))(x)
    x = Add()([shortcut, x])
    return LayerNormalization()(x)

def build_model(filters=128, conv_blocks=3, kernel_size=7, lstm_units=96,
                attention_heads=4, attention_key_dim=32, dense_units=96,
                dropout=0.30, weight_decay=1e-5):
    inp = Input(shape=(MAX_LEN, 54), name="features")
    x = Conv1D(filters, 1, padding="same", activation="relu",
               kernel_regularizer=l2(weight_decay))(inp)
    x = LayerNormalization()(x)
    dilation_cycle = [1, 2, 4, 8]
    for block_idx in range(conv_blocks):
        x = residual_conv_block(x, filters, kernel_size,
                                dilation_cycle[block_idx % 4], dropout, weight_decay)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=0.0,
                           kernel_regularizer=l2(weight_decay)))(x)
    x = LayerNormalization()(x)
    attention = MultiHeadAttention(num_heads=attention_heads,
                                   key_dim=attention_key_dim,
                                   dropout=dropout)(x, x)
    x = Add()([x, attention])
    x = LayerNormalization()(x)
    x = TimeDistributed(Dense(dense_units, activation="relu",
                              kernel_regularizer=l2(weight_decay)))(x)
    x = Dropout(dropout)(x)
    out = TimeDistributed(Dense(N_CLASSES, activation="softmax"), name="q3")(x)
    return Model(inp, out)

def compile_model(model, lr, gamma, label_smoothing):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss=categorical_focal_loss(gamma=gamma, label_smoothing=label_smoothing),
                  metrics=["accuracy"])

# ── Reconstruct study from previous run ──────────────────────────────────────
completed = [
    {"n":0,"v":0.39235064332814257,"p":{"filters":96,"conv_blocks":2,"kernel_size":5,"lstm_units":64,"attention_heads":4,"attention_key_dim":32,"dense_units":128,"dropout":0.42500129688358124,"weight_decay":4.463511431418793e-06,"lr":0.00024429808699515945,"focal_gamma":2.0640844153205906,"label_smoothing":0.010071340123843066,"batch_size":24}},
    {"n":1,"v":0.39056600650928036,"p":{"filters":160,"conv_blocks":2,"kernel_size":7,"lstm_units":128,"attention_heads":4,"attention_key_dim":24,"dense_units":64,"dropout":0.23293137834806305,"weight_decay":1.2246566459399416e-05,"lr":0.00016068353479048014,"focal_gamma":2.4632176428467845,"label_smoothing":0.026222016621662375,"batch_size":24}},
    {"n":2,"v":0.4226401980103853,"p":{"filters":128,"conv_blocks":3,"kernel_size":9,"lstm_units":128,"attention_heads":4,"attention_key_dim":24,"dense_units":64,"dropout":0.4292925873902135,"weight_decay":6.520634986276547e-06,"lr":0.0005839673571413508,"focal_gamma":1.5564884973478748,"label_smoothing":0.039072368316758595,"batch_size":8}},
    {"n":3,"v":0.3987939071227418,"p":{"filters":96,"conv_blocks":3,"kernel_size":7,"lstm_units":64,"attention_heads":4,"attention_key_dim":48,"dense_units":128,"dropout":0.44068854980132366,"weight_decay":4.0472710729335695e-06,"lr":0.00016119154744445005,"focal_gamma":2.4683472922364773,"label_smoothing":0.0025277907787420294,"batch_size":16}},
    {"n":4,"v":0.416807666634965,"p":{"filters":96,"conv_blocks":4,"kernel_size":9,"lstm_units":96,"attention_heads":4,"attention_key_dim":48,"dense_units":96,"dropout":0.32056788747890763,"weight_decay":1.911029778908754e-05,"lr":0.0003014218872197445,"focal_gamma":1.524545539178253,"label_smoothing":0.05015238658039042,"batch_size":16}},
    {"n":5,"v":0.4553463650939383,"p":{"filters":160,"conv_blocks":4,"kernel_size":5,"lstm_units":96,"attention_heads":4,"attention_key_dim":48,"dense_units":128,"dropout":0.18765573093945462,"weight_decay":1.594458073661454e-06,"lr":0.0004382415197347539,"focal_gamma":0.8397567034440895,"label_smoothing":0.0542868251118207,"batch_size":16}},
    {"n":6,"v":0.4699770100311988,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":4,"attention_key_dim":24,"dense_units":96,"dropout":0.314790294505932,"weight_decay":1.3530272242687265e-05,"lr":0.0006548609587643501,"focal_gamma":1.5729693732786927,"label_smoothing":0.02591347822865286,"batch_size":8}},
    {"n":7,"v":0.3961483296264325,"p":{"filters":160,"conv_blocks":3,"kernel_size":9,"lstm_units":128,"attention_heads":4,"attention_key_dim":32,"dense_units":128,"dropout":0.20572376262718164,"weight_decay":2.1223440076971256e-05,"lr":0.00014021565300956803,"focal_gamma":1.7366860877288137,"label_smoothing":0.02187662728854415,"batch_size":24}},
    {"n":8,"v":0.44204533373846616,"p":{"filters":96,"conv_blocks":3,"kernel_size":7,"lstm_units":96,"attention_heads":4,"attention_key_dim":32,"dense_units":96,"dropout":0.27022095009984753,"weight_decay":1.925684865268777e-06,"lr":0.001419617140545497,"focal_gamma":1.3197344594408522,"label_smoothing":0.03693350897345612,"batch_size":24}},
    {"n":9,"v":0.4615634545387964,"p":{"filters":96,"conv_blocks":3,"kernel_size":7,"lstm_units":128,"attention_heads":2,"attention_key_dim":24,"dense_units":128,"dropout":0.16870970092557652,"weight_decay":3.038876581861608e-06,"lr":0.0019020128110586942,"focal_gamma":0.9854047387403762,"label_smoothing":0.030260317821766382,"batch_size":8}},
    {"n":10,"v":0.44623294472939223,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.34714933889767907,"weight_decay":4.018182039500303e-05,"lr":0.0008408349693418696,"focal_gamma":1.9988506930031638,"label_smoothing":0.010291161093327015,"batch_size":8}},
    {"n":11,"v":0.4646542571644447,"p":{"filters":128,"conv_blocks":4,"kernel_size":7,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.3500264396804333,"weight_decay":2.965221680155223e-06,"lr":0.0019810980069995317,"focal_gamma":0.8341270960100763,"label_smoothing":0.021307708404248057,"batch_size":8}},
    {"n":12,"v":0.4399232427606507,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.37336659644535075,"weight_decay":9.852564348804659e-06,"lr":0.0012915363461119707,"focal_gamma":1.1754272574248672,"label_smoothing":0.017994222448687786,"batch_size":8}},
    {"n":13,"v":0.4709626676951785,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.27222123823040284,"weight_decay":2.541406040821239e-06,"lr":0.0008357617771722925,"focal_gamma":1.1872640828858707,"label_smoothing":0.03660491546633087,"batch_size":8}},
    {"n":14,"v":0.4682223166120964,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.27601581340237913,"weight_decay":1.1871520286118914e-06,"lr":0.0007884831473909963,"focal_gamma":1.241158538337709,"label_smoothing":0.04391381681992125,"batch_size":8}},
    {"n":15,"v":0.5039988462813795,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.254213135775041,"weight_decay":4.7312450386894165e-05,"lr":0.0008549138916301651,"focal_gamma":1.726190894118447,"label_smoothing":0.03484141203444936,"batch_size":8}},
    {"n":16,"v":0.494291898707119,"p":{"filters":128,"conv_blocks":4,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":96,"dropout":0.24018642178539354,"weight_decay":4.422216515723788e-05,"lr":0.001099996505921107,"focal_gamma":1.8365791485109115,"label_smoothing":0.05949666006353353,"batch_size":8}},
    {"n":17,"v":0.4698675363611789,"p":{"filters":128,"conv_blocks":3,"kernel_size":5,"lstm_units":64,"attention_heads":2,"attention_key_dim":24,"dense_units":64,"dropout":0.23193557725893987,"weight_decay":4.556056033606475e-05,"lr":0.0011543290932506955,"focal_gamma":1.8669416070039704,"label_smoothing":0.059347193452913054,"batch_size":8}},
]

distributions = {
    "filters":           CategoricalDistribution([96, 128, 160]),
    "conv_blocks":       IntDistribution(2, 4),
    "kernel_size":       CategoricalDistribution([5, 7, 9]),
    "lstm_units":        CategoricalDistribution([64, 96, 128]),
    "attention_heads":   CategoricalDistribution([2, 4]),
    "attention_key_dim": CategoricalDistribution([24, 32, 48]),
    "dense_units":       CategoricalDistribution([64, 96, 128]),
    "dropout":           FloatDistribution(0.15, 0.45),
    "weight_decay":      FloatDistribution(1e-6, 5e-5, log=True),
    "lr":                FloatDistribution(1e-4, 2e-3, log=True),
    "focal_gamma":       FloatDistribution(0.8, 2.5),
    "label_smoothing":   FloatDistribution(0.0, 0.06),
    "batch_size":        CategoricalDistribution([8, 16, 24]),
}

study = optuna.create_study(direction="maximize",
                            study_name="q3_rescnn_bilstm_attention_macro_f1")
for t in completed:
    study.add_trial(create_trial(
        params=t["p"],
        values=[t["v"]],
        distributions=distributions,
        state=TrialState.COMPLETE,
        system_attrs={"_number": t["n"]},
    ))

# Fix trial numbers
for i, t in enumerate(study.trials):
    t._trial_id = i
    t.number = completed[i]["n"]

print(f"Reconstructed study with {len(completed)} trials")
print(f"Best so far: Trial {study.best_trial.number}, value={study.best_value:.4f}")
print(f"Best params: {study.best_params}")

run_config = {
    "base_dir": BASE_DIR, "output_dir": OUTPUT_DIR,
    "train_file": TRAIN_FILE, "test_file": TEST_FILE,
    "seed": SEED, "max_len": MAX_LEN, "n_classes": N_CLASSES,
    "trials_remaining": TRIALS_REMAINING, "search_epochs": SEARCH_EPOCHS,
    "final_epochs": FINAL_EPOCHS, "val_fraction": VAL_FRACTION,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "continued_from": "previous run (trials 0-17 completed)",
    "best_so_far": {"trial": study.best_trial.number, "value": study.best_value, "params": study.best_params},
}
with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# ── Continue trial 18, 19 ────────────────────────────────────────────────────
def objective(trial):
    tf.keras.backend.clear_session()
    set_seed(SEED + trial.number)

    params = {
        "filters":           trial.suggest_categorical("filters",           [96, 128, 160]),
        "conv_blocks":       trial.suggest_int("conv_blocks", 2, 4),
        "kernel_size":       trial.suggest_categorical("kernel_size",       [5, 7, 9]),
        "lstm_units":        trial.suggest_categorical("lstm_units",        [64, 96, 128]),
        "attention_heads":   trial.suggest_categorical("attention_heads",   [2, 4]),
        "attention_key_dim": trial.suggest_categorical("attention_key_dim", [24, 32, 48]),
        "dense_units":       trial.suggest_categorical("dense_units",       [64, 96, 128]),
        "dropout":           trial.suggest_float("dropout", 0.15, 0.45),
        "weight_decay":      trial.suggest_float("weight_decay", 1e-6, 5e-5, log=True),
    }
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    gamma = trial.suggest_float("focal_gamma", 0.8, 2.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.06)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24])

    model = build_model(**params)
    compile_model(model, lr=lr, gamma=gamma, label_smoothing=label_smoothing)
    f1_cb = ValMacroF1Checkpoint(X_val, y_val, mask_val, filepath=None, batch_size=32)
    callbacks = [
        f1_cb,
        EarlyStopping(monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    model.fit(X_train, y_train, sample_weight=sw_train,
              validation_data=(X_val, y_val, sw_val),
              epochs=SEARCH_EPOCHS, batch_size=batch_size, callbacks=callbacks, verbose=1)
    trial.set_user_attr("best_val_macro_f1", f1_cb.best)
    trial.set_user_attr("params_full", {**params, "lr": lr, "focal_gamma": gamma,
                                        "label_smoothing": label_smoothing, "batch_size": batch_size})
    return f1_cb.best

study.optimize(objective, n_trials=TRIALS_REMAINING, gc_after_trial=True)

print(f"\n=== FINAL RESULTS ===")
print(f"Best trial: {study.best_trial.number}")
print(f"Best val macro-F1: {study.best_value}")
print(f"Best params: {study.best_params}")

# ── Save study results ───────────────────────────────────────────────────────
try:
    study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)
except Exception as e:
    print(f"Could not save CSV: {e}")

with open(os.path.join(OUTPUT_DIR, "optuna_best_params.json"), "w") as f:
    json.dump({
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [{"number": t.number, "value": t.value, "state": str(t.state),
                         "params": t.params, "user_attrs": t.user_attrs}
                       for t in study.trials],
    }, f, indent=2)

# ── Final training ───────────────────────────────────────────────────────────
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

with open(os.path.join(OUTPUT_DIR, "model_architecture.json"), "w") as f:
    f.write(final_model.to_json())

best_weights_path = os.path.join(OUTPUT_DIR, "best_rescnn_bilstm_attention_q3.weights.h5")
final_f1_cb = ValMacroF1Checkpoint(X_val, y_val, mask_val, filepath=best_weights_path, batch_size=32)
final_callbacks = [
    final_f1_cb,
    EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=False),
    ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

history = final_model.fit(
    X_train, y_train, sample_weight=sw_train,
    validation_data=(X_val, y_val, sw_val),
    epochs=FINAL_EPOCHS, batch_size=best["batch_size"],
    callbacks=final_callbacks, verbose=1,
)

final_model.load_weights(best_weights_path)
print("Loaded best weights — val macro-F1:", final_f1_cb.best)

with open(os.path.join(OUTPUT_DIR, "final_history.json"), "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

# ── Test evaluation ──────────────────────────────────────────────────────────
test_loss, test_acc = final_model.evaluate(X_test, y_test, sample_weight=sw_test, batch_size=32, verbose=1)
y_pred = final_model.predict(X_test, batch_size=32, verbose=0)
valid = mask_test.astype(bool).reshape(-1)
y_true_flat = y_test.reshape(-1, N_CLASSES)[valid]
y_pred_flat = y_pred.reshape(-1, N_CLASSES)[valid]
y_true_label = np.argmax(y_true_flat, axis=1)
y_pred_label = np.argmax(y_pred_flat, axis=1)

metrics = {
    "keras_masked_accuracy": float(test_acc),
    "sklearn_masked_accuracy": float(accuracy_score(y_true_label, y_pred_label)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true_label, y_pred_label)),
    "precision_macro": float(precision_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_true_label, y_pred_label, average="macro", zero_division=0)),
    "auc_ovr": float(roc_auc_score(y_true_flat, y_pred_flat, multi_class="ovr")),
    "test_loss": float(test_loss),
    "best_val_macro_f1": float(final_f1_cb.best),
    "best_params": best,
}

print("\n=== CB513 TEST METRICS ===")
for k, v in metrics.items():
    if k != "best_params":
        print(f"{k:28s}: {v:.4f}")

print("\nClassification report:")
report_text = classification_report(y_true_label, y_pred_label, digits=4, zero_division=0)
print(report_text)
cm = confusion_matrix(y_true_label, y_pred_label)
print("Confusion matrix:")
print(cm)

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)
with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(classification_report(y_true_label, y_pred_label, digits=4, zero_division=0, output_dict=True), f, indent=2)
np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
np.save(os.path.join(OUTPUT_DIR, "y_true_labels.npy"), y_true_label)
np.save(os.path.join(OUTPUT_DIR, "y_pred_labels.npy"), y_pred_label)
np.save(os.path.join(OUTPUT_DIR, "y_pred_probabilities.npy"), y_pred_flat)
final_model.save(os.path.join(OUTPUT_DIR, "final_rescnn_bilstm_attention_q3.keras"))

# ── Training curves ──────────────────────────────────────────────────────────
hist = history.history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.get("loss", []), label="train")
plt.plot(hist.get("val_loss", []), label="val")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist.get("accuracy", []), label="train acc")
plt.plot(hist.get("val_accuracy", []), label="val acc")
if "val_macro_f1" in hist:
    plt.plot(hist["val_macro_f1"], label="val_macro_f1")
plt.title("Accuracy / Macro-F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=160)
plt.show()
print("\nAll artifacts saved to:", OUTPUT_DIR)
