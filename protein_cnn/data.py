from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


AMINO_ACID_SLICE = slice(0, 21)
LABEL_SLICE = slice(22, 30)
PROFILE_SLICE = slice(35, 56)
MASK_INDEX = 56
MAX_LEN = 700
NUM_CLASSES = 8


@dataclass
class ProteinArrays:
    features: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


def _load_raw(path: str | Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return array.reshape(-1, MAX_LEN, 57)


def load_protein_arrays(path: str | Path) -> ProteinArrays:
    raw = _load_raw(path)
    aa = np.asarray(raw[:, :, AMINO_ACID_SLICE], dtype=np.float32)
    profile = np.asarray(raw[:, :, PROFILE_SLICE], dtype=np.float32)
    features = np.concatenate([aa, profile], axis=-1)

    labels = np.asarray(np.argmax(raw[:, :, LABEL_SLICE], axis=-1), dtype=np.int64)
    mask = np.asarray(raw[:, :, MASK_INDEX] == 0, dtype=bool)
    return ProteinArrays(features=features, labels=labels, mask=mask)


def split_train_val(
    arrays: ProteinArrays,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[ProteinArrays, ProteinArrays]:
    n = arrays.features.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = int(round(n * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    def take(idx: np.ndarray) -> ProteinArrays:
        return ProteinArrays(
            features=arrays.features[idx],
            labels=arrays.labels[idx],
            mask=arrays.mask[idx],
        )

    return take(train_idx), take(val_idx)


def summarize_dataset(arrays: ProteinArrays) -> dict:
    valid_lengths = arrays.mask.sum(axis=1)
    label_counts = np.bincount(
        arrays.labels[arrays.mask].reshape(-1),
        minlength=NUM_CLASSES,
    )
    return {
        "num_proteins": int(arrays.features.shape[0]),
        "max_len": int(arrays.features.shape[1]),
        "num_features": int(arrays.features.shape[2]),
        "num_valid_residues": int(arrays.mask.sum()),
        "min_length": int(valid_lengths.min()),
        "max_length": int(valid_lengths.max()),
        "mean_length": float(valid_lengths.mean()),
        "median_length": float(np.median(valid_lengths)),
        "label_counts": label_counts.astype(int).tolist(),
    }
