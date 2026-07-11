"""
Dataset loading utilities: npz loaders for train/val/test partitions and a
Leave-One-Patient-Out (LOOPO) cross-validation split generator (Table 6).
"""

import os
from typing import Dict, Iterator, Tuple

import numpy as np

from src.config import DE_CHAZAL_TEST_RECORDS, DE_CHAZAL_TRAIN_RECORDS, NUM_CLASSES

ALL_47_RECORDS = sorted(set(DE_CHAZAL_TRAIN_RECORDS) | set(DE_CHAZAL_TEST_RECORDS) | {
    102, 104, 107, 217,  # remaining MIT-BIH records not used in the de Chazal split
})


def load_split(path: str) -> Dict[str, np.ndarray]:
    """Load a train/val/test .npz produced by src.preprocessing.run_pipeline."""
    data = np.load(path, allow_pickle=True)
    out = {"X": data["X"], "rr": data["rr"], "y": data["y"]}
    if "record_id" in data:
        out["record_id"] = data["record_id"]
    return out


def to_one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def loopo_splits(
    per_record_data: Dict[str, Dict[str, np.ndarray]]
) -> Iterator[Tuple[str, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    """
    Generator yielding (held_out_record_id, train_data, test_data) for
    Leave-One-Patient-Out cross-validation (§3.1, Table 6). `per_record_data`
    maps record_id -> {"X":..., "rr":..., "y":...} as produced by
    `src.preprocessing.process_dataset`.
    """
    record_ids = list(per_record_data.keys())
    for held_out in record_ids:
        train_ids = [r for r in record_ids if r != held_out]

        def _concat(ids, key):
            return np.concatenate([per_record_data[i][key] for i in ids])

        train_data = {
            "X": _concat(train_ids, "beats")[..., None],
            "rr": _concat(train_ids, "rr"),
            "y": _concat(train_ids, "labels"),
        }
        test_data = {
            "X": per_record_data[held_out]["beats"][..., None],
            "rr": per_record_data[held_out]["rr"],
            "y": per_record_data[held_out]["labels"],
        }
        yield held_out, train_data, test_data


def make_tf_dataset(X: np.ndarray, rr: np.ndarray, y: np.ndarray, batch_size: int,
                     shuffle: bool = True, seed: int = 42):
    """Wrap (morphology, RR, label) arrays into a tf.data.Dataset yielding
    ((X_morph, X_rr), y_onehot) batches for the dual-branch FT-CNN, or a plain
    (X_morph, y_onehot) dataset for single-branch benchmark models."""
    import tensorflow as tf

    y_oh = to_one_hot(y)
    ds = tf.data.Dataset.from_tensor_slices(((X, rr), y_oh))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_tf_dataset_single_input(X: np.ndarray, y: np.ndarray, batch_size: int,
                                  shuffle: bool = True, seed: int = 42):
    import tensorflow as tf

    y_oh = to_one_hot(y)
    ds = tf.data.Dataset.from_tensor_slices((X, y_oh))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
