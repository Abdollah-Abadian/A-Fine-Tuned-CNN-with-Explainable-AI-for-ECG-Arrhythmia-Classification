"""
ECG preprocessing pipeline: filtering, R-peak-centered segmentation, AAMI
mapping, RR-feature extraction, patient-wise split, and augmentation.

Running this module end-to-end (`python -m src.preprocessing --raw data/raw
--out data/processed`) reproduces the pipeline of §3.1 of the paper and
writes train/val/test .npz files consumed by `src/train.py`.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from src.aami_mapping import AAMI_CLASS_TO_INDEX, is_beat_symbol, symbol_to_aami
from src.augmentation import augment_training_set
from src.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    DE_CHAZAL_TEST_RECORDS,
    DE_CHAZAL_TRAIN_RECORDS,
    POST_R_SAMPLES,
    PRE_R_SAMPLES,
    RANDOM_SEED,
    SAMPLING_RATE_HZ,
    VAL_FRACTION_OF_TRAIN_PATIENTS,
    WINDOW_SAMPLES,
)
from src.rr_features import compute_dynamic_rr_features, zscore_apply, zscore_fit


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def bandpass_filter(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> np.ndarray:
    """4th-order Butterworth band-pass, 0.5-45 Hz, zero-phase (filtfilt)."""
    nyq = 0.5 * fs
    low = BANDPASS_LOW_HZ / nyq
    high = BANDPASS_HIGH_HZ / nyq
    b, a = butter(BANDPASS_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def zscore_channel(signal: np.ndarray) -> np.ndarray:
    mu = signal.mean()
    sigma = signal.std()
    sigma = sigma if sigma > 1e-8 else 1e-8
    return (signal - mu) / sigma


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_record(
    channel_signal: np.ndarray,
    r_peaks: np.ndarray,
    symbols: List[str],
    pre: int = PRE_R_SAMPLES,
    post: int = POST_R_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fixed-length R-peak-centered windows from a single filtered,
    normalized channel signal.

    Returns
    -------
    beats     : (n_valid, pre+post) windows
    aami_idx  : (n_valid,) integer AAMI class index for each retained beat
    kept_rpos : (n_valid,) R-peak sample positions retained (needed for RR features)
    """
    beats, labels, kept_pos = [], [], []
    n = len(channel_signal)
    for pos, sym in zip(r_peaks, symbols):
        if not is_beat_symbol(sym):
            continue
        aami = symbol_to_aami(sym)
        if aami is None:
            continue
        start, end = pos - pre, pos + post
        if start < 0 or end > n:
            continue  # drop beats too close to record boundaries
        beats.append(channel_signal[start:end])
        labels.append(AAMI_CLASS_TO_INDEX[aami])
        kept_pos.append(pos)
    if not beats:
        return (
            np.empty((0, pre + post)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )
    return np.stack(beats), np.array(labels, dtype=int), np.array(kept_pos, dtype=int)


# --------------------------------------------------------------------------- #
# Full record processing (requires wfdb + raw files on disk)
# --------------------------------------------------------------------------- #
def process_record(record_path: str) -> Dict[str, np.ndarray]:
    """Load a single MIT-BIH record via wfdb, filter, segment, and compute
    RR features. `record_path` excludes the file extension, e.g. 'data/raw/101'.
    """
    import wfdb  # local import: only required when touching real raw data

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    channel0 = record.p_signal[:, 0].astype(np.float64)
    filtered = bandpass_filter(channel0, fs=record.fs)
    normalized = zscore_channel(filtered)

    beats, labels, r_pos = segment_record(normalized, annotation.sample, annotation.symbol)
    if len(beats) == 0:
        return dict(beats=beats, labels=labels, rr=np.empty((0, 4)))

    rr_all = compute_dynamic_rr_features(r_pos, fs=record.fs)
    return dict(beats=beats, labels=labels, rr=rr_all, r_positions=r_pos)


def process_dataset(raw_dir: str, record_ids: List[str]) -> Dict[int, Dict[str, np.ndarray]]:
    per_record = {}
    for rid in record_ids:
        path = os.path.join(raw_dir, rid)
        if not os.path.exists(path + ".dat"):
            raise FileNotFoundError(
                f"Missing {path}.dat — run src/download_mitbih.py first."
            )
        per_record[rid] = process_record(path)
    return per_record


# --------------------------------------------------------------------------- #
# Pipeline driver
# --------------------------------------------------------------------------- #
def build_split(
    raw_dir: str, record_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate beats/labels/RR-features/record-ids across a list of records."""
    all_beats, all_labels, all_rr, all_rids = [], [], [], []
    per_record = process_dataset(raw_dir, record_ids)
    for rid, d in per_record.items():
        if len(d["beats"]) == 0:
            continue
        all_beats.append(d["beats"])
        all_labels.append(d["labels"])
        all_rr.append(d["rr"])
        all_rids.append(np.array([rid] * len(d["labels"])))
    return (
        np.concatenate(all_beats),
        np.concatenate(all_labels),
        np.concatenate(all_rr),
        np.concatenate(all_rids),
    )


def run_pipeline(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    train_records_all = [str(r) for r in DE_CHAZAL_TRAIN_RECORDS]
    test_records = [str(r) for r in DE_CHAZAL_TEST_RECORDS]

    # Hold out 10% of training-partition patients for validation
    n_val = max(1, int(round(VAL_FRACTION_OF_TRAIN_PATIENTS * len(train_records_all))))
    shuffled = rng.permutation(train_records_all)
    val_records = list(shuffled[:n_val])
    train_records = list(shuffled[n_val:])

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Val records   ({len(val_records)}): {val_records}")
    print(f"Test records  ({len(test_records)}): {test_records}")

    X_train, y_train, rr_train, rid_train = build_split(raw_dir, train_records)
    X_val, y_val, rr_val, rid_val = build_split(raw_dir, val_records)
    X_test, y_test, rr_test, rid_test = build_split(raw_dir, test_records)

    # Fit RR Z-score scaler on training partition only, apply everywhere
    mu, sigma = zscore_fit(rr_train)
    rr_train = zscore_apply(rr_train, mu, sigma)
    rr_val = zscore_apply(rr_val, mu, sigma)
    rr_test = zscore_apply(rr_test, mu, sigma)

    # Augment training partition only (§3.1)
    X_train_aug, y_train_aug, rr_train_aug = augment_training_set(
        X_train, y_train, rr_train, rng=rng
    )

    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        X=X_train_aug[..., None], rr=rr_train_aug, y=y_train_aug,
    )
    np.savez_compressed(
        os.path.join(out_dir, "val.npz"),
        X=X_val[..., None], rr=rr_val, y=y_val, record_id=rid_val,
    )
    np.savez_compressed(
        os.path.join(out_dir, "test.npz"),
        X=X_test[..., None], rr=rr_test, y=y_test, record_id=rid_test,
    )

    manifest = {
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "rr_scaler_mean": mu.tolist(),
        "rr_scaler_std": sigma.tolist(),
        "n_train_raw": int(len(y_train)),
        "n_train_augmented": int(len(y_train_aug)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "window_samples": WINDOW_SAMPLES,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Preprocessing complete. Manifest:")
    print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# Synthetic illustrative sample (for offline smoke tests / docs, NOT for
# reporting results) — see data/README.md §6
# --------------------------------------------------------------------------- #
def _parametric_beat_template(aami_class: str, t: np.ndarray, rng) -> np.ndarray:
    """Generate a parametric Gaussian P-QRS-T template with class-specific
    morphology perturbations, purely for illustrative sample data."""
    def gauss(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    p = gauss(-0.30, 0.045, 0.15)
    q = gauss(-0.02, 0.008, -0.15)
    r = gauss(0.00, 0.010, 1.00)
    s = gauss(0.02, 0.010, -0.25)
    t_wave = gauss(0.24, 0.08, 0.30)

    if aami_class == "N":
        beat = p + q + r + s + t_wave
    elif aami_class == "S":
        # supraventricular: altered/absent P-wave morphology, near-normal QRS
        p = gauss(-0.22, 0.06, 0.08 + 0.05 * rng.random())
        beat = p + q + r + s + t_wave
    elif aami_class == "V":
        # ventricular: wide, bizarre QRS, no preceding P-wave, discordant T
        r = gauss(0.00, 0.028, 1.2)
        s = gauss(0.05, 0.022, -0.5)
        t_wave = gauss(0.28, 0.09, -0.35)
        beat = q * 0 + r + s + t_wave
    elif aami_class == "F":
        # fusion of normal and ventricular: hybrid morphology
        r_n = gauss(0.00, 0.010, 0.6)
        r_v = gauss(0.01, 0.02, 0.6)
        beat = p * 0.6 + q + r_n + r_v + s + t_wave
    else:  # Q: paced / unclassifiable
        spike = gauss(-0.05, 0.004, 1.4)
        r = gauss(0.00, 0.014, 0.7)
        beat = spike + r + t_wave * 0.5

    noise = rng.normal(0, 0.02, size=t.shape)
    return beat + noise


def synthesize_sample_csv(out_path: str, per_class: int = 40, seed: int = RANDOM_SEED) -> None:
    """Generate `data/sample/sample_beats.csv` — a small, clearly-synthetic,
    illustrative dataset with the exact same column schema as the real
    preprocessed data, used only for repository smoke tests / docs."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    t = np.linspace(-93 / SAMPLING_RATE_HZ, 94 / SAMPLING_RATE_HZ - 1 / SAMPLING_RATE_HZ, WINDOW_SAMPLES)

    rows = []
    beat_counter = 0
    for cls in ["N", "S", "V", "F", "Q"]:
        for i in range(per_class):
            beat = _parametric_beat_template(cls, t, rng)
            beat = zscore_channel(beat)
            rr_prev = rng.normal(0, 1)
            rr_post = rng.normal(0, 1)
            rr_local = rng.normal(0, 1)
            rr_ratio = rng.normal(0, 1)
            row = {
                "record_id": f"SAMPLE_{beat_counter + 1:03d}",
                "beat_index": i,
                "aami_class": cls,
                "rr_prev": rr_prev,
                "rr_post": rr_post,
                "rr_local": rr_local,
                "rr_ratio": rr_ratio,
            }
            for j, v in enumerate(beat):
                row[f"x_{j:03d}"] = float(v)
            rows.append(row)
            beat_counter += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)}-row synthetic illustrative sample to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument(
        "--make-sample", action="store_true",
        help="Only (re)generate the synthetic data/sample/sample_beats.csv illustrative file",
    )
    args = parser.parse_args()

    if args.make_sample:
        synthesize_sample_csv("data/sample/sample_beats.csv")
    else:
        run_pipeline(args.raw, args.out)
