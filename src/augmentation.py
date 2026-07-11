"""
Class-imbalance-aware data augmentation (§3.1).

Applies amplitude scaling and temporal warping within physiologically
plausible bounds to minority-class training beats until each class reaches
`TARGET_SAMPLES_PER_CLASS`. Augmented candidates whose R-peak shifts beyond
+/-5 samples or whose QRS amplitude changes beyond +/-20% relative to the
source beat are rejected and re-sampled.
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

from src.config import (
    AMPLITUDE_SCALE_RANGE,
    MAX_QRS_AMPLITUDE_CHANGE,
    MAX_RPEAK_SHIFT_SAMPLES,
    PRE_R_SAMPLES,
    TARGET_SAMPLES_PER_CLASS,
    TEMPORAL_WARP_RANGE,
    WINDOW_SAMPLES,
)


def _qrs_window(beat: np.ndarray, half_width: int = 15) -> np.ndarray:
    """Approximate QRS window centered on the R-peak (index PRE_R_SAMPLES)."""
    lo = max(0, PRE_R_SAMPLES - half_width)
    hi = min(len(beat), PRE_R_SAMPLES + half_width)
    return beat[lo:hi]


def _amplitude_scale(beat: np.ndarray, factor: float) -> np.ndarray:
    return beat * factor


def _temporal_warp(beat: np.ndarray, alpha: float) -> np.ndarray:
    """Stretch/compress the time axis by `alpha` then resample back to the
    original WINDOW_SAMPLES length, keeping the R-peak sample fixed at index
    PRE_R_SAMPLES."""
    n = len(beat)
    original_idx = np.arange(n)
    # warp around the R-peak location so the peak itself stays anchored
    warped_idx = PRE_R_SAMPLES + (original_idx - PRE_R_SAMPLES) * alpha
    warped_idx = np.clip(warped_idx, 0, n - 1)
    interpolator = interp1d(original_idx, beat, kind="cubic", fill_value="extrapolate")
    return interpolator(warped_idx)


def _find_rpeak_shift(original: np.ndarray, augmented: np.ndarray, search_radius: int = 10) -> int:
    """Estimate the shift in R-peak location (argmax within a local window)
    introduced by augmentation, relative to the expected index PRE_R_SAMPLES."""
    lo = max(0, PRE_R_SAMPLES - search_radius)
    hi = min(len(augmented), PRE_R_SAMPLES + search_radius)
    local_peak = lo + int(np.argmax(np.abs(augmented[lo:hi])))
    return local_peak - PRE_R_SAMPLES


def generate_augmented_beat(
    beat: np.ndarray, rng: np.random.Generator, max_attempts: int = 20
) -> np.ndarray:
    """Produce one augmented beat satisfying the morphology-preservation
    constraints; retries with fresh random parameters if rejected."""
    original_qrs_amp = np.max(np.abs(_qrs_window(beat))) + 1e-8

    for _ in range(max_attempts):
        amp_factor = rng.uniform(*AMPLITUDE_SCALE_RANGE)
        warp_alpha = rng.uniform(*TEMPORAL_WARP_RANGE)

        candidate = _temporal_warp(beat, warp_alpha)
        candidate = _amplitude_scale(candidate, amp_factor)

        shift = _find_rpeak_shift(beat, candidate)
        new_qrs_amp = np.max(np.abs(_qrs_window(candidate))) + 1e-8
        amp_change = abs(new_qrs_amp - original_qrs_amp) / original_qrs_amp

        if abs(shift) <= MAX_RPEAK_SHIFT_SAMPLES and amp_change <= MAX_QRS_AMPLITUDE_CHANGE:
            return candidate.astype(np.float32)

    # Fall back to the least aggressive perturbation (near-identity) if every
    # attempt was rejected, guaranteeing the augmenter always terminates.
    fallback = _amplitude_scale(beat, 1.0 + rng.uniform(-0.02, 0.02))
    return fallback.astype(np.float32)


def augment_training_set(
    X: np.ndarray, y: np.ndarray, rr: np.ndarray, rng: np.random.Generator,
    target_per_class: int = TARGET_SAMPLES_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified resampling: for every AAMI class, oversample (via augmentation)
    up to `target_per_class`, or randomly subsample if a class already
    exceeds the target (this occurs for class N, which is downsampled from
    ~72,000 patient-wise training beats towards the ~10,000/class balance
    point reported in the paper, consistent with the reported final
    ~50,000-sample balanced training set).
    """
    classes = np.unique(y)
    X_out, y_out, rr_out = [], [], []

    for c in classes:
        idx = np.where(y == c)[0]
        n_have = len(idx)

        if n_have >= target_per_class:
            chosen = rng.choice(idx, size=target_per_class, replace=False)
            X_out.append(X[chosen])
            rr_out.append(rr[chosen])
            y_out.append(np.full(target_per_class, c))
            continue

        # Keep all originals, then augment until reaching the target
        X_out.append(X[idx])
        rr_out.append(rr[idx])
        y_out.append(np.full(n_have, c))

        n_needed = target_per_class - n_have
        source_idx = rng.choice(idx, size=n_needed, replace=True)
        synth_beats = np.empty((n_needed, X.shape[1]), dtype=np.float32)
        synth_rr = rr[source_idx].copy()
        # RR features are jittered mildly (+/-3%) to avoid duplicate temporal
        # features across augmented copies of the same source beat
        synth_rr *= (1 + rng.uniform(-0.03, 0.03, size=synth_rr.shape))

        for i, src in enumerate(source_idx):
            synth_beats[i] = generate_augmented_beat(X[src], rng)

        X_out.append(synth_beats)
        rr_out.append(synth_rr)
        y_out.append(np.full(n_needed, c))

    X_final = np.concatenate(X_out).astype(np.float32)
    rr_final = np.concatenate(rr_out).astype(np.float32)
    y_final = np.concatenate(y_out).astype(np.int64)

    # shuffle
    perm = rng.permutation(len(y_final))
    return X_final[perm], y_final[perm], rr_final[perm]


def verify_no_duplicates(X: np.ndarray, tol: float = 1e-9) -> bool:
    """Sanity check used post-augmentation: ensure no two rows are bit-identical."""
    flat = X.reshape(len(X), -1)
    _, unique_idx = np.unique(np.round(flat / tol), axis=0, return_index=True)
    return len(unique_idx) == len(X)
