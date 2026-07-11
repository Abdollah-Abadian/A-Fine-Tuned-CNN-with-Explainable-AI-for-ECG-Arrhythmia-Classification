"""
RR-interval dynamic feature extraction.

Implements Equations (1)-(5) of the paper:

    RR_prev  (t) = t(R_i)   - t(R_{i-1})
    RR_post  (t) = t(R_{i+1}) - t(R_i)
    RR_local     = (1/N) * sum_{j in window} RR_j
    RR_ratio     = RR_prev / RR_local
    x'           = (x - mu) / sigma            (Z-score standardization)
"""

from typing import Tuple

import numpy as np


def compute_rr_intervals(r_peak_samples: np.ndarray, fs: int) -> np.ndarray:
    """Convert consecutive R-peak sample indices into RR intervals (seconds).

    Parameters
    ----------
    r_peak_samples : 1D array of R-peak sample indices (sorted ascending)
    fs : sampling frequency in Hz

    Returns
    -------
    rr : array of length len(r_peak_samples)-1, RR_i = t(R_{i+1}) - t(R_i)
    """
    r_peak_samples = np.asarray(r_peak_samples, dtype=np.float64)
    return np.diff(r_peak_samples) / float(fs)


def compute_dynamic_rr_features(
    r_peak_samples: np.ndarray, fs: int, local_window: int = 10
) -> np.ndarray:
    """
    Compute (RR_prev, RR_post, RR_local, RR_ratio) for every beat in a record.

    For the first beat, RR_prev is undefined and is imputed with the record's
    median RR interval (a beat cannot have a "previous" interval); analogously
    RR_post is imputed for the last beat.

    Returns
    -------
    features : ndarray of shape (num_beats, 4), columns =
               [RR_prev, RR_post, RR_local, RR_ratio]
    """
    r_peak_samples = np.asarray(r_peak_samples, dtype=np.float64)
    n_beats = len(r_peak_samples)
    if n_beats < 2:
        raise ValueError("Need at least 2 R-peaks to compute RR features.")

    rr = compute_rr_intervals(r_peak_samples, fs)          # length n_beats-1
    median_rr = float(np.median(rr))

    rr_prev = np.empty(n_beats, dtype=np.float64)
    rr_post = np.empty(n_beats, dtype=np.float64)

    rr_prev[0] = median_rr
    rr_prev[1:] = rr
    rr_post[-1] = median_rr
    rr_post[:-1] = rr

    rr_local = np.empty(n_beats, dtype=np.float64)
    half = local_window // 2
    for i in range(n_beats):
        lo = max(0, i - half)
        hi = min(n_beats, i + half + 1)
        # local mean computed over RR_prev values in the neighborhood window,
        # matching Eq. (3): local mean RR interval over N neighboring beats
        rr_local[i] = np.mean(rr_prev[lo:hi])

    rr_local_safe = np.where(rr_local == 0, median_rr, rr_local)
    rr_ratio = rr_prev / rr_local_safe

    return np.stack([rr_prev, rr_post, rr_local, rr_ratio], axis=1)


def zscore_fit(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean/std on the *training* partition only (Eq. 5)."""
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    return mu, sigma


def zscore_apply(features: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (features - mu) / sigma
