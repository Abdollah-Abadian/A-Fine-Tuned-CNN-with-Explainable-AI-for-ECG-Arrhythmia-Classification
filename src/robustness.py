"""
Robustness evaluation: additive Gaussian noise, amplitude scaling, temporal shifting,
baseline wander, motion artifacts, powerline interference.
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from src.preprocessing import bandpass_filter
from src.config import SAMPLING_RATE_HZ


def add_gaussian_noise(signal, snr_db):
    """Add zero-mean Gaussian noise to achieve a given SNR (dB)."""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def amplitude_scale(signal, scale_factor):
    return signal * scale_factor


def temporal_shift(signal, shift_samples):
    if shift_samples == 0:
        return signal
    if shift_samples > 0:
        return np.concatenate([np.zeros(shift_samples), signal[:-shift_samples]])
    else:
        return np.concatenate([signal[-shift_samples:], np.zeros(-shift_samples)])


def add_baseline_wander(signal, freq_hz, amplitude_mv, fs=SAMPLING_RATE_HZ):
    t = np.arange(len(signal)) / fs
    wander = amplitude_mv * np.sin(2 * np.pi * freq_hz * t)
    return signal + wander


def add_motion_artifact(signal, freq_hz, amplitude_mv, duration_ms, fs=SAMPLING_RATE_HZ):
    duration_samples = int(duration_ms * fs / 1000)
    if duration_samples <= 0:
        return signal
    start = np.random.randint(0, len(signal) - duration_samples)
    t = np.arange(duration_samples) / fs
    artifact = amplitude_mv * np.sin(2 * np.pi * freq_hz * t)
    signal[start:start+duration_samples] += artifact
    return signal


def add_powerline_interference(signal, freq_hz=60, amplitude_mv=0.5, fs=SAMPLING_RATE_HZ):
    t = np.arange(len(signal)) / fs
    return signal + amplitude_mv * np.sin(2 * np.pi * freq_hz * t)


def apply_robustness_pipeline(signal, condition):
    """
    Apply one of the perturbation conditions from Table 12/13.
    condition: str, one of 'snr20', 'snr10', 'snr5', 'amp_scale', 'time_shift',
               'baseline', 'motion', 'powerline'.
    Returns perturbed signal.
    """
    if condition == "snr20":
        return add_gaussian_noise(signal, 20)
    elif condition == "snr10":
        return add_gaussian_noise(signal, 10)
    elif condition == "snr5":
        return add_gaussian_noise(signal, 5)
    elif condition == "amp_scale":
        scale = np.random.uniform(0.9, 1.1)
        return amplitude_scale(signal, scale)
    elif condition == "time_shift":
        shift = np.random.randint(-10, 11)
        return temporal_shift(signal, shift)
    elif condition == "baseline":
        freq = np.random.uniform(0.1, 1.0)
        amp = np.random.uniform(0, 2.0)
        return add_baseline_wander(signal, freq, amp)
    elif condition == "motion":
        freq = np.random.uniform(10, 100)
        amp = np.random.uniform(0.5, 2.0)
        duration = 50  # ms
        return add_motion_artifact(signal, freq, amp, duration)
    elif condition == "powerline":
        return add_powerline_interference(signal, 60, 0.5)
    else:
        return signal
