from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import kurtosis, skew


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return numerator / np.maximum(denominator, eps)


def extract_signal_features(
    signals: np.ndarray,
    sampling_rate: float,
    *,
    n_bands: int = 8,
) -> np.ndarray:
    """Extract compact time/frequency-domain features from 2048-point segments.

    This deliberately uses established, interpretable signal descriptors so that
    the resulting SVMs serve as traditional signal-processing benchmarks rather
    than additional neural models.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(f"Expected a singleton channel dimension, got {x.shape}.")
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected shape (samples, points) or (samples, points, 1), got {x.shape}.")

    centered = x - np.mean(x, axis=1, keepdims=True)
    abs_x = np.abs(centered)
    std = np.std(centered, axis=1)
    rms = np.sqrt(np.mean(np.square(centered), axis=1))
    abs_mean = np.mean(abs_x, axis=1)
    peak = np.max(abs_x, axis=1)
    peak_to_peak = np.ptp(centered, axis=1)
    root_abs_mean = np.square(np.mean(np.sqrt(abs_x + 1e-12), axis=1))

    time_features = np.column_stack(
        [
            np.mean(x, axis=1),
            std,
            rms,
            abs_mean,
            peak,
            peak_to_peak,
            skew(centered, axis=1, bias=False, nan_policy="omit"),
            kurtosis(centered, axis=1, fisher=False, bias=False, nan_policy="omit"),
            _safe_divide(peak, rms),                 # crest factor
            _safe_divide(rms, abs_mean),             # shape factor
            _safe_divide(peak, abs_mean),            # impulse factor
            _safe_divide(peak, root_abs_mean),       # clearance factor
            np.mean(np.diff(np.signbit(centered), axis=1), axis=1),  # zero-crossing rate
        ]
    )

    window = np.hanning(x.shape[1])[None, :]
    spectrum = np.fft.rfft(centered * window, axis=1)
    power = np.square(np.abs(spectrum)) + 1e-18
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / float(sampling_rate))
    total_power = np.sum(power, axis=1)
    prob = power / total_power[:, None]
    centroid = np.sum(prob * freqs[None, :], axis=1)
    bandwidth = np.sqrt(np.sum(prob * np.square(freqs[None, :] - centroid[:, None]), axis=1))
    dominant_frequency = freqs[np.argmax(power, axis=1)]
    spectral_entropy = -np.sum(prob * np.log(prob), axis=1) / np.log(power.shape[1])

    positive_power = power[:, 1:] if power.shape[1] > 1 else power
    positive_freqs = freqs[1:] if len(freqs) > 1 else freqs
    low_frequency_ratio = np.sum(positive_power[:, positive_freqs <= sampling_rate * 0.05], axis=1) / np.sum(
        positive_power, axis=1
    )

    freq_features = [
        total_power,
        centroid,
        bandwidth,
        dominant_frequency,
        spectral_entropy,
        low_frequency_ratio,
    ]

    edges = np.linspace(0, len(freqs), int(n_bands) + 1, dtype=int)
    band_features = []
    for left, right in zip(edges[:-1], edges[1:]):
        right = max(right, left + 1)
        band_features.append(np.sum(power[:, left:right], axis=1) / total_power)

    features = np.column_stack([time_features, *freq_features, *band_features])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_multimodal_features(
    x1: np.ndarray,
    x2: np.ndarray,
    sampling_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return modality-1, modality-2, and early-fusion feature matrices."""
    f1 = extract_signal_features(x1, sampling_rate)
    f2 = extract_signal_features(x2, sampling_rate)
    return f1, f2, np.concatenate([f1, f2], axis=1).astype(np.float32)
