from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(array)), float(np.std(array, ddof=0))


def trimmed_mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size >= 3:
        array = np.sort(array)[1:-1]
    return mean_sd(array)


def paired_difference(first: Iterable[float], second: Iterable[float]) -> Tuple[np.ndarray, float, float]:
    a = np.asarray(list(first), dtype=np.float64)
    b = np.asarray(list(second), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape.")
    difference = a - b
    return difference, float(np.mean(difference)), float(np.std(difference, ddof=0))


def bootstrap_mean_ci(
    values: Iterable[float],
    seed: int,
    confidence: float = 0.95,
    resamples: int = 5000,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, array.size, size=(int(resamples), array.size))
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.mean(array)), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
