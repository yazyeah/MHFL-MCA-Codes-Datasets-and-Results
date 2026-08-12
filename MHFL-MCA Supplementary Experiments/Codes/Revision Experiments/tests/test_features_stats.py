import numpy as np

from mhfl_review.features import extract_signal_features
from mhfl_review.stats import trimmed_mean_sd


def test_features_are_finite():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(6, 2048, 1)).astype(np.float32)
    features = extract_signal_features(x, sampling_rate=25600.0)
    assert features.shape[0] == 6
    assert features.shape[1] >= 10
    assert np.isfinite(features).all()


def test_trimmed_mean():
    mean, sd = trimmed_mean_sd([1.0, 2.0, 3.0, 100.0])
    assert mean == 2.5
    assert sd >= 0.0
