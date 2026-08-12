import numpy as np

from mhfl_review.data import add_gaussian_noise, empirical_snr_db, segment_nonoverlap, zscore_per_segment


def test_nonoverlap_segmentation_and_ids():
    values = np.arange(20, dtype=np.float32)
    segments, starts = segment_nonoverlap(values, length=4, max_segments=3)
    assert segments.shape == (3, 4)
    assert np.array_equal(segments[1], np.array([4, 5, 6, 7], dtype=np.float32))
    assert list(starts) == [0, 4, 8]


def test_per_segment_zscore():
    x = np.array([[[1.0], [2.0], [3.0]], [[4.0], [8.0], [12.0]]], dtype=np.float32)
    z = zscore_per_segment(x)
    assert np.allclose(np.mean(z, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.std(z, axis=1), 1.0, atol=1e-6)


def test_noise_snr_is_close():
    rng = np.random.RandomState(42)
    x = rng.normal(size=(64, 2048, 1)).astype(np.float32)
    noisy = add_gaussian_noise(x, -4.0, np.random.RandomState(7))
    assert abs(empirical_snr_db(x, noisy) - (-4.0)) < 0.35
