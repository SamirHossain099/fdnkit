import numpy as np
import pytest

from fdnkit.preprocessing import flag_bad_channels, segment, sliding_windows, zscore


def test_zscore_unit_variance_zero_mean():
    x = np.random.default_rng(0).standard_normal((4, 1000)) * 3 + 5
    z = zscore(x)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_zscore_handles_flat_channel():
    x = np.vstack([np.ones(100), np.arange(100.0)])
    z = zscore(x)
    assert np.all(np.isfinite(z))


def test_flag_bad_flat_channel():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 1000))
    x[2] = 0.0  # flat
    bad = flag_bad_channels(x)
    assert 2 in bad


def test_flag_bad_by_name_prefix():
    x = np.random.default_rng(0).standard_normal((3, 500))
    bad = flag_bad_channels(x, channel_names=["LH1", "EKG1", "LH2"])
    assert 1 in bad


def test_flag_bad_amplitude_outlier():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 1000))
    x[0] *= 100  # huge amplitude
    bad = flag_bad_channels(x)
    assert 0 in bad


def test_segment_non_overlapping():
    x = np.arange(1000.0)
    wins = sliding_windows(x, 100)
    assert len(wins) == 10
    assert wins[0][0] == 0 and wins[0][1] == 100
    assert wins[-1][1] == 1000


def test_segment_drops_partial_tail():
    x = np.arange(950.0)
    wins = sliding_windows(x, 100)
    assert len(wins) == 9  # last 50 dropped


def test_segment_strided():
    x = np.arange(1000.0)
    wins = sliding_windows(x, 100, step=50)
    assert len(wins) == 19  # overlapping windows


def test_segment_2d_keeps_channels():
    x = np.random.default_rng(0).standard_normal((6, 1000))
    _, _, chunk = sliding_windows(x, 250)[0]
    assert chunk.shape == (6, 250)


def test_segment_rejects_bad_window():
    with pytest.raises(ValueError):
        list(segment(np.arange(10.0), 0))
