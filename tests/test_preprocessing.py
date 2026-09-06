import numpy as np
import pytest

from fdnkit.preprocessing import (
    find_flat_runs,
    flag_bad_channels,
    flat_fraction,
    segment,
    sliding_windows,
    zscore,
)


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


def test_find_flat_runs_basic():
    x = np.arange(100.0)
    x[40:60] = x[40]
    runs = find_flat_runs(x, min_length=10)
    assert runs.tolist() == [[40, 60]]


def test_find_flat_runs_none_and_all():
    assert find_flat_runs(np.arange(100.0), min_length=10).shape == (0, 2)
    assert find_flat_runs(np.zeros(50), min_length=10).tolist() == [[0, 50]]


def test_find_flat_runs_respects_min_length():
    x = np.arange(100.0)
    x[10:15] = x[10]  # 5-sample run
    assert find_flat_runs(x, min_length=10).shape == (0, 2)
    assert find_flat_runs(x, min_length=4).tolist() == [[10, 15]]


def test_find_flat_runs_multiple_and_tolerance():
    x = np.arange(200.0)
    x[20:40] = x[20]
    x[100:130] = x[100]
    assert find_flat_runs(x, min_length=10).tolist() == [[20, 40], [100, 130]]
    y = np.arange(100.0) * 1e-15  # near-flat, not exactly flat
    assert find_flat_runs(y, min_length=10).shape == (0, 2)
    assert find_flat_runs(y, min_length=10, atol=1e-12).tolist() == [[0, 100]]


def test_find_flat_runs_edge_cases():
    assert find_flat_runs(np.array([1.0]), min_length=2).shape == (0, 2)
    with pytest.raises(ValueError):
        find_flat_runs(np.arange(10.0), min_length=1)


def test_flat_fraction():
    x = np.arange(100.0)
    x[40:60] = x[40]
    assert abs(flat_fraction(x, min_length=10) - 0.20) < 1e-9
    assert flat_fraction(np.arange(100.0), min_length=10) == 0.0


def test_flag_bad_channels_flags_partially_flat_channel():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 2000))
    x[1, 200:600] = x[1, 200]  # 20% flat, well above the 5% default
    assert 1 in flag_bad_channels(x)
    assert 1 not in flag_bad_channels(x, max_flat_fraction=None)
