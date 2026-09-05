import numpy as np
import pytest

from fdnkit.synthetic import binomial_cascade, fbm, fgn, synthetic_ieeg


def test_fgn_shape_and_reproducibility():
    a = fgn(1024, 0.7, seed=42)
    b = fgn(1024, 0.7, seed=42)
    assert a.shape == (1024,)
    assert np.allclose(a, b)
    assert not np.allclose(a, fgn(1024, 0.7, seed=43))


def test_fgn_is_finite_and_zero_mean_ish():
    x = fgn(8192, 0.6, seed=1)
    assert np.all(np.isfinite(x))
    assert abs(x.mean()) < 0.1


@pytest.mark.parametrize("bad_h", [0.0, 1.0, -0.2, 1.5])
def test_fgn_rejects_out_of_range_hurst(bad_h):
    with pytest.raises(ValueError):
        fgn(256, bad_h)


def test_fbm_is_cumsum_of_fgn():
    g = fgn(512, 0.7, seed=7)
    b = fbm(512, 0.7, seed=7)
    assert np.allclose(b, np.cumsum(g))


def test_binomial_cascade_length_and_positive():
    c = binomial_cascade(10, 0.3, seed=0)
    assert c.shape == (2**10,)
    assert np.all(c >= 0)


def test_synthetic_ieeg_shape_and_names():
    sig, names = synthetic_ieeg(n_channels=8, n_samples=1000, seed=0)
    assert sig.shape == (8, 1000)
    assert names == [f"CH{i+1}" for i in range(8)]
    assert np.all(np.isfinite(sig))
