import numpy as np
import pytest

from fdnkit.fodn import FODN, FODNResult, HaarWaveletTransform, fit_fodn
from fdnkit.synthetic import synthetic_ieeg


def test_fodn_runs_and_shapes():
    sig, _ = synthetic_ieeg(n_channels=6, n_samples=1000, seed=0)
    res = fit_fodn(sig, n_iter=3)
    assert isinstance(res, FODNResult)
    assert res.alpha.shape == (6,)
    assert res.coupling.shape == (6, 6)
    assert res.eigenvalues.shape == (6,)
    assert res.dominant_eigvec.shape == (6,)


def test_fodn_outputs_finite():
    sig, _ = synthetic_ieeg(n_channels=5, n_samples=800, seed=1)
    res = fit_fodn(sig, n_iter=2)
    assert np.all(np.isfinite(res.alpha))
    assert np.all(np.isfinite(res.coupling))
    assert np.isfinite(res.leading_eig)
    assert 0.0 <= res.sparseness <= 1.0


def test_fodn_deterministic():
    sig, _ = synthetic_ieeg(n_channels=5, n_samples=800, seed=2)
    a = fit_fodn(sig, n_iter=2)
    b = fit_fodn(sig, n_iter=2)
    assert np.allclose(a.coupling, b.coupling)
    assert np.allclose(a.alpha, b.alpha)


def test_fodn_rejects_single_channel():
    with pytest.raises(ValueError):
        FODN().fit(np.random.default_rng(0).standard_normal((1, 500)))


def test_fodn_rejects_more_channels_than_samples():
    with pytest.raises(ValueError):
        FODN().fit(np.random.default_rng(0).standard_normal((50, 10)))


def test_result_before_fit_raises():
    with pytest.raises(RuntimeError):
        FODN().result()


def test_haar_transform_shapes():
    x = np.random.default_rng(0).standard_normal(256)
    wt = HaarWaveletTransform(x)
    wt.normalize()
    approx, detail = wt.transform()
    assert approx.shape == detail.shape
    assert approx.shape[0] == 128


def test_haar_rejects_2d():
    with pytest.raises(ValueError):
        HaarWaveletTransform(np.ones((4, 4)))
