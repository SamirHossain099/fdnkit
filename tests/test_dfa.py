import numpy as np
import pytest

from fdnkit.dfa import DFAResult, dfa, hurst
from fdnkit.synthetic import fgn


@pytest.mark.parametrize("target", [0.3, 0.5, 0.7, 0.9])
def test_dfa_recovers_hurst_of_fgn(target):
    # Average over realizations to beat single-sample variance.
    est = np.mean([hurst(fgn(8192, target, seed=s)) for s in range(6)])
    assert abs(est - target) < 0.07, f"H={est} far from target {target}"


def test_white_noise_hurst_near_half():
    rng = np.random.default_rng(0)
    ests = [hurst(rng.standard_normal(8192)) for _ in range(5)]
    assert abs(np.mean(ests) - 0.5) < 0.08


def test_brownian_motion_hurst_near_one_and_half():
    rng = np.random.default_rng(0)
    ests = [hurst(np.cumsum(rng.standard_normal(8192))) for _ in range(5)]
    assert abs(np.mean(ests) - 1.5) < 0.1


def test_dfa_result_fields():
    res = dfa(fgn(4096, 0.7, seed=0))
    assert isinstance(res, DFAResult)
    assert np.isfinite(res.hurst)
    assert res.fluct.shape == res.scales.shape
    assert np.all(res.fluct[np.isfinite(res.fluct)] > 0)


def test_dfa_raises_on_too_short_signal():
    with pytest.raises(ValueError):
        dfa(np.arange(5.0))


def test_dfa_rejects_scales_the_polynomial_fits_exactly():
    # A cubic passes through any 4 samples, so the default grid (which starts at 4)
    # leaves only rounding error at its first scale. Before this was rejected, the
    # fit below returned H near 3.6 for a signal whose true exponent is 0.7.
    x = fgn(8000, 0.7, seed=0)
    with pytest.raises(ValueError, match=r"order \+ 2 = 5"):
        dfa(x, order=3)
    with pytest.raises(ValueError, match="too small"):
        dfa(x, scales=[2, 4, 8, 16], order=1)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_dfa_recovers_hurst_at_higher_detrending_orders(order):
    # With every scale at or above order + 2 the estimate stays near the truth
    # whatever the detrending order.
    scales = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    est = np.mean([dfa(fgn(8192, 0.7, seed=s), scales=scales, order=order).hurst
                   for s in range(4)])
    assert abs(est - 0.7) < 0.07, f"order {order}: H={est}"


def test_dfa_accepts_the_smallest_valid_scale():
    # order + 2 is the boundary: one residual degree of freedom, a real fluctuation.
    res = dfa(fgn(4096, 0.7, seed=0), scales=[3, 6, 12, 24, 48], order=1)
    assert np.all(res.fluct > 1e-6)


@pytest.mark.parametrize("scales", [[0, 4, 8], [-4, 8, 16], [], [[4, 8], [16, 32]]])
def test_dfa_rejects_malformed_scales(scales):
    with pytest.raises(ValueError):
        dfa(fgn(1024, 0.7, seed=0), scales=scales)
