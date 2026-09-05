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
