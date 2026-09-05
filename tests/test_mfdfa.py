import numpy as np
import pytest

from fdnkit.dfa import hurst
from fdnkit.mfdfa import (
    MFDFAResult,
    delta_hq,
    generalized_hurst,
    mfdfa,
    multifractal_spectrum,
)
from fdnkit.synthetic import binomial_cascade, fgn


def test_mfdfa_q2_matches_dfa_hurst():
    x = fgn(8192, 0.7, seed=3)
    res = mfdfa(x, q=[2])
    # h(q=2) equals the monofractal DFA Hurst exponent.
    assert abs(res.hq_at(2) - hurst(x)) < 1e-9
    assert abs(res.hurst - hurst(x)) < 1e-9


def test_monofractal_has_narrow_spectrum_multifractal_wide():
    mono = delta_hq(fgn(16384, 0.7, seed=1))
    multi = delta_hq(binomial_cascade(14, 0.3, seed=1))
    assert multi > mono
    assert multi > 0.3


def test_cascade_hq_decreasing():
    res = mfdfa(binomial_cascade(14, 0.25, seed=2))
    order = np.argsort(res.q)
    hq_sorted = res.hq[order]
    # h(q) is non-increasing in q for a multifractal (allow tiny numerical noise).
    assert np.all(np.diff(hq_sorted) <= 1e-3)


def test_generalized_hurst_wrapper():
    q, hq = generalized_hurst(fgn(4096, 0.6, seed=0))
    assert q.shape == hq.shape
    assert np.all(np.isfinite(hq))


def test_multifractal_spectrum_shape():
    res = mfdfa(binomial_cascade(13, 0.3, seed=0))
    alpha, f_alpha = multifractal_spectrum(res)
    assert alpha.shape == f_alpha.shape
    assert alpha.size >= 3
    # f(alpha) peak should be near 1 (support dimension of a full-measure cascade).
    assert f_alpha.max() <= 1.05


def test_mfdfa_result_type():
    assert isinstance(mfdfa(fgn(2048, 0.5, seed=0)), MFDFAResult)


def test_mfdfa_rejects_short_signal():
    with pytest.raises(ValueError):
        mfdfa(np.arange(6.0))


def test_quantized_signal_does_not_explode_delta_h():
    # Integer-quantized real recordings produce degenerately-detrended segments
    # whose RMS collapses to ~0; without a scale-relative floor these blow up the
    # negative-q moments and fabricate a huge delta_h. The floor keeps it sane.
    x = fgn(16384, 0.7, seed=5)
    quantized = np.round(x * 8) / 8.0  # coarse quantization -> many flat segments

    guarded = mfdfa(quantized).delta_h            # default rel_floor=1e-3
    unguarded = mfdfa(quantized, rel_floor=0.0).delta_h

    assert np.isfinite(guarded)
    assert guarded < 3.0, f"guarded delta_h unexpectedly large: {guarded}"
    assert unguarded > guarded  # disabling the floor makes it worse


def test_rel_floor_preserves_clean_signal():
    # On a well-behaved signal the floor never binds: results match rel_floor=0.
    x = fgn(8192, 0.6, seed=2)
    assert abs(mfdfa(x).delta_h - mfdfa(x, rel_floor=0.0).delta_h) < 1e-9
