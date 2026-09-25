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

    # check_flat=False: this test is about the floor, not the flat-run warning.
    guarded = mfdfa(quantized, check_flat=False).delta_h        # default rel_floor=1e-3
    unguarded = mfdfa(quantized, rel_floor=0.0, check_flat=False).delta_h

    assert np.isfinite(guarded)
    assert guarded < 3.0, f"guarded delta_h unexpectedly large: {guarded}"
    assert unguarded > guarded  # disabling the floor makes it worse


def test_rel_floor_preserves_clean_signal():
    # On a well-behaved signal the floor never binds: results match rel_floor=0.
    x = fgn(8192, 0.6, seed=2)
    assert abs(mfdfa(x).delta_h - mfdfa(x, rel_floor=0.0).delta_h) < 1e-9


def test_warns_on_flat_runs_with_negative_q():
    import warnings

    x = fgn(16384, 0.7, seed=1)
    x[5000:5032] = x[5000]  # one flat run, as from dropout or saturation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mfdfa(x)
    assert any(issubclass(rec.category, RuntimeWarning) for rec in w)
    assert "constant run" in str(w[0].message)


def test_no_flat_warning_when_clean_or_positive_q_only():
    import warnings

    clean = fgn(8192, 0.7, seed=2)
    flat = clean.copy()
    flat[1000:1064] = flat[1000]
    for sig, kwargs in ((clean, {}), (flat, {"q": [1, 2, 3]}), (flat, {"check_flat": False})):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mfdfa(sig, **kwargs)
        assert not [r for r in w if issubclass(r.category, RuntimeWarning)]


def test_flat_run_inflates_delta_h_when_unguarded():
    # Documents the failure mode the QC step exists to catch.
    x = fgn(16384, 0.7, seed=1)
    flat = x.copy()
    flat[5000:5032] = flat[5000]
    scales = [8, 16, 32, 64, 128, 256]
    q = [-5, -3, -1, 1, 3, 5]
    clean_dh = mfdfa(x, scales=scales, q=q, rel_floor=0.0, check_flat=False).delta_h
    flat_dh = mfdfa(flat, scales=scales, q=q, rel_floor=0.0, check_flat=False).delta_h
    assert flat_dh > 10 * clean_dh


# --- scale invariance -------------------------------------------------------------------------
#
# h(q) is defined so that multiplying a signal by a positive constant leaves it unchanged: the
# scaling of the fluctuation function with window size cannot depend on the unit the signal is
# measured in. An implementation that discards segments using an *absolute* variance or
# fluctuation cutoff silently breaks this, because rescaling moves segments across a fixed
# threshold and changes which ones contribute to the log-log fit. The failure is quiet and
# unit-dependent, which is what makes it worth a regression test rather than a docstring note.
#
# `rel_floor` is the guard: the floor is a fraction of the median positive fluctuation at each
# scale, so it moves with the signal instead of staying put. These tests pin that in both
# directions, including the case where the guard is disabled.

RESCALE_FACTORS = (1e6, 1e3, 1e0, 1e-3, 1e-6)


def test_hq_is_invariant_under_rescaling():
    # Twelve orders of magnitude of input scaling must not move h(q). The bound is 1e-8 rather
    # than exactly zero because factors such as 1e-3 are not exactly representable in binary, so
    # the rescaled signal is not a perfectly scaled copy of the original. Measured worst drift is
    # around 7e-10, so this leaves headroom without being vacuous.
    x = fgn(8192, 0.7, seed=11)
    reference = mfdfa(x).hq

    for factor in RESCALE_FACTORS:
        drift = np.max(np.abs(mfdfa(x * factor).hq - reference))
        assert drift < 1e-8, f"h(q) moved by {drift:.3g} when the input was scaled by {factor:g}"


def test_white_noise_exponent_is_invariant_and_near_one_half():
    # Checks invariance against a known answer rather than only against self-consistency. Only
    # h(q=2) is asserted to sit near 0.5: at negative q a finite white-noise sample legitimately
    # spreads well above that, so asserting it across the whole q grid would be wrong.
    x = np.random.default_rng(5).standard_normal(8192)
    reference = mfdfa(x).hq_at(2)

    assert abs(reference - 0.5) < 0.1
    for factor in RESCALE_FACTORS:
        # Worst measured drift is around 9e-11, at the 1e-6 factor.
        assert abs(mfdfa(x * factor).hq_at(2) - reference) < 1e-9


def test_relative_floor_contains_drift_that_an_absolute_floor_does_not():
    # The complement of the tests above. A constant run drives many fluctuations down onto the
    # floor, which is exactly where an absolute threshold starts behaving like a unit conversion.
    # With rel_floor disabled the implementation falls back to a machine-epsilon absolute floor.
    # Without this test the guard could be deleted and the rest of the suite would still pass.
    x = fgn(8192, 0.7, seed=17)
    x[1000:1064] = 0.0

    def worst_drift(**kwargs):
        curves = [mfdfa(x * f, check_flat=False, **kwargs).hq for f in RESCALE_FACTORS]
        baseline = curves[RESCALE_FACTORS.index(1e0)]
        return max(np.max(np.abs(c - baseline)) for c in curves)

    guarded = worst_drift()
    unguarded = worst_drift(rel_floor=0.0)

    # Measured: guarded around 1.6e-4, unguarded around 4.0, a factor of roughly 25,000. The
    # guarded value is not exactly zero because segments sitting on the floor can cross it under
    # inexact rescaling; that is boundary jitter, not unit dependence.
    assert guarded < 1e-2, f"the relative floor let h(q) drift by {guarded:.3g}"
    assert unguarded > 1.0, f"expected the absolute fallback to drift badly, got {unguarded:.3g}"
    assert unguarded > 100 * guarded
