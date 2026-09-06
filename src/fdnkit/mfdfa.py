"""Multifractal detrended fluctuation analysis (MFDFA).

Ports the validated reference implementation, cross-checked against its MATLAB
counterpart, into a clean, array-first API.

The generalized Hurst exponent ``h(q)`` describes how the ``q``-th order
fluctuation of a signal scales with window size. A signal is *monofractal* when
``h(q)`` is (nearly) constant in ``q`` and *multifractal* when it varies; the
width ``delta_h = max h(q) - min h(q)`` quantifies multifractality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MFDFAResult", "mfdfa", "generalized_hurst", "delta_hq", "multifractal_spectrum"]

# Default analysis grids (match the validated reference / MATLAB settings).
DEFAULT_SCALES = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
DEFAULT_Q = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5], dtype=float)


@dataclass
class MFDFAResult:
    """Container for the output of :func:`mfdfa`.

    Attributes
    ----------
    hurst : float
        Monofractal Hurst exponent ``H`` (slope of ``log2 F`` vs ``log2 scale``,
        equivalent to ``h(q=2)``).
    hq : numpy.ndarray
        Generalized Hurst exponents, one per entry of :attr:`q`.
    q : numpy.ndarray
        The moment orders used.
    scales : numpy.ndarray
        The window sizes used.
    fluct : numpy.ndarray
        Standard (``q=2``) fluctuation function ``F`` per scale.
    fluct_q : numpy.ndarray
        Fluctuation function per ``(q, scale)`` -- shape ``(len(q), len(scales))``.
    """

    hurst: float
    hq: np.ndarray
    q: np.ndarray
    scales: np.ndarray
    fluct: np.ndarray
    fluct_q: np.ndarray

    @property
    def delta_h(self) -> float:
        """Multifractal width ``max h(q) - min h(q)`` (0 for a monofractal)."""
        finite = self.hq[np.isfinite(self.hq)]
        if finite.size == 0:
            return float("nan")
        return float(finite.max() - finite.min())

    def hq_at(self, q_value: float) -> float:
        """Return ``h(q)`` at the grid point nearest ``q_value``."""
        idx = int(np.argmin(np.abs(self.q - q_value)))
        return float(self.hq[idx])


def _fluctuations(signal, scales, order, rel_floor: float = 1e-3):
    """Local detrended RMS fluctuations for each scale (q=2 base quantities).

    Returns a list whose ``i``-th entry is the array of per-segment RMS values at
    ``scales[i]`` (empty if the scale exceeds the signal length).

    ``rel_floor`` sets a *scale-relative* lower bound on each RMS value.

    This guards a documented failure mode of MFDFA rather than a novel one. A
    segment whose detrended variance is (near) zero makes ``F_q(s)`` diverge for
    ``q < 0``, because the negative moment weights it as ``[F^2]^(-|q|/2)``; a
    single such segment can dominate the sum and produce a spuriously broad
    singularity spectrum. This is well described in the spurious-multifractality
    literature -- see e.g. Ludescher et al. (2011, *Physica A*) on spurious and
    corrupted multifractality, and the finite-size/discreteness analysis in
    arXiv:2603.04609, which identifies exactly this zero-local-variance mechanism.

    Standard remedies in that literature are to restrict the analysis to positive
    ``q``, narrow ``|q|``, or drop the smallest scales. Flooring each segment at
    ``rel_floor * median(RMS)`` for its scale is a milder alternative that keeps
    negative ``q`` usable; it does not perturb well-behaved signals, whose
    fluctuations never approach the floor. Set ``rel_floor=0`` to disable it and
    reproduce the unguarded behaviour. Users relying on negative ``q`` should still
    validate against surrogates (phase randomization, shuffling) as recommended in
    that literature.
    """
    eps = np.finfo(float).eps
    x = np.asarray(signal, dtype=float)
    n = x.size
    y = np.cumsum(x - x.mean())

    rms_per_scale = []
    for s in scales:
        s = int(s)
        segs = n // s
        if segs == 0:  # scale larger than the signal
            rms_per_scale.append(np.empty(0))
            continue
        t = np.arange(s)
        rms = np.empty(segs)
        for v in range(segs):
            seg = y[v * s : (v + 1) * s]
            coef = np.polyfit(t, seg, order)
            fit = np.polyval(coef, t)
            rms[v] = np.sqrt(np.mean((seg - fit) ** 2))
        positive = rms[rms > 0]
        floor = rel_floor * np.median(positive) if positive.size else eps
        floor = max(floor, eps)
        np.maximum(rms, floor, out=rms)
        rms_per_scale.append(rms)
    return rms_per_scale


def _loglog_slope(scales, values):
    """Slope of log2(values) vs log2(scales) over finite, positive points."""
    lg_s = np.log2(np.asarray(scales, dtype=float))
    lg_v = np.log2(np.asarray(values, dtype=float))
    good = np.isfinite(lg_v) & np.isfinite(lg_s)
    if good.sum() < 2:
        return float("nan")
    # Ordinary least-squares slope.
    xs, ys = lg_s[good], lg_v[good]
    xm, ym = xs.mean(), ys.mean()
    denom = np.sum((xs - xm) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((xs - xm) * (ys - ym)) / denom)


def mfdfa(signal, scales=None, q=None, order: int = 1, rel_floor: float = 1e-3) -> MFDFAResult:
    """Run multifractal detrended fluctuation analysis on a 1-D signal.

    Parameters
    ----------
    signal : array-like
        1-D time series.
    scales : array-like, optional
        Window sizes (in samples). Defaults to
        ``[4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]``. Every scale
        should be smaller than ``len(signal)``.
    q : array-like, optional
        Moment orders. Defaults to ``[-5, -3, -2, -1, 0, 1, 2, 3, 5]``.
    order : int
        Order of the polynomial used to detrend each segment (1 = linear).
    rel_floor : float
        Scale-relative floor on per-segment fluctuations, as a fraction of the
        median fluctuation at each scale. Guards the negative-``q`` moments
        against near-zero-variance segments, a documented cause of spurious
        multifractality; set to 0 to disable. Does not affect well-behaved
        signals. See :func:`_fluctuations` for references.

    Returns
    -------
    MFDFAResult

    Notes
    -----
    Follows Kantelhardt et al. (2002) with a forward (non-overlapping) segment
    partition, matching the validated reference implementation. The ``q = 0``
    moment uses the logarithmic-average limit
    ``F_0(s) = exp(0.5 * mean(log RMS^2))``.
    """
    eps = np.finfo(float).eps
    scales = DEFAULT_SCALES if scales is None else np.asarray(scales)
    scales = np.asarray(scales, dtype=int)
    q = DEFAULT_Q if q is None else np.asarray(q, dtype=float)

    x = np.asarray(signal, dtype=float).ravel()
    if x.size < int(scales.min()) * 2:
        raise ValueError(
            f"signal length {x.size} too short for smallest scale {int(scales.min())}"
        )

    rms_per_scale = _fluctuations(x, scales, order, rel_floor=rel_floor)

    fluct = np.full(len(scales), np.nan)
    fluct_q = np.full((len(q), len(scales)), np.nan)
    for i, rms in enumerate(rms_per_scale):
        if rms.size == 0:
            continue
        fluct[i] = np.sqrt(np.mean(rms**2))
        for j, qq in enumerate(q):
            if qq == 0:
                fluct_q[j, i] = np.exp(0.5 * np.mean(np.log(rms**2)))
            else:
                fluct_q[j, i] = np.mean(rms**qq) ** (1.0 / qq)

    hurst = _loglog_slope(scales, fluct + eps)
    hq = np.array([_loglog_slope(scales, fluct_q[j] + eps) for j in range(len(q))])

    return MFDFAResult(
        hurst=hurst, hq=hq, q=np.asarray(q, dtype=float),
        scales=np.asarray(scales), fluct=fluct, fluct_q=fluct_q,
    )


def generalized_hurst(signal, scales=None, q=None, order: int = 1):
    """Convenience wrapper returning ``(q, h(q))`` arrays only."""
    res = mfdfa(signal, scales=scales, q=q, order=order)
    return res.q, res.hq


def delta_hq(signal, scales=None, q=None, order: int = 1) -> float:
    """Multifractal width ``max h(q) - min h(q)`` for a signal."""
    return mfdfa(signal, scales=scales, q=q, order=order).delta_h


def multifractal_spectrum(result: MFDFAResult):
    """Legendre-transform the generalized Hurst exponents to ``(alpha, f(alpha))``.

    Uses the standard MFDFA relations

    ``tau(q) = q * h(q) - 1``,
    ``alpha = d tau / d q``,
    ``f(alpha) = q * alpha - tau(q)``.

    Parameters
    ----------
    result : MFDFAResult
        Output of :func:`mfdfa` (needs at least 3 finite ``h(q)`` points).

    Returns
    -------
    alpha : numpy.ndarray
        Holder exponents (singularity strengths).
    f_alpha : numpy.ndarray
        Singularity spectrum values.
    """
    q = result.q
    hq = result.hq
    good = np.isfinite(hq)
    q, hq = q[good], hq[good]
    if q.size < 3:
        raise ValueError("need at least 3 finite h(q) points for a spectrum")
    order = np.argsort(q)
    q, hq = q[order], hq[order]
    tau = q * hq - 1.0
    alpha = np.gradient(tau, q)
    f_alpha = q * alpha - tau
    return alpha, f_alpha
