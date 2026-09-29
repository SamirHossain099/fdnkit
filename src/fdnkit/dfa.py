"""Monofractal detrended fluctuation analysis (DFA).

DFA estimates the Hurst exponent ``H`` of a time series: the scaling exponent of
its detrended root-mean-square fluctuation against window size. It is the
``q = 2`` special case of :mod:`fdnkit.mfdfa`, exposed here as a lightweight,
single-purpose entry point.

Reference: Peng et al. (1994); the implementation matches the validated
reference and MATLAB code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mfdfa import DEFAULT_SCALES, _fluctuations, _loglog_slope, _validate_scales

__all__ = ["DFAResult", "dfa", "hurst"]


@dataclass
class DFAResult:
    """Output of :func:`dfa`.

    Attributes
    ----------
    hurst : float
        Hurst exponent (log-log slope of fluctuation vs scale).
    scales : numpy.ndarray
        Window sizes used.
    fluct : numpy.ndarray
        Fluctuation function ``F`` per scale.
    """

    hurst: float
    scales: np.ndarray
    fluct: np.ndarray


def dfa(signal, scales=None, order: int = 1, rel_floor: float = 1e-3) -> DFAResult:
    """Estimate the Hurst exponent of a 1-D signal by DFA.

    Parameters
    ----------
    signal : array-like
        1-D time series.
    scales : array-like, optional
        Window sizes in samples. Defaults to the standard FDNkit grid, which
        starts at 4. Every scale must be at least ``order + 2``.
    order : int
        Detrending polynomial order (1 = linear). With ``order >= 3`` pass
        ``scales`` explicitly, since the default grid starts below
        ``order + 2``.
    rel_floor : float
        Scale-relative floor on per-segment fluctuations (see
        :func:`fdnkit.mfdfa.mfdfa`). Has negligible effect on the (positive-moment)
        Hurst estimate but keeps behaviour consistent with MFDFA.

    Returns
    -------
    DFAResult

    Raises
    ------
    ValueError
        If any scale is below ``order + 2``, or the signal is shorter than twice
        the smallest scale.

    Examples
    --------
    >>> from fdnkit.synthetic import fgn
    >>> from fdnkit.dfa import dfa
    >>> H = dfa(fgn(8000, 0.7, seed=0)).hurst  # ~0.7
    """
    eps = np.finfo(float).eps
    scales = _validate_scales(DEFAULT_SCALES if scales is None else scales, order)
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < int(scales.min()) * 2:
        raise ValueError(
            f"signal length {x.size} too short for smallest scale {int(scales.min())}"
        )

    rms_per_scale = _fluctuations(x, scales, order, rel_floor=rel_floor)
    fluct = np.full(len(scales), np.nan)
    for i, rms in enumerate(rms_per_scale):
        if rms.size:
            fluct[i] = np.sqrt(np.mean(rms**2))

    H = _loglog_slope(scales, fluct + eps)
    return DFAResult(hurst=H, scales=np.asarray(scales), fluct=fluct)


def hurst(signal, scales=None, order: int = 1) -> float:
    """Return just the Hurst exponent (shorthand for ``dfa(...).hurst``)."""
    return dfa(signal, scales=scales, order=order).hurst
