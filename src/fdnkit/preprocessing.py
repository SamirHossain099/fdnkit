"""Preprocessing helpers: normalization, artifact-channel flagging, windowing.

Kept deliberately small -- FDNkit depends on MNE-Python for montages, filtering,
and ICA rather than reimplementing them. These utilities cover only what the
fractal/FODN pipeline needs: z-scoring, dropping obviously-bad channels,
screening for flat (constant) runs, and cutting a recording into fixed-length
analysis windows.
"""

from __future__ import annotations

import numpy as np

__all__ = ["zscore", "flag_bad_channels", "find_flat_runs", "flat_fraction",
           "segment", "sliding_windows"]


def zscore(signals, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Z-score signals along ``axis`` (per-channel by default).

    Parameters
    ----------
    signals : array-like
        1-D or 2-D ``(n_channels, n_samples)`` array.
    axis : int
        Axis along which to standardize (default ``-1`` = time).
    eps : float
        Floor for the standard deviation to avoid divide-by-zero on flat channels.
    """
    x = np.asarray(signals, dtype=float)
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def find_flat_runs(signal, min_length: int = 16, atol: float = 0.0):
    """Locate runs of (near-)constant samples in a 1-D signal.

    Flat runs arise from amplifier saturation, clipping, dropped or interpolated
    samples, and disconnected channels. They matter for multifractal analysis:
    a flat run makes the integrated profile locally linear, so a detrended
    segment falling inside it has (near-)zero variance. Negative moment orders
    weight each segment as ``[F^2]^(-|q|/2)``, so one such segment can dominate
    ``F_q(s)`` and produce a spuriously broad singularity spectrum. See
    :func:`fdnkit.mfdfa.mfdfa` and the references in its ``rel_floor`` notes.

    Parameters
    ----------
    signal : array-like
        1-D time series.
    min_length : int
        Minimum run length, in samples, to report. Runs shorter than the
        smallest MFDFA scale cannot contain a whole segment, so this defaults to
        a value near the small end of the usual scale grid.
    atol : float
        Absolute tolerance on successive differences. ``0.0`` (the default)
        finds exactly-repeated samples, which is what saturation and dropout
        produce; raise it to catch near-flat stretches.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_runs, 2)`` with half-open ``[start, stop)`` sample
        bounds, sorted by start. Empty with shape ``(0, 2)`` if none are found.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(100.0); x[40:60] = x[40]
    >>> find_flat_runs(x, min_length=10)
    array([[40, 60]])
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        return np.empty((0, 2), dtype=int)
    if min_length < 2:
        raise ValueError("min_length must be at least 2")

    same = (np.abs(np.diff(x)) <= atol).astype(np.int8)
    # Boundaries of maximal runs of equal successive samples.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], same, [0]))))
    starts, stops = edges[0::2], edges[1::2]
    # `same[i]` true means x[i] == x[i+1], so a diff-run [a, b) spans samples [a, b].
    lengths = stops - starts + 1
    keep = lengths >= min_length
    if not np.any(keep):
        return np.empty((0, 2), dtype=int)
    return np.column_stack((starts[keep], starts[keep] + lengths[keep])).astype(int)


def flat_fraction(signal, min_length: int = 16, atol: float = 0.0) -> float:
    """Fraction of samples lying inside flat runs (see :func:`find_flat_runs`).

    A convenient quality-control scalar: values above roughly 1% are worth
    investigating before trusting negative-``q`` multifractal estimates.
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    runs = find_flat_runs(x, min_length=min_length, atol=atol)
    if runs.size == 0:
        return 0.0
    return float((runs[:, 1] - runs[:, 0]).sum() / x.size)


def flag_bad_channels(
    signals,
    channel_names=None,
    *,
    flat_std: float = 1e-8,
    amplitude_z: float = 5.0,
    max_flat_fraction: float | None = 0.05,
    flat_run_length: int = 16,
    name_prefixes=("EKG", "ECG", "X1 DC", "DC", "TRIG", "STI"),
) -> list[int]:
    """Return indices of channels that look like artifacts.

    A channel is flagged if it is (a) effectively flat, (b) has an
    root-mean-square amplitude more than ``amplitude_z`` robust-SDs from the
    median across channels, (c) spends more than ``max_flat_fraction`` of its
    samples inside constant runs (saturation, clipping, dropout), or (d) its
    name starts with a known non-neural prefix (EKG, trigger, DC, ...).

    Parameters
    ----------
    signals : array-like, shape (n_channels, n_samples)
    channel_names : sequence of str, optional
        Used only for prefix-based flagging.
    flat_std : float
        Channels with standard deviation below this are flat.
    amplitude_z : float
        Robust z-threshold on per-channel RMS.
    max_flat_fraction : float or None
        Flag channels whose :func:`flat_fraction` exceeds this. Set to ``None``
        to skip the check.
    flat_run_length : int
        Minimum run length counted as flat, passed to :func:`find_flat_runs`.
    name_prefixes : tuple of str
        Case-insensitive channel-name prefixes to treat as non-neural.
    """
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2:
        raise ValueError("signals must be 2-D (n_channels, n_samples)")
    n_ch = x.shape[0]
    bad = set()

    std = x.std(axis=1)
    bad.update(np.where(std < flat_std)[0].tolist())

    rms = np.sqrt(np.mean(x**2, axis=1))
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-12
    robust_z = 0.6745 * (rms - med) / mad
    bad.update(np.where(np.abs(robust_z) > amplitude_z)[0].tolist())

    if max_flat_fraction is not None:
        for i in range(n_ch):
            if flat_fraction(x[i], min_length=flat_run_length) > max_flat_fraction:
                bad.add(i)

    if channel_names is not None:
        prefixes = tuple(p.upper() for p in name_prefixes)
        for i, name in enumerate(channel_names):
            if i < n_ch and str(name).upper().startswith(prefixes):
                bad.add(i)

    return sorted(bad)


def segment(signals, window: int, *, step: int = None, min_size: int = None):
    """Cut a signal into non-overlapping (or strided) windows along time.

    Parameters
    ----------
    signals : array-like
        1-D ``(n_samples,)`` or 2-D ``(n_channels, n_samples)`` array.
    window : int
        Window length in samples.
    step : int, optional
        Hop size in samples. Defaults to ``window`` (non-overlapping).
    min_size : int, optional
        Discard a trailing window shorter than this. Defaults to ``window``
        (i.e. drop any partial final window).

    Yields
    ------
    (start, stop, chunk) : tuple[int, int, numpy.ndarray]
        Sample bounds and the windowed data (``chunk`` keeps the input ndim).
    """
    x = np.asarray(signals)
    if window <= 0:
        raise ValueError("window must be positive")
    step = window if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    min_size = window if min_size is None else min_size
    n = x.shape[-1]
    start = 0
    while start < n:
        stop = min(start + window, n)
        if stop - start < min_size:
            break
        chunk = x[..., start:stop]
        yield start, stop, chunk
        start += step


def sliding_windows(signals, window: int, *, step: int = None, min_size: int = None):
    """List form of :func:`segment` -- returns ``[(start, stop, chunk), ...]``."""
    return list(segment(signals, window, step=step, min_size=min_size))
