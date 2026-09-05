"""Preprocessing helpers: normalization, artifact-channel flagging, windowing.

Kept deliberately small -- FDNkit depends on MNE-Python for montages, filtering,
and ICA rather than reimplementing them. These utilities cover only what the
fractal/FODN pipeline needs: z-scoring, dropping obviously-bad channels, and
cutting a recording into fixed-length analysis windows.
"""

from __future__ import annotations

import numpy as np

__all__ = ["zscore", "flag_bad_channels", "segment", "sliding_windows"]


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


def flag_bad_channels(
    signals,
    channel_names=None,
    *,
    flat_std: float = 1e-8,
    amplitude_z: float = 5.0,
    name_prefixes=("EKG", "ECG", "X1 DC", "DC", "TRIG", "STI"),
) -> list[int]:
    """Return indices of channels that look like artifacts.

    A channel is flagged if it is (a) effectively flat, (b) has an
    root-mean-square amplitude more than ``amplitude_z`` robust-SDs from the
    median across channels, or (c) its name starts with a known non-neural
    prefix (EKG, trigger, DC, ...).

    Parameters
    ----------
    signals : array-like, shape (n_channels, n_samples)
    channel_names : sequence of str, optional
        Used only for prefix-based flagging.
    flat_std : float
        Channels with standard deviation below this are flat.
    amplitude_z : float
        Robust z-threshold on per-channel RMS.
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
