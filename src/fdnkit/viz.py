"""Plotting utilities (matplotlib, optional dependency).

Import matplotlib lazily so the core library has no hard plotting dependency.
Install with ``pip install fdnkit[viz]``. Every function accepts an optional
``ax`` and returns the Axes it drew on, so plots compose into larger figures.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_fluctuation",
    "plot_hq",
    "plot_multifractal_spectrum",
    "plot_hurst_over_time",
    "plot_alpha_distribution",
    "plot_coupling_matrix",
    "plot_eigenvector_hubs",
]


def _get_ax(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install fdnkit[viz]`."
        ) from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    return ax


def plot_fluctuation(result, ax=None, **kwargs):
    """Log-log fluctuation function ``F`` vs scale, with the fitted Hurst slope.

    Parameters
    ----------
    result : DFAResult | MFDFAResult
        Must expose ``scales``, ``fluct``, and ``hurst``.
    """
    ax = _get_ax(ax)
    scales = np.asarray(result.scales, dtype=float)
    fluct = np.asarray(result.fluct, dtype=float)
    good = np.isfinite(fluct) & (fluct > 0)
    ax.plot(np.log2(scales[good]), np.log2(fluct[good]), "o-", **kwargs)
    ax.set_xlabel("log2(scale)")
    ax.set_ylabel("log2(F)")
    ax.set_title(f"DFA fluctuation (H = {result.hurst:.3f})")
    return ax


def plot_hq(result, ax=None, **kwargs):
    """Generalized Hurst exponent ``h(q)`` against ``q``."""
    ax = _get_ax(ax)
    order = np.argsort(result.q)
    ax.plot(np.asarray(result.q)[order], np.asarray(result.hq)[order], "s-", **kwargs)
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.set_title(f"Generalized Hurst (delta_h = {result.delta_h:.3f})")
    return ax


def plot_multifractal_spectrum(result, ax=None, **kwargs):
    """Singularity spectrum ``f(alpha)`` from an :class:`~fdnkit.mfdfa.MFDFAResult`."""
    from .mfdfa import multifractal_spectrum

    ax = _get_ax(ax)
    alpha, f_alpha = multifractal_spectrum(result)
    ax.plot(alpha, f_alpha, "o-", **kwargs)
    ax.set_xlabel(r"$\alpha$ (Holder exponent)")
    ax.set_ylabel(r"$f(\alpha)$")
    ax.set_title("Multifractal spectrum")
    return ax


def plot_hurst_over_time(times, hurst_values, ax=None, *, label=None, **kwargs):
    """Trace of a scaling exponent (Hurst / alpha) across analysis windows.

    Parameters
    ----------
    times : array-like
        Window centre times (or indices).
    hurst_values : array-like
        One value per window.
    """
    ax = _get_ax(ax)
    ax.plot(np.asarray(times), np.asarray(hurst_values), "-o", label=label, **kwargs)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hurst / exponent")
    ax.set_title("Scaling exponent over time")
    if label:
        ax.legend()
    return ax


def plot_alpha_distribution(alphas, ax=None, *, bins=20, **kwargs):
    """Histogram of per-channel (or per-chunk) fractional orders ``alpha``."""
    ax = _get_ax(ax)
    a = np.asarray(alphas, dtype=float).ravel()
    a = a[np.isfinite(a)]
    ax.hist(a, bins=bins, **kwargs)
    ax.set_xlabel(r"fractional order $\alpha$")
    ax.set_ylabel("count")
    ax.set_title("FODN alpha distribution")
    return ax


def plot_coupling_matrix(coupling, channel_names=None, ax=None, *, cmap="RdBu_r", **kwargs):
    """Heatmap of a FODN coupling matrix ``A``.

    Parameters
    ----------
    coupling : array-like, shape (n, n)
    channel_names : sequence of str, optional
    """
    ax = _get_ax(ax)
    a = np.asarray(coupling, dtype=float)
    vmax = np.max(np.abs(a)) or 1.0
    im = ax.imshow(a, cmap=cmap, vmin=-vmax, vmax=vmax, **kwargs)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coupling")
    ax.set_title("FODN coupling matrix A")
    ax.set_xlabel("source channel")
    ax.set_ylabel("target channel")
    if channel_names is not None:
        ax.set_xticks(range(len(channel_names)))
        ax.set_yticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
        ax.set_yticklabels(channel_names, fontsize=7)
    return ax


def plot_eigenvector_hubs(dominant_eigvec, channel_names=None, ax=None, **kwargs):
    """Bar chart of per-channel hub scores (dominant-eigenvector magnitude)."""
    ax = _get_ax(ax)
    v = np.asarray(dominant_eigvec, dtype=float).ravel()
    idx = np.arange(v.size)
    ax.bar(idx, v, **kwargs)
    ax.set_xlabel("channel")
    ax.set_ylabel("hub score |eigvec|")
    ax.set_title("FODN eigenvector hubs")
    if channel_names is not None:
        ax.set_xticks(idx)
        ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
    return ax
