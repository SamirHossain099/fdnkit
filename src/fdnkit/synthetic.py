"""Synthetic signal generators for testing, examples, and quickstarts.

These let FDNkit's tutorials and test suite run with **no data download** and give
analyses ground-truth answers to check against:

* :func:`fgn` / :func:`fbm` -- fractional Gaussian noise / motion with a known
  Hurst exponent (exact Davies-Harte circulant-embedding synthesis). Detrended
  fluctuation analysis of ``fgn(H)`` should recover ``H``.
* :func:`binomial_cascade` -- a multiplicative cascade whose multifractal
  spectrum is known in closed form; used to check that MFDFA reports a genuinely
  broad ``h(q)``.
* :func:`synthetic_ieeg` -- a small multi-channel, sparsely-coupled recording
  that stands in for an intracranial-EEG trial in examples and FODN tests.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fgn", "fbm", "binomial_cascade", "synthetic_ieeg"]


def _as_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def fgn(n: int, hurst: float = 0.7, *, seed=None) -> np.ndarray:
    """Exact fractional Gaussian noise via Davies-Harte circulant embedding.

    Parameters
    ----------
    n : int
        Number of samples.
    hurst : float
        Target Hurst exponent in (0, 1). ``0.5`` gives ordinary white noise.
    seed : int | numpy.random.Generator | None
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Length-``n`` unit-variance fractional Gaussian noise. Its DFA exponent is
        (asymptotically) ``hurst``.

    Notes
    -----
    Davies, R. B. & Harte, D. S. (1987). *Tests for Hurst effect.* Biometrika 74.
    The circulant embedding is exact when all embedded eigenvalues are
    non-negative; for the fGn autocovariance and ``0 < H < 1`` this holds.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be in the open interval (0, 1)")
    rng = _as_rng(seed)
    H = float(hurst)

    # fGn autocovariance gamma(k) for unit-variance increments.
    k = np.arange(n)
    gamma = 0.5 * (
        np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H)
    )

    # First row of the size-M = 2(n-1) circulant embedding.
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    m = row.size  # 2*(n-1)
    eig = np.fft.fft(row).real
    # Clip tiny negative eigenvalues from floating error; warn only if large.
    if np.any(eig < -1e-8 * np.abs(eig).max()):
        eig = np.clip(eig, 0.0, None)
    else:
        eig = np.clip(eig, 0.0, None)

    # Build spectral coefficients W with the exact Davies-Harte weighting.
    w = np.zeros(m, dtype=complex)
    half = m // 2
    v1 = rng.standard_normal(m)
    v2 = rng.standard_normal(m)
    w[0] = np.sqrt(eig[0]) * v1[0]
    w[half] = np.sqrt(eig[half]) * v1[half]
    idx = np.arange(1, half)
    w[idx] = np.sqrt(eig[idx] / 2.0) * (v1[idx] + 1j * v2[idx])
    w[m - idx] = np.conj(w[idx])

    y = np.fft.fft(w) / np.sqrt(m)
    return y.real[:n]


def fbm(n: int, hurst: float = 0.7, *, seed=None) -> np.ndarray:
    """Fractional Brownian motion: the cumulative sum of :func:`fgn`.

    Returns a length-``n`` path with Hurst exponent ``hurst`` (DFA exponent
    ``hurst + 1``).
    """
    return np.cumsum(fgn(n, hurst, seed=seed))


def binomial_cascade(n_levels: int = 12, p: float = 0.3, *, seed=None) -> np.ndarray:
    """Deterministic-weight binomial multiplicative cascade (a multifractal).

    Builds a measure on ``2**n_levels`` points by repeatedly splitting mass with
    multipliers ``p`` and ``1 - p``. The result is strongly multifractal, so
    MFDFA should return a wide ``h(q)`` (large ``delta_h``). With ``p = 0.5`` the
    cascade is uniform (monofractal).

    Parameters
    ----------
    n_levels : int
        Number of cascade levels; output length is ``2 ** n_levels``.
    p : float
        Multiplier in (0, 1). Distance of ``p`` from 0.5 sets multifractal width.
    seed : int | Generator | None
        Randomises which child receives ``p`` at each split (order only).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    rng = _as_rng(seed)
    measure = np.array([1.0])
    for _ in range(n_levels):
        left = rng.random(measure.size) < 0.5
        m_left = np.where(left, p, 1 - p)
        m_right = 1.0 - m_left
        nxt = np.empty(measure.size * 2)
        nxt[0::2] = measure * m_left
        nxt[1::2] = measure * m_right
        measure = nxt
    # Return as increments of the cumulative measure (a fluctuating series).
    return measure * measure.size


def synthetic_ieeg(
    n_channels: int = 8,
    n_samples: int = 5000,
    fs: float = 1000.0,
    *,
    hurst: float = 0.7,
    coupling_density: float = 0.25,
    coupling_strength: float = 0.35,
    noise: float = 0.1,
    seed=None,
) -> tuple[np.ndarray, list[str]]:
    """A small, sparsely-coupled multi-channel recording resembling iEEG.

    Each channel starts as long-range-correlated fractional Gaussian noise, then
    a sparse random coupling matrix mixes a fraction of each channel's past into
    its neighbours. This yields data with (a) non-trivial Hurst exponents and
    (b) a recoverable directed coupling structure -- suitable for exercising the
    DFA, MFDFA, and FODN paths in examples and tests.

    Returns
    -------
    signals : numpy.ndarray
        Array of shape ``(n_channels, n_samples)``.
    channel_names : list of str
        Names like ``["CH1", "CH2", ...]``.
    """
    rng = _as_rng(seed)
    base = np.vstack([fgn(n_samples, hurst, seed=rng) for _ in range(n_channels)])

    # Sparse random coupling matrix A (off-diagonal drives).
    A = np.zeros((n_channels, n_channels))
    mask = rng.random((n_channels, n_channels)) < coupling_density
    np.fill_diagonal(mask, False)
    A[mask] = coupling_strength * rng.standard_normal(mask.sum())

    signals = base.copy()
    for t in range(1, n_samples):
        signals[:, t] += A @ signals[:, t - 1] + noise * rng.standard_normal(n_channels)

    names = [f"CH{i + 1}" for i in range(n_channels)]
    return signals, names
