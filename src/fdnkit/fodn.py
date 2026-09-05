"""Fractional-Order Dynamical Network (FODN) model.

Ports the FODN estimator from the original reference implementation into a
clean, documented, sklearn-style estimator. The numerical procedure is preserved faithfully:

1. **Fractional order per channel** (:math:`\\alpha_i`) is estimated from the
   variance decay of a Haar wavelet transform across dyadic scales.
2. **Grunwald-Letnikov fractional differencing** builds the fractional-derivative
   signal ``z`` for each channel from its ``alpha``.
3. **Coupling matrix** ``A`` is fit by regularized least squares
   (``z_k ≈ A x_{k-1}``), then refined by an ADMM-LASSO unknown-input step that
   promotes a sparse directed network.

The model underlies the "fractional dynamical network" features in Beeram et al.
(2026, *Front. Netw. Physiol.* 6:1768476); the underlying method is due to
Gupta, Pequito & Bogdan (2018) and Xue & Bogdan (2017).

The heavy inner loop is unchanged from the validated source; the public surface
(:class:`FODN`, :func:`fit_fodn`, :class:`FODNResult`) is new.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as LA
from scipy.special import gamma

__all__ = ["HaarWaveletTransform", "FODN", "FODNResult", "fit_fodn"]


class HaarWaveletTransform:
    """Fast in-place Haar wavelet transform of a 1-D signal.

    Used by the FODN model to estimate a channel's fractional order from the
    variance of detail coefficients across dyadic scales.
    """

    def __init__(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim > 1:
            x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError("HaarWaveletTransform accepts only 1-D signals")
        self.x = x
        self._n = x.size

    def normalize(self):
        """Subtract the mean in place."""
        self.x = self.x - np.mean(self.x)

    @staticmethod
    def _dwt_haar(signal):
        n_use = int(np.floor(signal.size / 2))
        c = (signal[: 2 * n_use : 2] + signal[1 : 2 * n_use : 2]) / 2
        s = signal[: 2 * n_use : 2] - c
        c = 2 * c / np.sqrt(2)
        s = -2 * s / np.sqrt(2)
        return c, s

    def transform(self):
        """Return approximation (``W``) and detail (``D``) coefficient tables."""
        n_by2 = int(np.floor(self._n / 2))
        approx = np.zeros((n_by2, n_by2))
        detail = np.zeros((n_by2, n_by2))
        j = self._n
        signal = self.x
        for i in range(int(np.floor(np.log2(self._n)))):
            j = int(np.floor(j / 2))
            w, d = self._dwt_haar(signal)
            approx[i, :j] = w
            detail[i, :j] = d
            signal = w
        return approx, detail


@dataclass
class FODNResult:
    """Output of a fitted :class:`FODN` model.

    Attributes
    ----------
    alpha : numpy.ndarray
        Per-channel fractional orders (length ``n_channels``).
    coupling : numpy.ndarray
        Estimated directed coupling matrix ``A`` (``n_channels x n_channels``).
    eigenvalues : numpy.ndarray
        Eigenvalues of ``A`` (complex).
    leading_eig : float
        Spectral radius: ``max |eigenvalue|`` (network gain / stability proxy).
    dominant_eigvec : numpy.ndarray
        Magnitude of the eigenvector for the largest-real-part eigenvalue;
        a per-channel "hub" score.
    sparseness : float
        Fraction of ``|A|`` entries exceeding ``1e-2`` (network density).
    """

    alpha: np.ndarray
    coupling: np.ndarray
    eigenvalues: np.ndarray
    leading_eig: float
    dominant_eigvec: np.ndarray
    sparseness: float


class FODN:
    """Fractional-Order Dynamical Network estimator.

    Parameters
    ----------
    num_inputs : int, optional
        Number of unknown inputs for the LASSO step. Defaults to
        ``floor(n_channels / 2)``.
    num_fract : int
        Truncation length of the Grunwald-Letnikov fractional-difference kernel.
    n_iter : int
        Number of ADMM refinement iterations.
    lambda_ : float
        LASSO sparsity weight for the unknown-input estimate.
    verbose : bool
        Print per-iteration MSE and timing.

    Attributes (after :meth:`fit`)
    ------------------------------
    alpha_ : numpy.ndarray
        Per-channel fractional orders.
    coupling_ : numpy.ndarray
        Final coupling matrix ``A``.
    """

    def __init__(self, num_inputs=None, num_fract=50, n_iter=10, lambda_=0.5, verbose=False):
        self.num_inputs = num_inputs
        self.num_fract = num_fract
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.verbose = verbose

        # populated during fit
        self._n_ch = None
        self._k = None
        self._order = None
        self._z = None
        self._b = None
        self._a_hist = None
        self._u = None
        self._pre = None
        self.alpha_ = None
        self.coupling_ = None

    # ---- fractional order estimation ---------------------------------------
    def _fractional_order(self, x):
        num_scales = int(np.floor(np.log2(self._k)))
        log_scales = np.zeros(num_scales)
        scale = np.arange(1, num_scales + 1)

        wt = HaarWaveletTransform(x)
        wt.normalize()
        _, detail = wt.transform()
        j = int(np.floor(self._k / 2))
        for i in range(num_scales - 1):
            y = detail[i, :j]
            variance = np.var(y, ddof=1)
            if variance <= 0:  # guard log2(0)
                variance = 1e-10
            log_scales[i] = np.log2(variance)
            j = int(np.floor(j / 2))
        p = np.polyfit(scale[: num_scales - 1], log_scales[: num_scales - 1], 1)
        return p[0] / 2

    def _estimate_order(self, x):
        self._order = np.array([self._fractional_order(x[i, :]) for i in range(self._n_ch)])

    def _update_z(self, x):
        self._z = np.empty((self._n_ch, self._k))
        j = np.arange(0, self.num_fract + 1)
        for i in range(self._n_ch):
            prefactor = gamma(-self._order[i] + j) / gamma(-self._order[i]) / gamma(j + 1)
            y = np.convolve(x[i, :], prefactor)
            self._z[i, :] = y[: self._k]

    # ---- coupling matrix ----------------------------------------------------
    def _heuristic_b(self, a):
        b = np.zeros((self._n_ch, self._n_ch))
        b[np.abs(a) > 0.01] = a[np.abs(a) > 0.01]
        _, r = LA.qr(b)
        col_ind = np.where(np.abs(np.diag(r)) > 1e-7)
        if np.size(col_ind[0]) < self.num_inputs:
            self._b = np.vstack(
                (np.eye(self.num_inputs), np.zeros((self._n_ch - self.num_inputs, self.num_inputs)))
            )
        else:
            col_ind = col_ind[0][: self.num_inputs]
            self._b = b[:, col_ind]
        if np.linalg.matrix_rank(b) < self.num_inputs:
            # fall back to a well-conditioned selector instead of failing
            self._b = np.vstack(
                (np.eye(self.num_inputs), np.zeros((self._n_ch - self.num_inputs, self.num_inputs)))
            )

    def _least_squares(self, y, x):
        x_use = np.vstack((np.zeros((1, self._n_ch)), x[:-1, :]))
        reg = 1e-8 * np.eye(x_use.shape[1])  # avoid singular normal equations
        a = np.matmul(np.matmul(y.T, x_use), LA.inv(np.matmul(x_use.T, x_use) + reg))
        mse = LA.norm(y - np.matmul(x_use, a.T), axis=0) ** 2 / self._k
        return a, np.mean(mse)

    @staticmethod
    def _factor(a, rho):
        m, n = np.shape(a)
        if m >= n:
            lower = LA.cholesky(np.matmul(a.T, a) + rho * np.eye(n), lower=True)
        else:
            lower = LA.cholesky(np.eye(m) + 1 / rho * np.matmul(a, a.T), lower=True)
        return lower, lower.T

    @staticmethod
    def _shrinkage(x, kappa):
        return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)

    class _PreComputed:
        def __init__(self, b, rho):
            self.l, self.u = FODN._factor(b, rho)
            self.l_inv = LA.inv(self.l)
            self.u_inv = LA.inv(self.u)

    def _lasso(self, b_vec, lambda_):
        a = self._b
        b_vec = np.reshape(b_vec, (np.size(b_vec), 1))
        max_iter, abstol, reltol = 100, 1e-4, 1e-2
        m, n = np.shape(a)
        atb = np.matmul(a.T, b_vec)
        rho = 1 / lambda_
        alpha = 1.0

        z = np.zeros((n, 1))
        u = np.zeros((n, 1))
        l_inv, u_inv = self._pre.l_inv, self._pre.u_inv

        for _ in range(max_iter):
            q = atb + rho * (z - u)
            if m >= n:
                x = np.matmul(u_inv, np.matmul(l_inv, q))
            else:
                x = q / rho - np.matmul(
                    a.T, np.matmul(LA.inv(u_inv), np.matmul(LA.inv(l_inv), np.matmul(a, q)))
                ) / rho**2

            z_old = np.array(z)
            x_hat = alpha * x + (1 - alpha) * z_old
            z = self._shrinkage(x_hat + u, lambda_ / rho)
            u += x_hat - z

            r_norm = LA.norm(x - z)
            s_norm = LA.norm(-rho * (z - z_old))
            eps_pri = np.sqrt(n) * abstol + reltol * np.max((LA.norm(x), LA.norm(-z)))
            eps_dual = np.sqrt(n) * abstol + reltol * LA.norm(rho * u)
            if r_norm < eps_pri and s_norm < eps_dual:
                break
        return np.squeeze(z)

    # ---- public API ---------------------------------------------------------
    def fit(self, x):
        """Fit the FODN model to a ``(n_channels, n_timepoints)`` array.

        Returns ``self``; populates :attr:`alpha_` and :attr:`coupling_`.
        """
        import time

        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D (n_channels, n_timepoints)")
        self._n_ch, self._k = x.shape
        if self._n_ch < 2:
            raise ValueError("FODN needs more than one channel")
        if self._k < self._n_ch:
            raise ValueError("number of timepoints must be >= number of channels")
        if self.num_inputs is None:
            self.num_inputs = int(np.floor(self._n_ch / 2))
        self.num_inputs = max(1, int(self.num_inputs))

        self._a_hist = np.empty((self.n_iter + 1, self._n_ch, self._n_ch))
        self._u = np.zeros((self.num_inputs, self._k))

        self._estimate_order(x)
        self._update_z(x)
        self._a_hist[0], mse = self._least_squares(self._z.T, x.T)
        self._heuristic_b(self._a_hist[0])
        self._pre = self._PreComputed(self._b, 1 / self.lambda_)

        t0 = time.time()
        if self.verbose:
            print(f"beginning mse = {mse:.6f}")
        for it in range(self.n_iter):
            for k in range(1, self._k):
                residual = self._z[:, k] - np.matmul(self._a_hist[it], x[:, k - 1])
                self._u[:, k] = self._lasso(residual, self.lambda_)
            self._a_hist[it + 1], mse = self._least_squares(
                (self._z - np.matmul(self._b, self._u)).T, x.T
            )
            if self.verbose:
                print(f"iter {it}: mse = {mse:.6f}")
        if self.verbose:
            print(f"time taken = {time.time() - t0:.3f}s")

        self.alpha_ = self._order
        self.coupling_ = self._a_hist[-1]
        return self

    def result(self) -> FODNResult:
        """Assemble a :class:`FODNResult` (eigen-decomposition + summaries)."""
        if self.coupling_ is None:
            raise RuntimeError("call fit() before result()")
        a = self.coupling_
        w, v = np.linalg.eig(a)
        leading = float(np.max(np.abs(w)))
        dom = np.abs(v[:, int(np.argmax(w.real))])
        sparseness = float(np.count_nonzero(np.abs(a) > 0.01) / a.size)
        return FODNResult(
            alpha=self.alpha_,
            coupling=a,
            eigenvalues=w,
            leading_eig=leading,
            dominant_eigvec=dom,
            sparseness=sparseness,
        )


def fit_fodn(x, *, num_inputs=None, num_fract=50, n_iter=10, lambda_=0.5, verbose=False) -> FODNResult:
    """Functional wrapper: fit a :class:`FODN` and return its :class:`FODNResult`.

    Parameters
    ----------
    x : array-like, shape (n_channels, n_timepoints)
        Multi-channel signal segment.
    num_inputs, num_fract, n_iter, lambda_, verbose
        Passed through to :class:`FODN`.
    """
    model = FODN(
        num_inputs=num_inputs,
        num_fract=num_fract,
        n_iter=n_iter,
        lambda_=lambda_,
        verbose=verbose,
    ).fit(x)
    return model.result()
