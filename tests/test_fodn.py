import numpy as np
import pytest
import scipy.linalg as LA

from fdnkit.fodn import FODN, FODNResult, HaarWaveletTransform, fit_fodn
from fdnkit.synthetic import synthetic_ieeg


def test_fodn_runs_and_shapes():
    sig, _ = synthetic_ieeg(n_channels=6, n_samples=1000, seed=0)
    res = fit_fodn(sig, n_iter=3)
    assert isinstance(res, FODNResult)
    assert res.alpha.shape == (6,)
    assert res.coupling.shape == (6, 6)
    assert res.eigenvalues.shape == (6,)
    assert res.dominant_eigvec.shape == (6,)


def test_fodn_outputs_finite():
    sig, _ = synthetic_ieeg(n_channels=5, n_samples=800, seed=1)
    res = fit_fodn(sig, n_iter=2)
    assert np.all(np.isfinite(res.alpha))
    assert np.all(np.isfinite(res.coupling))
    assert np.isfinite(res.leading_eig)
    assert 0.0 <= res.sparseness <= 1.0


def test_fodn_deterministic():
    sig, _ = synthetic_ieeg(n_channels=5, n_samples=800, seed=2)
    a = fit_fodn(sig, n_iter=2)
    b = fit_fodn(sig, n_iter=2)
    assert np.allclose(a.coupling, b.coupling)
    assert np.allclose(a.alpha, b.alpha)


def test_fodn_rejects_single_channel():
    with pytest.raises(ValueError):
        FODN().fit(np.random.default_rng(0).standard_normal((1, 500)))


def test_fodn_rejects_more_channels_than_samples():
    with pytest.raises(ValueError):
        FODN().fit(np.random.default_rng(0).standard_normal((50, 10)))


def test_result_before_fit_raises():
    with pytest.raises(RuntimeError):
        FODN().result()


def test_haar_transform_shapes():
    x = np.random.default_rng(0).standard_normal(256)
    wt = HaarWaveletTransform(x)
    wt.normalize()
    approx, detail = wt.transform()
    assert approx.shape == detail.shape
    assert approx.shape[0] == 128


def test_haar_rejects_2d():
    with pytest.raises(ValueError):
        HaarWaveletTransform(np.ones((4, 4)))


def _serial_lasso(model, b_vec, lambda_):
    """The ADMM-LASSO step as the validated source wrote it: one residual vector
    at a time. Kept here verbatim as the oracle for the batched solver."""
    a = model._b
    b_vec = np.reshape(b_vec, (np.size(b_vec), 1))
    max_iter, abstol, reltol = 100, 1e-4, 1e-2
    m, n = np.shape(a)
    atb = np.matmul(a.T, b_vec)
    rho = 1 / lambda_
    alpha = 1.0
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))
    l_inv, u_inv = model._pre.l_inv, model._pre.u_inv
    for _ in range(max_iter):
        q = atb + rho * (z - u)
        x = np.matmul(u_inv, np.matmul(l_inv, q))
        z_old = np.array(z)
        x_hat = alpha * x + (1 - alpha) * z_old
        z = FODN._shrinkage(x_hat + u, lambda_ / rho)
        u += x_hat - z
        r_norm = LA.norm(x - z)
        s_norm = LA.norm(-rho * (z - z_old))
        eps_pri = np.sqrt(n) * abstol + reltol * np.max((LA.norm(x), LA.norm(-z)))
        eps_dual = np.sqrt(n) * abstol + reltol * LA.norm(rho * u)
        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return np.squeeze(z)


def _fitted_model_and_residuals(n_channels=8, n_samples=1200, seed=5):
    sig, _ = synthetic_ieeg(n_channels=n_channels, n_samples=n_samples, seed=seed)
    model = FODN(n_iter=1).fit(sig)
    residual = model._z[:, 1:] - np.matmul(model._a_hist[0], sig[:, :-1])
    # Rescale blocks of columns so the batch mixes problems that converge in a
    # step or two with ones that run to the iteration cap. The batched solver
    # must freeze each column at its own stopping point, not at the slowest one.
    residual = residual.copy()
    residual[:, ::3] *= 1e3
    residual[:, 1::3] *= 1e-3
    return model, residual


def test_batched_lasso_matches_serial_column_by_column():
    model, residual = _fitted_model_and_residuals()
    batched = model._lasso(residual, model.lambda_)
    assert batched.shape == (model.num_inputs, residual.shape[1])
    serial = np.column_stack(
        [_serial_lasso(model, residual[:, k], model.lambda_) for k in range(residual.shape[1])]
    )
    # Both solvers take the same number of ADMM steps on every column, so the
    # only residue is rounding, which grows with the column's magnitude over
    # the 100-step cap. A mismatched step count would differ at the 1e-2
    # relative stopping tolerance, eight orders of magnitude above this bound.
    scale = np.maximum(1.0, np.abs(serial).max(axis=0))
    assert np.all(np.abs(batched - serial) <= 1e-12 * scale)
    assert np.any(np.abs(serial) > 0)


def test_lasso_single_vector_matches_its_batch_column():
    model, residual = _fitted_model_and_residuals(n_channels=6, n_samples=600, seed=6)
    batched = model._lasso(residual, model.lambda_)
    for k in (0, 1, 2, residual.shape[1] - 1):
        one = model._lasso(residual[:, k], model.lambda_)
        assert one.shape == (model.num_inputs,)
        scale = max(1.0, float(np.abs(one).max()))
        assert np.all(np.abs(one - batched[:, k]) <= 1e-12 * scale)
