import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np  # noqa: E402

from fdnkit import viz  # noqa: E402
from fdnkit.dfa import dfa  # noqa: E402
from fdnkit.fodn import fit_fodn  # noqa: E402
from fdnkit.mfdfa import mfdfa  # noqa: E402
from fdnkit.synthetic import binomial_cascade, fgn, synthetic_ieeg  # noqa: E402


def test_plot_fluctuation():
    ax = viz.plot_fluctuation(dfa(fgn(4096, 0.7, seed=0)))
    assert ax.has_data()


def test_plot_hq_and_spectrum():
    res = mfdfa(binomial_cascade(13, 0.3, seed=0))
    assert viz.plot_hq(res).has_data()
    assert viz.plot_multifractal_spectrum(res).has_data()


def test_plot_hurst_over_time():
    ax = viz.plot_hurst_over_time(np.arange(5), np.linspace(0.6, 0.8, 5), label="ch1")
    assert ax.has_data()


def test_fodn_plots():
    sig, names = synthetic_ieeg(n_channels=5, n_samples=800, seed=0)
    res = fit_fodn(sig, n_iter=2)
    assert viz.plot_alpha_distribution(res.alpha).has_data()
    assert viz.plot_coupling_matrix(res.coupling, names).has_data()
    assert viz.plot_eigenvector_hubs(res.dominant_eigvec, names).has_data()
