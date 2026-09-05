"""FDNkit -- Fractional Dynamical Network & Multifractal toolkit for iEEG.

Turn intracranial-EEG recordings into fractal / fractional-dynamical-network
features and evaluate them honestly:

* :mod:`fdnkit.dfa` -- monofractal DFA Hurst exponent.
* :mod:`fdnkit.mfdfa` -- multifractal generalized Hurst ``h(q)`` and spectrum.
* :mod:`fdnkit.fodn` -- fractional-order dynamical network (alpha, coupling A,
  eigenvector hubs).
* :mod:`fdnkit.features` -- tidy per-trial feature tables.
* :mod:`fdnkit.classify` -- classification with subject-wise CV by default.
* :mod:`fdnkit.io`, :mod:`fdnkit.preprocessing`, :mod:`fdnkit.viz`,
  :mod:`fdnkit.synthetic` -- supporting IO, windowing, plots, and test signals.

Quickstart
----------
>>> from fdnkit.synthetic import synthetic_ieeg
>>> from fdnkit.features import extract_features
>>> sig, names = synthetic_ieeg(n_channels=6, n_samples=2000, seed=0)
>>> feats = extract_features(sig)
>>> sorted(feats)[:3]
['DFA_H_max', 'DFA_H_mean', 'DFA_H_min']
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dfa import DFAResult, dfa, hurst
from .features import (
    CORE_FEATURES,
    dfa_features,
    extract_features,
    feature_table,
    fodn_features,
    mfdfa_features,
)
from .fodn import FODN, FODNResult, fit_fodn
from .mfdfa import (
    MFDFAResult,
    delta_hq,
    generalized_hurst,
    mfdfa,
    multifractal_spectrum,
)
from .preprocessing import flag_bad_channels, segment, sliding_windows, zscore
from .synthetic import binomial_cascade, fbm, fgn, synthetic_ieeg

__all__ = [
    "__version__",
    # dfa
    "dfa",
    "hurst",
    "DFAResult",
    # mfdfa
    "mfdfa",
    "generalized_hurst",
    "delta_hq",
    "multifractal_spectrum",
    "MFDFAResult",
    # fodn
    "FODN",
    "FODNResult",
    "fit_fodn",
    # features
    "extract_features",
    "feature_table",
    "dfa_features",
    "mfdfa_features",
    "fodn_features",
    "CORE_FEATURES",
    # preprocessing
    "zscore",
    "flag_bad_channels",
    "segment",
    "sliding_windows",
    # synthetic
    "fgn",
    "fbm",
    "binomial_cascade",
    "synthetic_ieeg",
]


def __getattr__(name):
    # Lazily expose the classification helpers to keep import light and
    # optional-dependency-safe. Note: ``fdnkit.classify`` is the *module*
    # (call ``fdnkit.classify.classify(...)`` or import the function directly);
    # ``classify_dataframe`` and ``ClassificationResult`` are surfaced here.
    if name in ("classify_dataframe", "ClassificationResult"):
        from . import classify as _classify

        return getattr(_classify, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
