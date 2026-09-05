"""Assemble tidy per-trial / per-segment feature tables.

Combines the DFA, MFDFA, and FODN analyses into flat dictionaries and pandas
DataFrames (one row per trial or segment). This is the port and generalization
of the original feature extractor -- but computed directly from
arrays rather than by scraping a directory of CSVs.

The five "core" features reproduce the set used in Beeram et al. (2026):
``MeanAlpha``, ``VarAlpha``, ``LeadingEig``, ``MF_DFA_H``, ``MF_DFA_Hq_mean``.
:func:`extract_features` additionally returns a richer set (multifractal width,
sparseness, hub concentration, ...) that callers can opt into.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dfa import dfa
from .fodn import fit_fodn
from .mfdfa import mfdfa

__all__ = [
    "CORE_FEATURES",
    "dfa_features",
    "mfdfa_features",
    "fodn_features",
    "extract_features",
    "feature_table",
]

CORE_FEATURES = ["MeanAlpha", "VarAlpha", "LeadingEig", "MF_DFA_H", "MF_DFA_Hq_mean"]


def _clean_channels(signals):
    x = np.asarray(signals, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("signals must be 1-D or 2-D (n_channels, n_samples)")
    return x


def dfa_features(signals, scales=None, order: int = 1, prefix: str = "DFA") -> dict:
    """Per-channel DFA Hurst, summarized across channels.

    Returns ``{<prefix>_H_mean, <prefix>_H_std, <prefix>_H_max, <prefix>_H_min}``.
    """
    x = _clean_channels(signals)
    h = np.array([dfa(x[i], scales=scales, order=order).hurst for i in range(x.shape[0])])
    h = h[np.isfinite(h)]
    if h.size == 0:
        return {}
    return {
        f"{prefix}_H_mean": float(h.mean()),
        f"{prefix}_H_std": float(h.std()),
        f"{prefix}_H_max": float(h.max()),
        f"{prefix}_H_min": float(h.min()),
    }


def mfdfa_features(signals, scales=None, q=None, order: int = 1, prefix: str = "MFDFA") -> dict:
    """Per-channel MFDFA, summarized across channels.

    Returns generalized-Hurst mean/std, mean multifractal width ``delta_h``, and
    the mean ``h(q)`` over all channels and moments.
    """
    x = _clean_channels(signals)
    hq_means, deltas, hq_all = [], [], []
    for i in range(x.shape[0]):
        res = mfdfa(x[i], scales=scales, q=q, order=order)
        finite = res.hq[np.isfinite(res.hq)]
        if finite.size == 0:
            continue
        hq_means.append(float(finite.mean()))
        deltas.append(res.delta_h)
        hq_all.append(finite)
    if not hq_means:
        return {}
    hq_all = np.concatenate(hq_all)
    return {
        f"{prefix}_Hq_mean": float(np.mean(hq_means)),
        f"{prefix}_Hq_std": float(np.std(hq_means)),
        f"{prefix}_delta_h_mean": float(np.nanmean(deltas)),
        f"{prefix}_Hq_grand_mean": float(hq_all.mean()),
    }


def fodn_features(signals, *, n_iter: int = 10, lambda_: float = 0.5, num_fract: int = 50,
                  top_k: int = 3, prefix: str = "FODN", **fodn_kwargs) -> dict:
    """FODN network features for a multi-channel segment.

    Returns fractional-order (alpha) mean/std, leading eigenvalue, network
    sparseness, and the summed hub score of the top-``k`` channels.
    """
    x = _clean_channels(signals)
    if x.shape[0] < 2:
        return {}
    res = fit_fodn(x, n_iter=n_iter, lambda_=lambda_, num_fract=num_fract, **fodn_kwargs)
    alpha = res.alpha[np.isfinite(res.alpha)]
    if alpha.size == 0:
        return {}
    hub = res.dominant_eigvec
    hub = hub / (hub.sum() + 1e-12)
    k = max(1, min(top_k, hub.size))
    top_hub = float(np.sort(hub)[-k:].sum())
    return {
        f"{prefix}_alpha_mean": float(alpha.mean()),
        f"{prefix}_alpha_std": float(alpha.std()),
        f"{prefix}_alpha_var": float(alpha.var()),
        f"{prefix}_leading_eig": float(res.leading_eig),
        f"{prefix}_sparseness": float(res.sparseness),
        f"{prefix}_hub_top{k}": top_hub,
    }


def extract_features(
    signals,
    *,
    do_dfa: bool = True,
    do_mfdfa: bool = True,
    do_fodn: bool = True,
    scales=None,
    q=None,
    order: int = 1,
    fodn_kwargs: dict | None = None,
    include_core_aliases: bool = True,
) -> dict:
    """Compute a single tidy feature row for one multi-channel segment.

    Parameters
    ----------
    signals : array-like, shape (n_channels, n_samples)
    do_dfa, do_mfdfa, do_fodn : bool
        Toggle each analysis family.
    scales, q, order : see the analysis modules.
    fodn_kwargs : dict, optional
        Extra keyword arguments for :func:`fdnkit.fodn.fit_fodn`.
    include_core_aliases : bool
        Also emit the five canonical column names in :data:`CORE_FEATURES`
        (``MeanAlpha`` etc.) so results line up with the reference study.

    Returns
    -------
    dict
        Feature name -> value.
    """
    feats: dict = {}
    if do_dfa:
        feats.update(dfa_features(signals, scales=scales, order=order))
    if do_mfdfa:
        feats.update(mfdfa_features(signals, scales=scales, q=q, order=order))
    if do_fodn:
        feats.update(fodn_features(signals, **(fodn_kwargs or {})))

    if include_core_aliases:
        alias = {}
        if "FODN_alpha_mean" in feats:
            alias["MeanAlpha"] = feats["FODN_alpha_mean"]
        if "FODN_alpha_var" in feats:
            alias["VarAlpha"] = feats["FODN_alpha_var"]
        if "FODN_leading_eig" in feats:
            alias["LeadingEig"] = feats["FODN_leading_eig"]
        if "DFA_H_mean" in feats:
            alias["MF_DFA_H"] = feats["DFA_H_mean"]
        if "MFDFA_Hq_grand_mean" in feats:
            alias["MF_DFA_Hq_mean"] = feats["MFDFA_Hq_grand_mean"]
        feats.update(alias)
    return feats


def feature_table(trials, *, id_key="trial_id", group_key="group", label_key="label",
                  progress: bool = False, **extract_kwargs) -> pd.DataFrame:
    """Build a per-trial feature DataFrame from an iterable of trial records.

    Parameters
    ----------
    trials : iterable of dict
        Each record must have a ``"signals"`` array of shape
        ``(n_channels, n_samples)`` and may carry ``trial_id``, ``group``
        (e.g. subject id, for honest CV), and ``label``.
    id_key, group_key, label_key : str
        Keys copied through to identifier columns when present.
    progress : bool
        Print a short progress line per trial.
    **extract_kwargs
        Forwarded to :func:`extract_features`.

    Returns
    -------
    pandas.DataFrame
        One row per trial; identifier columns first, then features.
    """
    rows = []
    trials = list(trials)
    for i, rec in enumerate(trials):
        if "signals" not in rec:
            raise KeyError("each trial record needs a 'signals' array")
        if progress:
            print(f"[fdnkit] features {i + 1}/{len(trials)}", flush=True)
        feats = extract_features(rec["signals"], **extract_kwargs)
        row = {}
        for key, col in ((id_key, id_key), (group_key, group_key), (label_key, label_key)):
            if key in rec:
                row[col] = rec[key]
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    id_cols = [c for c in (id_key, group_key, label_key) if c in df.columns]
    other = [c for c in df.columns if c not in id_cols]
    return df[id_cols + other]
