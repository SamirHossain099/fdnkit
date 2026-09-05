import numpy as np
import pandas as pd

from fdnkit.features import (
    CORE_FEATURES,
    dfa_features,
    extract_features,
    feature_table,
    fodn_features,
    mfdfa_features,
)
from fdnkit.synthetic import synthetic_ieeg


def _demo_signal(seed=0):
    sig, _ = synthetic_ieeg(n_channels=6, n_samples=1500, seed=seed)
    return sig


def test_dfa_features_keys():
    f = dfa_features(_demo_signal())
    assert {"DFA_H_mean", "DFA_H_std", "DFA_H_max", "DFA_H_min"} <= set(f)
    assert all(np.isfinite(v) for v in f.values())


def test_mfdfa_features_keys():
    f = mfdfa_features(_demo_signal())
    assert "MFDFA_Hq_mean" in f and "MFDFA_delta_h_mean" in f


def test_fodn_features_keys():
    f = fodn_features(_demo_signal(), n_iter=2)
    assert "FODN_alpha_mean" in f and "FODN_leading_eig" in f
    assert 0.0 <= f["FODN_sparseness"] <= 1.0


def test_extract_features_has_core_aliases():
    f = extract_features(_demo_signal(), fodn_kwargs={"n_iter": 2})
    for name in CORE_FEATURES:
        assert name in f, f"missing core feature {name}"
        assert np.isfinite(f[name])


def test_extract_features_toggles():
    f = extract_features(_demo_signal(), do_fodn=False, do_mfdfa=False)
    assert any(k.startswith("DFA_") for k in f)
    assert not any(k.startswith("FODN_") for k in f)


def test_feature_table_one_row_per_trial():
    trials = [
        {"trial_id": f"t{i}", "group": f"S{i % 2}", "label": i % 2,
         "signals": _demo_signal(seed=i)}
        for i in range(4)
    ]
    df = feature_table(trials, fodn_kwargs={"n_iter": 2})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df.columns[:3]) == ["trial_id", "group", "label"]
    assert "MeanAlpha" in df.columns


def test_feature_table_missing_signals_raises():
    import pytest

    with pytest.raises(KeyError):
        feature_table([{"trial_id": "x"}])
