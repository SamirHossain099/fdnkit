import warnings

import numpy as np
import pandas as pd
import pytest

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


def test_channel_with_nan_is_skipped_with_a_warning_not_silently():
    # dfa and mfdfa now reject nan, so the per-channel summaries must say
    # which channel they left out, and the rest of the trial must still be
    # summarised exactly as if that channel had not been supplied.
    sig = _demo_signal()
    bad = sig.copy()
    bad[2, 400] = np.nan
    clean = np.delete(sig, 2, axis=0)
    with pytest.warns(RuntimeWarning, match=r"dfa_features: skipping channel\(s\) \[2\]"):
        f_bad = dfa_features(bad)
    assert f_bad == dfa_features(clean)
    with pytest.warns(RuntimeWarning, match=r"mfdfa_features: skipping channel\(s\) \[2\]"):
        f_bad = mfdfa_features(bad)
    assert f_bad == mfdfa_features(clean)


def test_all_channels_non_finite_returns_empty_features():
    sig = np.full((3, 1500), np.nan)
    with pytest.warns(RuntimeWarning, match="remaining 0 of 3"):
        assert dfa_features(sig) == {}
    with pytest.warns(RuntimeWarning, match="remaining 0 of 3"):
        assert mfdfa_features(sig) == {}


def test_finite_signals_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dfa_features(_demo_signal())
        mfdfa_features(_demo_signal())
