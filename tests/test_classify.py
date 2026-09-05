import numpy as np
import pandas as pd
import pytest

from fdnkit.classify import classify, classify_dataframe, make_classifier


def _make_dataset(n_subjects=6, per_subject=12, sep=2.5, seed=0):
    """Two-class data with a real, subject-independent signal (separable)."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(n_subjects):
        for _ in range(per_subject):
            label = rng.integers(0, 2)
            center = sep if label else -sep
            X.append(rng.standard_normal(4) + center)
            y.append(label)
            groups.append(f"S{s}")
    return np.array(X), np.array(y), np.array(groups)


def test_loso_requires_groups():
    X, y, _ = _make_dataset()
    with pytest.raises(ValueError):
        classify(X, y, cv="loso")


def test_loso_detects_real_signal():
    X, y, g = _make_dataset(sep=2.5)
    res = classify(X, y, groups=g, cv="loso", n_permutations=200)
    assert res.honest
    assert res.balanced_accuracy > 0.8
    assert res.permutation_p is not None and res.permutation_p < 0.05
    assert res.ci95 is not None


def test_loso_null_signal_not_significant():
    X, y, g = _make_dataset(sep=0.0, seed=1)  # no class separation
    res = classify(X, y, groups=g, cv="loso", n_permutations=300)
    # With no signal, honest CV should sit near chance and not be significant.
    assert res.permutation_p > 0.01


def test_loo_is_flagged_optimistic():
    X, y, _ = _make_dataset()
    res = classify(X, y, cv="loo", permutation=False)
    assert not res.honest
    assert "leakage" in res.notes.lower()


def test_group_kfold_runs():
    X, y, g = _make_dataset()
    res = classify(X, y, groups=g, cv="group_kfold", n_splits=3, n_permutations=100)
    assert res.honest
    assert np.isfinite(res.balanced_accuracy)


def test_per_group_reported():
    X, y, g = _make_dataset()
    res = classify(X, y, groups=g, cv="loso", permutation=False)
    assert len(res.per_group) == len(np.unique(g))


def test_summary_is_string():
    X, y, g = _make_dataset()
    res = classify(X, y, groups=g, cv="loso", permutation=False, bootstrap=False)
    assert isinstance(res.summary(), str)
    assert "balanced accuracy" in res.summary()


def test_classify_dataframe():
    X, y, g = _make_dataset()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df["group"] = g
    res = classify_dataframe(df, cv="loso", permutation=False)
    assert res.n == len(y)


def test_single_class_rejected():
    X = np.random.default_rng(0).standard_normal((10, 3))
    with pytest.raises(ValueError):
        classify(X, np.zeros(10), groups=np.arange(10))


def test_make_classifier_is_fittable():
    clf = make_classifier()
    X = np.random.default_rng(0).standard_normal((20, 3))
    y = (X[:, 0] > 0).astype(int)
    clf.fit(X, y)
    assert clf.predict(X).shape == (20,)
