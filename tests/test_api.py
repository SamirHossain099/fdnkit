import fdnkit


def test_version_exposed():
    assert isinstance(fdnkit.__version__, str)


def test_public_api_importable():
    for name in [
        "dfa", "hurst", "mfdfa", "generalized_hurst", "delta_hq",
        "fit_fodn", "FODN", "extract_features", "feature_table",
        "fgn", "fbm", "binomial_cascade", "synthetic_ieeg", "zscore",
    ]:
        assert hasattr(fdnkit, name), f"fdnkit.{name} missing"


def test_lazy_classify_access():
    # classify_dataframe / ClassificationResult are exposed lazily via __getattr__;
    # fdnkit.classify is the submodule (call fdnkit.classify.classify(...)).
    assert callable(fdnkit.classify_dataframe)
    assert isinstance(fdnkit.ClassificationResult, type)
    from fdnkit.classify import classify as classify_fn
    assert callable(classify_fn)


def test_quickstart_docstring_example():
    from fdnkit.features import extract_features
    from fdnkit.synthetic import synthetic_ieeg

    sig, _ = synthetic_ieeg(n_channels=6, n_samples=2000, seed=0)
    feats = extract_features(sig)
    assert sorted(feats)[:3] == ["DFA_H_max", "DFA_H_mean", "DFA_H_min"]
