# API reference

All functions have NumPy-style docstrings; this page is a map of the public
surface. In a Python session, `help(fdnkit.mfdfa.mfdfa)` gives full detail.

## `fdnkit.dfa`

- `dfa(signal, scales=None, order=1) -> DFAResult`: Hurst exponent via DFA.
- `hurst(signal, scales=None, order=1) -> float`: shorthand for `dfa(...).hurst`.
- `DFAResult`: fields: `hurst`, `scales`, `fluct`.

## `fdnkit.mfdfa`

- `mfdfa(signal, scales=None, q=None, order=1, rel_floor=1e-3, check_flat=True) -> MFDFAResult`:
  multifractal DFA; warns on constant runs when negative `q` is requested.
- `generalized_hurst(signal, ...) -> (q, hq)`: arrays only.
- `delta_hq(signal, ...) -> float`: multifractal width.
- `multifractal_spectrum(result) -> (alpha, f_alpha)`: Legendre transform.
- `MFDFAResult`: fields: `hurst`, `hq`, `q`, `scales`, `fluct`, `fluct_q`;
  properties/methods: `delta_h`, `hq_at(q)`.

## `fdnkit.fodn`

- `fit_fodn(x, *, num_inputs=None, num_fract=50, n_iter=10, lambda_=0.5) -> FODNResult`:
  fit and summarize a fractional-order dynamical network for a
  `(n_channels, n_timepoints)` array.
- `FODN(...)`: the estimator class (`.fit(x)`, `.result()`).
- `FODNResult`: fields: `alpha`, `coupling`, `eigenvalues`, `leading_eig`,
  `dominant_eigvec`, `sparseness`.
- `HaarWaveletTransform(x)`: helper (`.normalize()`, `.transform()`).

## `fdnkit.features`

- `extract_features(signals, *, do_dfa=True, do_mfdfa=True, do_fodn=True, ...) -> dict`:
  one tidy feature row for a multi-channel segment.
- `feature_table(trials, ...) -> pandas.DataFrame`: one row per trial record
  (each record has `signals` and optionally `trial_id`, `group`, `label`).
- `dfa_features`, `mfdfa_features`, `fodn_features`: per-family dictionaries.
- `CORE_FEATURES`: the five reference-study feature names.

## `fdnkit.classify`

- `classify(X, y, groups=None, *, cv="loso", permutation=True, bootstrap=True, ...) -> ClassificationResult`:
  subject-wise CV by default; `groups` required for `cv in {"loso","group_kfold"}`.
- `classify_dataframe(df, *, feature_cols=None, label_col="label", group_col="group", ...) -> ClassificationResult`.
- `make_classifier(C=1.0, max_iter=1000)`: the scaler + logistic-regression pipeline.
- `ClassificationResult`: fields include `balanced_accuracy`, `auc`,
  `permutation_p`, `ci95`, `per_group`; method `summary()`.

## `fdnkit.io`

- `load_edf(path) -> Recording`: via MNE (`fdnkit[io]`).
- `load_h5(path, ...) -> Recording`: via h5py (`fdnkit[io]`).
- `load_labels_excel(path, ...) -> dict`: `{trial_id: 0/1}` from a scoresheet.
- `save_features(df, path)`, `load_features(path)`.
- `Recording`: fields: `signals`, `fs`, `channel_names`, `times`; properties:
  `n_channels`, `n_samples`, `duration`.

## `fdnkit.preprocessing`

- `zscore(signals, axis=-1)`.
- `flag_bad_channels(signals, channel_names=None, *, max_flat_fraction=0.05, ...) -> list[int]`.
- `find_flat_runs(signal, min_length=16, atol=0.0) -> ndarray` of `[start, stop)` bounds:
  constant runs from saturation, clipping, or dropout.
- `flat_fraction(signal, min_length=16, atol=0.0) -> float`: share of samples in such runs.
- `segment(signals, window, *, step=None, min_size=None)`: generator of
  `(start, stop, chunk)`; `sliding_windows(...)` returns the list.

## `fdnkit.viz` (needs `fdnkit[viz]`)

- `plot_fluctuation`, `plot_hq`, `plot_multifractal_spectrum`,
  `plot_hurst_over_time`, `plot_alpha_distribution`, `plot_coupling_matrix`,
  `plot_eigenvector_hubs`, each takes an optional `ax` and returns it.

## `fdnkit.synthetic`

- `fgn(n, hurst=0.7, *, seed=None)`: fractional Gaussian noise (Davies–Harte).
- `fbm(n, hurst=0.7, *, seed=None)`: fractional Brownian motion.
- `binomial_cascade(n_levels=12, p=0.3, *, seed=None)`: a multifractal.
- `synthetic_ieeg(n_channels=8, n_samples=5000, ...) -> (signals, names)`.

## Command line

- `fdnkit extract <recording> --window 1.0 [--drop-bad --zscore] --out feats.csv`
- `fdnkit classify <feats.csv> --label label --group group [--cv loso]`
- `fdnkit demo --out demo_features.csv`
