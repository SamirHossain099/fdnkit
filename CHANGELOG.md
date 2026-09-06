# Changelog

All notable changes to FDNkit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-05

### Added
- `fdnkit.preprocessing.find_flat_runs` and `flat_fraction`: locate constant
  runs (amplifier saturation, clipping, dropped or interpolated samples) and
  report the fraction of samples they cover.
- `flag_bad_channels` gains `max_flat_fraction` (default `0.05`) and
  `flat_run_length`, so partially-flat channels are flagged, not only wholly
  flat ones.
- `mfdfa(..., check_flat=True)` warns when negative `q` is requested and the
  signal contains constant runs long enough to fill an analysis segment. Such
  runs give a segment near-zero detrended variance, which dominates negative-`q`
  moments and can inflate the multifractal width by orders of magnitude.

## [1.0.0] - 2026-09-05

Initial public release.

### Added
- `fdnkit.dfa`: monofractal DFA Hurst exponent.
- `fdnkit.mfdfa`: MFDFA generalized Hurst `h(q)`, multifractal width `Δh`, and
  the `f(α)` singularity spectrum.
- `fdnkit.fodn`: fractional-order dynamical network model (per-channel `α`,
  sparse directed coupling matrix, eigenvector hub scores, sparseness).
- `fdnkit.features`: tidy per-trial feature tables with a five-feature "core"
  set matching the reference study.
- `fdnkit.classify`: subject-wise cross-validation by default (leave-one-subject-out),
  group-aware permutation test, bootstrap confidence intervals; trial-wise LOO is
  opt-in and labeled optimistic.
- `fdnkit.io`: EDF (MNE) and HDF5 (h5py) readers, Excel label loading, feature CSV IO.
- `fdnkit.preprocessing`: z-scoring, bad-channel flagging, windowing.
- `fdnkit.viz`: fluctuation, `h(q)`, spectrum, coupling-heatmap, and hub plots.
- `fdnkit.synthetic`: fractional Gaussian noise / motion (Davies–Harte),
  binomial cascades, and coupled multi-channel iEEG-like signals.
- `fdnkit` command-line interface: `extract`, `classify`, `demo`.
- Test suite (79 tests, ~91% coverage) validating the numerical core against
  signals with known scaling properties; GitHub Actions CI on Python 3.9–3.12.
- JOSS paper draft (`paper/`), documentation (`docs/`), and runnable examples:
  a synthetic quickstart (`examples/quickstart.py`) and a real-data walkthrough
  on public PhysioNet EEG (`examples/real_data_eegbci.py`).
