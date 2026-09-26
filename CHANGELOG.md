# Changelog

All notable changes to FDNkit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Regression tests pinning the scale invariance of `h(q)`. Multiplying a signal
  by a positive constant cannot change its scaling exponents, and `rel_floor`
  is what makes that hold: the floor tracks the median positive fluctuation at
  each scale instead of sitting at a fixed value. The tests cover a clean
  fractional Gaussian noise signal, white noise against its known `h(q=2)`, and
  a signal carrying a constant run where an absolute floor drifts by about 4.0
  across twelve orders of magnitude of input scaling while the relative floor
  drifts by about 1.6e-4.
- `CODE_OF_CONDUCT.md`, adapted from the Contributor Covenant 2.1.
- A `py.typed` marker (PEP 561), so mypy and pyright now read FDNkit's inline
  annotations instead of treating the whole API as `Any`. A test checks that the
  marker ships with the package.

### Fixed
- Type hints that would have been wrong once visible downstream.
  `segment` and `sliding_windows` declared `step: int = None` and
  `min_size: int = None`, so a user passing the documented default `step=None`
  would get a false error from their type checker. They are now `int | None`,
  and both functions declare their return types. `mfdfa_features` reused one
  variable name for a list and an array; the array now has its own name.

### Changed
- CI now runs on Linux, Windows and macOS rather than Linux alone, and adds
  Python 3.13. Windows matters here specifically: NumPy's default integer is
  32-bit there, which is the kind of platform difference that silently changes
  numerical results rather than raising.
- CI now type-checks the package with mypy, and `mypy` joins the `dev` extras.
  With the marker in place the annotations are part of the public contract, so
  they need a check that keeps them accurate.

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
