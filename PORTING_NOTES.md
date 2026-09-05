# Porting notes

FDNkit generalizes a validated but single-purpose research codebase into a
reusable library. This table maps each ported piece to its origin in the
original single-purpose research tool and records what changed.

| FDNkit module | Ported from | Notes on the port |
|---|---|---|
| `mfdfa.py` | `software_app/gui/mfdfa_analysis_window.py::run_mfdfa`; `MATLAB_Code/MFDFA.m`, `DFA_of_EEG.m` | Extracted the core maths out of the PyQt window into pure functions. Same forward-partition algorithm, `q=0` log-average limit, and `eps` flooring. Added `MFDFAResult`, `delta_h`, and a Legendre-transform `f(α)` spectrum. |
| `dfa.py` | same as above (the `q=2` special case) | Thin, single-purpose entry point sharing `mfdfa.py`'s internals. Verified `h(q=2) == dfa.hurst` exactly. |
| `fodn.py` | `software_app/utils/fodn_code.py::fracOrdUU`, `HaarWaveletTransform` | Preserved the numerical procedure (Haar-wavelet order estimation, Grünwald–Letnikov differencing, regularized least squares + ADMM-LASSO). Renamed to a scikit-learn-style `FODN.fit`/`result`. Replaced silent `print`-and-continue error handling with proper exceptions; a rank-deficient `B` now falls back to a well-conditioned selector instead of raising. Removed the unused sparse-computation branches. |
| `features.py` | `software_app/utils/feature_extractor.py::FeatureExtractor` | Reimplemented to compute features directly from arrays / result objects instead of scraping a directory tree of CSVs. Kept the five "core" feature names (`MeanAlpha`, `VarAlpha`, `LeadingEig`, `MF_DFA_H`, `MF_DFA_Hq_mean`) as aliases so results line up with the reference study. |
| `io.py` | `software_app/utils/file_utils.py`, `label_strategies.py::ExcelMathScoreLabeler` | Removed the PyQt `QMessageBox` coupling; IO now raises plain exceptions. EDF via MNE and HDF5 via h5py are optional, lazily imported dependencies. |
| `classify.py` | the original model-training window and honest-CV revalidation script | Folded the honest-CV revalidation logic into the library. The GUI's random 80/20 split is deliberately **not** the default; leave-one-subject-out is, with a required `groups` argument, a group-aware permutation test, and bootstrap CIs. |
| `viz.py` | `scripts/analysis_scripts/{plotter,EigenvectorHeatmap}.py`; the GUI plotting code | Reduced to composable, `ax`-returning functions with matplotlib as an optional dependency. |
| `preprocessing.py` | scattered windowing/artifact logic in the GUI windows | Consolidated z-scoring, bad-channel flagging, and fixed-length windowing. |
| `synthetic.py` | new | Added for tests/examples: exact fGn/fBm (Davies–Harte), binomial cascades, and coupled multi-channel signals, so the toolkit runs with no data download. |

## Known issues from the source that were fixed cleanly

- `log2(0)` / divide-by-zero warnings in DFA/MFDFA: handled with `eps` flooring
  and finite-value masking in the log–log regression.
- **Negative-`q` MFDFA blow-up on real (integer-quantized) recordings**: the
  original machine-epsilon RMS floor let degenerately-detrended segments collapse
  to ~0 and dominate negative-`q` moments, fabricating a huge multifractal width
  (observed `Δh ≈ 10` on real EEG). Replaced with a **scale-relative floor**
  (`rel_floor`, a fraction of each scale's median fluctuation, default `1e-3`);
  this restores physical `Δh` on real data while leaving synthetic results
  (cascade `Δh`, monofractal `Δh`, Hurst recovery) numerically unchanged.
- FODN order-estimation `log2` of zero variance: floored at `1e-10` (as in the
  source) and documented.
- Rank-deficient heuristic `B` matrix previously raised: now falls back to a
  standard-basis selector.
- Silent `except: print(...)` blocks in `fracOrdUU.fit`: replaced with explicit
  `ValueError`s so callers can handle failures.

## Deliberately deferred (out of scope for v1)

- DuckDB result store (`database_scripts/`).
- Directed-connectivity measures (Granger/transfer entropy).
- A graphical interface: the core is headless by design.
