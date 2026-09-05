# FDNkit

**Fractional Dynamical Network & Multifractal toolkit for intracranial EEG**

[![CI](https://github.com/SamirHossain099/fdnkit/actions/workflows/ci.yml/badge.svg)](https://github.com/SamirHossain099/fdnkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

FDNkit turns intracranial-EEG (iEEG) recordings into **fractal** and
**fractional-dynamical-network** features and evaluates them **honestly**. It
packages methods validated in Beeram et al. (2026, *Front. Netw. Physiol.*
6:1768476) and the fractional-dynamical-network line of work of Gupta et al.
(2018) and Xue & Bogdan (2017) as a clean, documented, tested library, not a one-off GUI welded to a single dataset.

It computes:

- **DFA**: monofractal Hurst exponent `H`.
- **MFDFA**: generalized Hurst `h(q)`, multifractal width `Δh`, and the
  singularity spectrum `f(α)`.
- **FODN**: a fractional-order dynamical network: per-channel fractional orders
  `α`, a sparse directed coupling matrix `A`, and eigenvector "hub" scores.
- **Feature tables**: tidy, one-row-per-trial pandas DataFrames.
- **Honest classification**: a logistic-regression harness that defaults to
  **leave-one-subject-out** cross-validation with a subject-level permutation
  test, because row-wise splits leak patient identity and inflate accuracy.

## Install

```bash
pip install fdnkit                # core: numpy, scipy, pandas, scikit-learn
pip install "fdnkit[viz]"         # + matplotlib for plots
pip install "fdnkit[io]"          # + mne (EDF) and h5py (HDF5) readers
pip install "fdnkit[all]"         # everything
```

From source:

```bash
git clone https://github.com/SamirHossain099/fdnkit
cd fdnkit
pip install -e ".[dev]"
pytest
```

## 60-second example (no data download)

```python
from fdnkit.synthetic import synthetic_ieeg
from fdnkit.features import extract_features

# A small, sparsely-coupled synthetic iEEG trial (8 channels, 5 s @ 1 kHz).
signals, channel_names = synthetic_ieeg(n_channels=8, n_samples=5000, seed=0)

# One tidy feature row: DFA H, MFDFA h(q)/Δh, FODN α / leading eigenvalue / hubs.
features = extract_features(signals)
print(features["MF_DFA_H"], features["MeanAlpha"], features["LeadingEig"])
```

Analyze a single signal directly:

```python
import numpy as np
from fdnkit.dfa import dfa
from fdnkit.mfdfa import mfdfa
from fdnkit.fodn import fit_fodn

x = signals[0]
print("Hurst:", dfa(x).hurst)
print("multifractal width Δh:", mfdfa(x).delta_h)

fodn = fit_fodn(signals)                 # (channels, timepoints)
print("leading eigenvalue:", fodn.leading_eig)
print("hub scores:", np.round(fodn.dominant_eigvec, 3))
```

## Honest classification

```python
from fdnkit.classify import classify_dataframe

# df has feature columns plus 'label' and 'group' (e.g. subject id) columns.
result = classify_dataframe(df, label_col="label", group_col="group", cv="loso")
print(result.summary())
# Leave-one-subject-out balanced accuracy, ROC-AUC, a subject-level
# permutation p-value, and a bootstrap 95% CI.
```

`cv="loso"` (the default) holds out whole subjects and **requires** `groups`.
Trial-wise `cv="loo"` is available but must be requested explicitly and is
labeled *optimistic*: it is the leakage-prone scheme FDNkit exists to warn about.

## Command line

```bash
# Self-contained demo: synthesize a labeled cohort and classify it honestly.
fdnkit demo --out demo_features.csv
fdnkit classify demo_features.csv --label label --group group

# Extract features from your own recording (EDF via MNE, or HDF5).
fdnkit extract recording.edf --window 1.0 --drop-bad --zscore --out features.csv
```

## Design principles

- **Array-first core.** `dfa(signal)`, `mfdfa(signal)`, `fit_fodn(signals)` are
  pure functions on NumPy arrays. Pandas/IO/plotting layer on top.
- **Depend, don't duplicate.** IO, montages, and filtering defer to
  [MNE-Python](https://mne.tools); FDNkit adds only the fractal/FODN methods.
- **Deterministic and seedable.** Bad channels log a warning instead of crashing.
- **Honest by default.** Subject-wise CV and permutation testing are the
  headline, not an afterthought.

## Validation

FDNkit's numerical core is checked against ground truth (see `tests/`):

- DFA recovers the Hurst exponent of fractional Gaussian noise across
  `H = 0.3…0.9`; white noise → `H ≈ 0.5`, Brownian motion → `H ≈ 1.5`.
- MFDFA reports a wide `h(q)` for a multiplicative binomial cascade and a narrow
  one for a monofractal signal; `h(q=2)` matches the DFA Hurst exponent exactly.
- FODN recovers finite fractional orders, coupling, and hubs on synthetic
  coupled systems.

## Citation

If you use FDNkit, please cite the software (see [`CITATION.cff`](CITATION.cff))
and the methods paper:

> Beeram, S. P., Farris, M., Hossain, S., Rethans, N., Kang, J. Y., & Pereira,
> E. A. (2026). *Quantifying cognitive effort's impact on suppression of
> epilepsy-associated after discharges.* Frontiers in Network Physiology, 6,
> 1768476. https://doi.org/10.3389/fnetp.2026.1768476

## License

MIT; see [LICENSE](LICENSE). The underlying fractional-dynamical-network method
is due to Gupta, Pequito & Bogdan (2018) and Xue & Bogdan (2017); please cite
them when using the FODN module.
