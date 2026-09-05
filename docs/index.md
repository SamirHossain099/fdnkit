# FDNkit

**Fractional Dynamical Network & Multifractal toolkit for intracranial EEG.**

FDNkit turns intracranial-EEG (iEEG) recordings into fractal and
fractional-dynamical-network features and evaluates them with honest,
subject-aware cross-validation.

## Install

```bash
pip install fdnkit            # core
pip install "fdnkit[all]"     # + matplotlib, mne, h5py, duckdb
```

## What it computes

| Module | What it gives you |
|---|---|
| `fdnkit.dfa` | Monofractal Hurst exponent `H` |
| `fdnkit.mfdfa` | Generalized Hurst `h(q)`, multifractal width `Δh`, spectrum `f(α)` |
| `fdnkit.fodn` | Per-channel fractional orders `α`, sparse coupling matrix `A`, eigenvector hubs |
| `fdnkit.features` | Tidy per-trial feature tables |
| `fdnkit.classify` | Subject-wise CV, permutation test, bootstrap CIs |

## 30-second taste

```python
from fdnkit.synthetic import synthetic_ieeg
from fdnkit.features import extract_features

signals, names = synthetic_ieeg(n_channels=8, n_samples=5000, seed=0)
features = extract_features(signals)
print(features["MF_DFA_H"], features["MeanAlpha"], features["LeadingEig"])
```

See the [Tutorial](tutorial.md) for the full pipeline and the
[API reference](api.md) for every function. The [Methods](methods.md) page
summarizes the algorithms and their validation.
