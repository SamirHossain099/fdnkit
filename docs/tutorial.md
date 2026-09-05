# Tutorial

This walkthrough covers the full FDNkit pipeline on synthetic data, then shows
how to swap in your own recordings. Everything here runs with no downloads.

## 1. Analyze a single signal

```python
import numpy as np
from fdnkit.dfa import dfa
from fdnkit.mfdfa import mfdfa, multifractal_spectrum
from fdnkit.synthetic import fgn

x = fgn(8000, hurst=0.75, seed=0)      # long-range-correlated noise

print("Hurst exponent:", dfa(x).hurst)          # ~0.75
res = mfdfa(x)
print("generalized Hurst h(q):", np.round(res.hq, 3))
print("multifractal width:", res.delta_h)
alpha, f_alpha = multifractal_spectrum(res)     # singularity spectrum
```

## 2. Fit a fractional-order dynamical network

```python
from fdnkit.fodn import fit_fodn
from fdnkit.synthetic import synthetic_ieeg

signals, names = synthetic_ieeg(n_channels=8, n_samples=5000, seed=1)
fodn = fit_fodn(signals, n_iter=10)

print("per-channel alpha:", np.round(fodn.alpha, 3))
print("leading eigenvalue (spectral radius):", fodn.leading_eig)
print("hub scores:", np.round(fodn.dominant_eigvec, 3))
print("network sparseness:", fodn.sparseness)
```

## 3. Build a per-trial feature table

Organize your data as a list of trial records. The `group` field (e.g. subject
id) is what enables honest cross-validation later.

```python
from fdnkit.features import feature_table
from fdnkit.synthetic import synthetic_ieeg

trials = []
for s in range(6):                      # subjects
    for t in range(8):                  # trials per subject
        sig, _ = synthetic_ieeg(n_channels=6, n_samples=3000, seed=(s, t))
        trials.append({
            "trial_id": f"S{s}_T{t}",
            "group": f"S{s}",
            "label": (s + t) % 2,       # your class label
            "signals": sig,
        })

df = feature_table(trials, fodn_kwargs={"n_iter": 5})
print(df.shape)                          # (48, ~22)
```

## 4. Classify honestly

```python
from fdnkit.classify import classify_dataframe

result = classify_dataframe(df, label_col="label", group_col="group", cv="loso")
print(result.summary())
```

`cv="loso"` (the default) holds out whole subjects, so no patient appears in both
train and test. It **requires** the `group` column. The report includes balanced
accuracy, ROC-AUC, a subject-level permutation p-value, and a bootstrap 95% CI.

!!! warning "Trial-wise CV is optimistic"
    `cv="loo"` evaluates leave-one-*trial*-out. Because the same subject can then
    appear in both training and test folds, it typically **overestimates**
    accuracy. FDNkit lets you request it, but labels it `OPTIMISTIC`.

## 5. Use your own recordings

```python
from fdnkit.io import load_edf
from fdnkit.preprocessing import flag_bad_channels, zscore, segment

rec = load_edf("recording.edf")          # pip install "fdnkit[io]"
bad = set(flag_bad_channels(rec.signals, rec.channel_names))
keep = [i for i in range(rec.n_channels) if i not in bad]
signals = zscore(rec.signals[keep])

win = int(1.0 * rec.fs)                   # 1-second windows
trials = [{"trial_id": f"w{k}", "group": "subj-01", "signals": chunk}
          for k, (_a, _b, chunk) in enumerate(segment(signals, win))]
```

Or from the shell:

```bash
fdnkit extract recording.edf --window 1.0 --drop-bad --zscore --out feats.csv
fdnkit classify feats.csv --label label --group group
```
