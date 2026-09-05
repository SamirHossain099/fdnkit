# Examples

## `quickstart.py`

A complete, self-contained pipeline that needs **no data download**:

```bash
python examples/quickstart.py
```

It

1. synthesizes a small labeled cohort (6 subjects × 8 trials) with
   `fdnkit.synthetic.synthetic_ieeg`,
2. runs single-trial DFA, MFDFA, and FODN analyses,
3. builds a per-trial feature table, and
4. classifies it with honest leave-one-subject-out cross-validation, a
   permutation test, and a bootstrap confidence interval.

If matplotlib is installed (`pip install "fdnkit[viz]"`) it also writes a
four-panel figure (`quickstart_panel.png`) showing the DFA fluctuation curve, the
generalized Hurst `h(q)`, the FODN coupling matrix, and eigenvector hub scores.

## `real_data_eegbci.py`

The same pipeline on a **real public EEG recording** instead of synthetic data:

```bash
pip install "fdnkit[io,viz]"
python examples/real_data_eegbci.py
```

It downloads one short run (~2 min, a few MB) of the PhysioNet EEG Motor
Movement/Imagery dataset via MNE (cached after the first run), loads the EDF
through `fdnkit.io.load_edf`, flags and drops bad channels, and runs DFA, MFDFA,
FODN, and feature-table extraction on the real signals. It exits cleanly with
instructions if MNE is missing or the machine is offline.

## Using your own recordings

Swap the synthetic generator for a real recording:

```python
from fdnkit.io import load_edf          # or load_h5
from fdnkit.preprocessing import zscore, flag_bad_channels, segment

rec = load_edf("recording.edf")          # needs: pip install "fdnkit[io]"
bad = flag_bad_channels(rec.signals, rec.channel_names)
keep = [i for i in range(rec.n_channels) if i not in bad]
signals = zscore(rec.signals[keep])

trials = [
    {"trial_id": f"win{k}", "group": "subject-01", "signals": chunk}
    for k, (_s, _e, chunk) in enumerate(segment(signals, int(1.0 * rec.fs)))
]
```

Then `fdnkit.features.feature_table(trials)` and
`fdnkit.classify.classify_dataframe(...)` as in the quickstart. The same flow is
available from the command line via `fdnkit extract` and `fdnkit classify`.
