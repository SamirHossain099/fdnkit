"""FDNkit on a real public EEG recording (PhysioNet EEGBCI via MNE).

Unlike ``quickstart.py`` (which uses synthetic data), this script runs the full
FDNkit pipeline on a genuine EDF recording so you can see it work end-to-end on
real signals. It downloads one short run (~2 min, a few MB) of the PhysioNet
EEG Motor Movement/Imagery dataset the first time it runs, then caches it.

Requirements::

    pip install "fdnkit[io,viz]"      # needs mne + matplotlib

Run::

    python examples/real_data_eegbci.py

The script is deliberately defensive: if MNE is not installed or the dataset
cannot be downloaded (e.g. offline), it prints instructions and exits cleanly
rather than raising.
"""

from __future__ import annotations

import sys

import numpy as np


def load_one_eegbci_run():
    """Download+load a single EEGBCI run as an fdnkit Recording, or return None."""
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError:
        print("This example needs MNE. Install with:  pip install \"fdnkit[io]\"")
        return None

    try:
        # Subject 1, run 4 (a motor-imagery run). Small EDF, cached after first use.
        paths = eegbci.load_data(subjects=1, runs=[4], update_path=True)
    except Exception as exc:  # network/download problems -> graceful exit
        print(f"Could not download the EEGBCI sample ({exc}).")
        print("Connect to the internet once to cache it, then re-run.")
        return None

    from fdnkit.io import load_edf

    rec = load_edf(paths[0])
    # EEGBCI channel names have trailing dots ('Fc5.'); tidy them for display.
    rec.channel_names = [n.strip(". ") for n in rec.channel_names]
    return rec


def main():
    print("1) Loading a real EEG run (PhysioNet EEGBCI, subject 1) ...")
    rec = load_one_eegbci_run()
    if rec is None:
        sys.exit(0)
    print(f"   {rec.n_channels} channels, {rec.n_samples} samples @ {rec.fs:.0f} Hz "
          f"({rec.duration:.1f} s)")

    from fdnkit.dfa import dfa
    from fdnkit.features import feature_table
    from fdnkit.fodn import fit_fodn
    from fdnkit.mfdfa import mfdfa
    from fdnkit.preprocessing import flag_bad_channels, segment, zscore

    # Drop non-neural / bad channels, then z-score.
    bad = set(flag_bad_channels(rec.signals, rec.channel_names))
    keep = [i for i in range(rec.n_channels) if i not in bad]
    signals = zscore(rec.signals[keep])
    names = [rec.channel_names[i] for i in keep]
    print(f"2) Kept {len(keep)} channels after artifact flagging.")

    # Single-channel fractal analysis on the first channel.
    x0 = signals[0]
    print(f"3) Channel {names[0]}: Hurst={dfa(x0).hurst:.3f}, "
          f"delta_h={mfdfa(x0).delta_h:.3f}")

    # FODN on a compact subset of channels (keeps it fast on 64-channel data).
    subset = signals[:8]
    fodn = fit_fodn(subset, n_iter=5)
    print(f"4) FODN on {subset.shape[0]} channels: "
          f"leading_eig={fodn.leading_eig:.3f}, mean_alpha={fodn.alpha.mean():.3f}")

    # Window the run into 5-second trials and build a feature table.
    win = int(5.0 * rec.fs)
    trials = []
    for k, (_a, _b, chunk) in enumerate(segment(subset, win)):
        trials.append({"trial_id": f"win{k}", "group": "S1", "signals": chunk})
    df = feature_table(trials, fodn_kwargs={"n_iter": 4})
    print(f"5) Feature table from real data: {df.shape[0]} windows x {df.shape[1]} cols")
    print("   columns:", ", ".join(c for c in df.columns if c not in ("trial_id", "group")))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from fdnkit import viz

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        viz.plot_fluctuation(dfa(x0), ax=axes[0])
        viz.plot_coupling_matrix(fodn.coupling, names[:8], ax=axes[1])
        fig.suptitle(f"FDNkit on real EEG (EEGBCI subject 1, channel {names[0]})")
        fig.tight_layout()
        fig.savefig("real_data_panel.png", dpi=120)
        print("6) Saved figure -> real_data_panel.png")
    except ImportError:
        print("6) (install fdnkit[viz] to also render a figure)")

    print("\nSame flow via the CLI:")
    print(f"   fdnkit extract <that .edf> --window 5.0 --drop-bad --zscore --out feats.csv")


if __name__ == "__main__":
    main()
