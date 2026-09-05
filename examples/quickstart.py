"""FDNkit quickstart -- runs end-to-end on synthetic data (no download needed).

    python examples/quickstart.py

Generates a small labeled synthetic cohort, extracts fractal + FODN features for
every trial, and evaluates them with honest leave-one-subject-out CV. If
matplotlib is installed it also writes a figure panel to ``quickstart_panel.png``.
"""

from __future__ import annotations

import numpy as np

from fdnkit.classify import classify_dataframe
from fdnkit.dfa import dfa
from fdnkit.features import feature_table
from fdnkit.fodn import fit_fodn
from fdnkit.mfdfa import mfdfa
from fdnkit.synthetic import synthetic_ieeg


def build_cohort(n_subjects=6, trials_per_subject=8, seed=0):
    """A synthetic cohort with a faint, learnable class difference in Hurst."""
    rng = np.random.default_rng(seed)
    trials = []
    for s in range(n_subjects):
        for t in range(trials_per_subject):
            label = int(rng.random() < 0.5)
            hurst = 0.68 + 0.08 * label  # class signal lives in the scaling exponent
            sig, _ = synthetic_ieeg(n_channels=6, n_samples=3000, hurst=hurst, seed=rng)
            trials.append({
                "trial_id": f"S{s}_T{t}",
                "group": f"S{s}",           # subject id -> honest CV grouping
                "label": label,
                "signals": sig,
            })
    return trials


def main():
    print("1) Generating a synthetic cohort ...")
    trials = build_cohort()

    print("2) Single-trial analyses on trial 0:")
    x0 = trials[0]["signals"]
    print(f"   channel-0 Hurst          : {dfa(x0[0]).hurst:.3f}")
    print(f"   channel-0 delta_h (MFDFA): {mfdfa(x0[0]).delta_h:.3f}")
    fodn0 = fit_fodn(x0, n_iter=4)
    print(f"   FODN leading eigenvalue  : {fodn0.leading_eig:.3f}")
    print(f"   FODN mean alpha          : {fodn0.alpha.mean():.3f}")

    print("3) Building the per-trial feature table ...")
    df = feature_table(trials, fodn_kwargs={"n_iter": 4}, progress=False)
    print(f"   feature table: {df.shape[0]} trials x {df.shape[1]} columns")

    print("4) Honest leave-one-subject-out classification:")
    result = classify_dataframe(df, label_col="label", group_col="group",
                                cv="loso", n_permutations=500)
    print(result.summary())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from fdnkit import viz

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        viz.plot_fluctuation(dfa(x0[0]), ax=axes[0, 0])
        viz.plot_hq(mfdfa(x0[0]), ax=axes[0, 1])
        viz.plot_coupling_matrix(fodn0.coupling, ax=axes[1, 0])
        viz.plot_eigenvector_hubs(fodn0.dominant_eigvec, ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig("quickstart_panel.png", dpi=120)
        print("5) Saved figure panel -> quickstart_panel.png")
    except ImportError:
        print("5) (install fdnkit[viz] to also render the figure panel)")


if __name__ == "__main__":
    main()
