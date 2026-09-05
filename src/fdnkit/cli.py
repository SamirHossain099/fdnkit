"""Command-line interface: ``fdnkit extract`` and ``fdnkit classify``.

Examples
--------
Extract features from an EDF or HDF5 recording into a one-row CSV::

    fdnkit extract recording.edf --window 1.0 --out features.csv

Run a self-contained demo on synthetic data (no files needed)::

    fdnkit demo --out demo_features.csv

Evaluate a feature CSV with honest subject-wise CV::

    fdnkit classify features.csv --label label --group subject
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def _cmd_extract(args):
    from .features import feature_table
    from .io import Recording, load_edf, load_h5, save_features
    from .preprocessing import flag_bad_channels, segment, zscore

    path = args.input
    if path.lower().endswith((".edf", ".edf+", ".bdf")):
        rec = load_edf(path)
    elif path.lower().endswith((".h5", ".hdf5")):
        rec = load_h5(path, fs=args.fs)
    else:
        raise SystemExit(f"unsupported input extension: {path}")

    assert isinstance(rec, Recording)
    signals = rec.signals
    if args.drop_bad:
        bad = set(flag_bad_channels(signals, rec.channel_names))
        keep = [i for i in range(signals.shape[0]) if i not in bad]
        signals = signals[keep]
        print(f"[fdnkit] dropped {len(bad)} bad channel(s); kept {len(keep)}")
    if args.zscore:
        signals = zscore(signals)

    win = int(args.window * rec.fs)
    trials = []
    for k, (_start, _stop, chunk) in enumerate(segment(signals, win)):
        trials.append({
            "trial_id": f"win{k}",
            "group": args.group or "unknown",
            "signals": chunk,
        })
    if not trials:
        # whole-recording single window fallback
        trials = [{"trial_id": "full", "group": args.group or "unknown", "signals": signals}]

    df = feature_table(
        trials,
        do_fodn=not args.no_fodn,
        progress=True,
        fodn_kwargs={"n_iter": args.fodn_iter},
    )
    save_features(df, args.out)
    print(f"[fdnkit] wrote {len(df)} row(s) x {df.shape[1]} cols -> {args.out}")


def _cmd_demo(args):
    from .features import feature_table
    from .io import save_features
    from .synthetic import synthetic_ieeg

    rng = np.random.default_rng(args.seed)
    trials = []
    for subj in range(args.subjects):
        for t in range(args.trials_per_subject):
            label = int(rng.random() < 0.5)
            h = 0.7 + 0.06 * label  # a faint, learnable class signal
            sig, _ = synthetic_ieeg(
                n_channels=args.channels, n_samples=args.samples,
                hurst=h, seed=rng,
            )
            trials.append({
                "trial_id": f"S{subj}_T{t}",
                "group": f"S{subj}",
                "label": label,
                "signals": sig,
            })
    df = feature_table(trials, do_fodn=not args.no_fodn,
                       fodn_kwargs={"n_iter": args.fodn_iter}, progress=True)
    save_features(df, args.out)
    print(f"[fdnkit] demo wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")
    print(f"[fdnkit] try: fdnkit classify {args.out} --label label --group group")


def _cmd_classify(args):
    from .classify import classify_dataframe

    df = pd.read_csv(args.input)
    res = classify_dataframe(
        df,
        label_col=args.label,
        group_col=args.group,
        cv=args.cv,
        permutation=not args.no_permutation,
        n_permutations=args.n_permutations,
        bootstrap=not args.no_bootstrap,
    )
    print(res.summary())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fdnkit", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    pe = sub.add_parser("extract", help="extract features from a recording")
    pe.add_argument("input", help="EDF/BDF or HDF5 file")
    pe.add_argument("--out", default="features.csv", help="output CSV path")
    pe.add_argument("--window", type=float, default=1.0, help="window length (seconds)")
    pe.add_argument("--fs", type=float, default=1000.0, help="sampling rate for HDF5 without a time vector")
    pe.add_argument("--group", default=None, help="group/subject id to tag rows with")
    pe.add_argument("--zscore", action="store_true", help="z-score channels before analysis")
    pe.add_argument("--drop-bad", action="store_true", help="auto-drop flat/EKG/DC channels")
    pe.add_argument("--no-fodn", action="store_true", help="skip the (slow) FODN features")
    pe.add_argument("--fodn-iter", type=int, default=5, help="FODN ADMM iterations")
    pe.set_defaults(func=_cmd_extract)

    pd_ = sub.add_parser("demo", help="generate a synthetic feature table (no data needed)")
    pd_.add_argument("--out", default="demo_features.csv")
    pd_.add_argument("--subjects", type=int, default=6)
    pd_.add_argument("--trials-per-subject", type=int, default=8)
    pd_.add_argument("--channels", type=int, default=6)
    pd_.add_argument("--samples", type=int, default=2000)
    pd_.add_argument("--fodn-iter", type=int, default=3)
    pd_.add_argument("--no-fodn", action="store_true")
    pd_.add_argument("--seed", type=int, default=0)
    pd_.set_defaults(func=_cmd_demo)

    pc = sub.add_parser("classify", help="honest cross-validated classification of a feature CSV")
    pc.add_argument("input", help="feature CSV")
    pc.add_argument("--label", default="label", help="label column")
    pc.add_argument("--group", default="group", help="group/subject column")
    pc.add_argument("--cv", default="loso", choices=["loso", "group_kfold", "loo"])
    pc.add_argument("--no-permutation", action="store_true")
    pc.add_argument("--n-permutations", type=int, default=1000)
    pc.add_argument("--no-bootstrap", action="store_true")
    pc.set_defaults(func=_cmd_classify)

    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"fdnkit {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
