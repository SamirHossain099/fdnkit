import numpy as np
import pandas as pd
import pytest

from fdnkit.cli import main


def test_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    assert "fdnkit" in capsys.readouterr().out


def test_help_without_command(capsys):
    rc = main([])
    assert rc == 1


def test_demo_then_classify(tmp_path, capsys):
    out = tmp_path / "demo.csv"
    rc = main([
        "demo", "--out", str(out), "--subjects", "4",
        "--trials-per-subject", "5", "--channels", "5",
        "--samples", "1200", "--fodn-iter", "2", "--seed", "0",
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert len(df) == 20
    assert "label" in df.columns and "group" in df.columns

    rc = main([
        "classify", str(out), "--label", "label", "--group", "group",
        "--n-permutations", "100",
    ])
    assert rc == 0
    text = capsys.readouterr().out
    assert "balanced accuracy" in text
    assert "loso" in text


def test_extract_from_h5(tmp_path, capsys):
    h5py = pytest.importorskip("h5py")
    # Write a small synthetic recording to HDF5, then run `fdnkit extract`.
    from fdnkit.synthetic import synthetic_ieeg

    sig, names = synthetic_ieeg(n_channels=5, n_samples=4000, seed=0)
    rec_path = tmp_path / "rec.h5"
    with h5py.File(rec_path, "w") as f:
        f.create_dataset("data/Signals", data=sig)
        f.create_dataset("data/Time", data=np.arange(sig.shape[1]) / 1000.0)
        f.create_dataset("metadata/channel_names", data=np.array(names, dtype="S10"))

    out = tmp_path / "feats.csv"
    rc = main([
        "extract", str(rec_path), "--out", str(out),
        "--window", "1.0", "--group", "S1", "--zscore",
        "--drop-bad", "--fodn-iter", "2",
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert len(df) == 4  # 4 one-second windows from 4000 samples
    assert "MeanAlpha" in df.columns
    assert (df["group"] == "S1").all()


def test_classify_optimistic_loo(tmp_path, capsys):
    out = tmp_path / "demo.csv"
    main(["demo", "--out", str(out), "--subjects", "3", "--trials-per-subject", "4",
          "--channels", "4", "--samples", "1000", "--fodn-iter", "2"])
    rc = main(["classify", str(out), "--cv", "loo", "--no-permutation"])
    assert rc == 0
    assert "OPTIMISTIC" in capsys.readouterr().out
