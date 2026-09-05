import numpy as np
import pandas as pd
import pytest

from fdnkit.io import Recording, load_features, load_h5, load_labels_excel, save_features


def test_recording_properties():
    rec = Recording(signals=np.zeros((4, 1000)), fs=500.0, channel_names=["a", "b", "c", "d"])
    assert rec.n_channels == 4
    assert rec.n_samples == 1000
    assert rec.duration == 2.0


def test_features_roundtrip(tmp_path):
    df = pd.DataFrame({"trial_id": ["t0", "t1"], "MeanAlpha": [0.1, 0.2], "label": [0, 1]})
    path = tmp_path / "feats.csv"
    save_features(df, path)
    back = load_features(path)
    pd.testing.assert_frame_equal(df, back)


def test_h5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    sig = np.random.default_rng(0).standard_normal((5, 2000))
    t = np.arange(2000) / 1000.0
    names = [f"LH{i}" for i in range(5)]
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data/Signals", data=sig)
        f.create_dataset("data/Time", data=t)
        f.create_dataset("metadata/channel_names",
                         data=np.array(names, dtype="S10"))
    rec = load_h5(path)
    assert rec.signals.shape == (5, 2000)
    assert rec.channel_names == names
    assert abs(rec.fs - 1000.0) < 1e-6


def test_load_labels_excel(tmp_path):
    pytest.importorskip("openpyxl")
    # Header on the 2nd row (header=1), matching the reference scoresheet layout.
    path = tmp_path / "labels.xlsx"
    df = pd.DataFrame({
        "Patient_Session_Trial": ["P1_S1_T1", "P1_S1_T2", "P2_S1_T1", "P2_S1_T2"],
        "Math_Score": ["M1", "M0", "M1", "MC"],
    })
    # Write with a junk first row so real header sits on row index 1.
    with pd.ExcelWriter(path) as xl:
        pd.concat([pd.DataFrame([["", ""]], columns=df.columns), df]).to_excel(
            xl, index=False, header=True
        )
    # The written file has: row0=header names, row1=junk, row2..=data.
    # load_labels_excel uses header=1 -> treats row1(junk) as header. So instead
    # write a version whose second row is the header:
    path2 = tmp_path / "labels2.xlsx"
    top = pd.DataFrame([["title", "sheet"]])
    with pd.ExcelWriter(path2) as xl:
        top.to_excel(xl, index=False, header=False, startrow=0)
        df.to_excel(xl, index=False, header=True, startrow=1)
    mapping = load_labels_excel(path2)
    assert mapping["P1_S1_T1"] == 1
    assert mapping["P1_S1_T2"] == 0
    assert "P2_S1_T2" not in mapping  # 'MC' skipped


def test_load_h5_missing_backend_message(monkeypatch):
    # If h5py import fails, error should be actionable.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "h5py":
            raise ImportError("no h5py")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="h5py"):
        load_h5("nonexistent.h5")
