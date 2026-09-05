"""Input/output: load recordings, load trial labels, read/write feature tables.

Heavy IO dependencies are optional and imported lazily so the core analysis
stack (numpy/scipy/pandas/scikit-learn) stays lightweight:

* EDF reading uses **MNE-Python** (``pip install fdnkit[io]``).
* HDF5 reading uses **h5py**.

Both raise a clear, actionable error if the backend is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Recording", "load_edf", "load_h5", "load_labels_excel",
           "save_features", "load_features"]


@dataclass
class Recording:
    """A loaded multi-channel recording.

    Attributes
    ----------
    signals : numpy.ndarray
        Shape ``(n_channels, n_samples)``.
    fs : float
        Sampling frequency in Hz.
    channel_names : list of str
    times : numpy.ndarray | None
        Optional per-sample time vector (seconds).
    """

    signals: np.ndarray
    fs: float
    channel_names: list
    times: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return self.signals.shape[0]

    @property
    def n_samples(self) -> int:
        return self.signals.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs


def load_edf(path, *, preload: bool = True) -> Recording:
    """Load an EDF/EDF+ recording via MNE-Python.

    Parameters
    ----------
    path : str | pathlib.Path
    preload : bool
        Read sample data into memory immediately.

    Returns
    -------
    Recording
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised only without mne
        raise ImportError(
            "Reading EDF requires MNE-Python. Install with `pip install fdnkit[io]` "
            "or `pip install mne`."
        ) from exc

    raw = mne.io.read_raw_edf(str(path), preload=preload, verbose="ERROR")
    signals = raw.get_data()
    fs = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    times = raw.times.copy()
    return Recording(signals=signals, fs=fs, channel_names=names, times=times)


def load_h5(path, *, signals_key="data/Signals", time_key="data/Time",
            names_key="metadata/channel_names", fs: float = 1000.0) -> Recording:
    """Load a recording from an HDF5 file (h5py).

    Defaults match the reference layout: signals at ``data/Signals`` with shape
    ``(n_channels, n_samples)``, an optional time vector at ``data/Time``, and
    optional channel names at ``metadata/channel_names``.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading HDF5 requires h5py. Install with `pip install fdnkit[io]` "
            "or `pip install h5py`."
        ) from exc

    with h5py.File(str(path), "r") as f:
        signals = np.asarray(f[signals_key][:])
        times = np.asarray(f[time_key][:]) if time_key in f else None
        if names_key in f:
            raw_names = f[names_key][:]
            names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw_names]
        else:
            names = [f"CH{i + 1}" for i in range(signals.shape[0])]

    if times is not None and times.size > 1:
        dt = float(np.median(np.diff(times)))
        if dt > 0:
            fs = 1.0 / dt
    return Recording(signals=signals, fs=fs, channel_names=names, times=times)


def load_labels_excel(path, *, id_col="Patient_Session_Trial", score_col="Math_Score",
                      success_code="M1", failure_code="M0", header: int = 1) -> dict:
    """Load a ``{trial_id: 0/1}`` label mapping from an Excel scoresheet.

    Ports ``utils/label_strategies.ExcelMathScoreLabeler``: rows whose score
    equals ``success_code`` map to 1, ``failure_code`` to 0; anything else
    (blank, "MC", ...) is skipped.

    Parameters
    ----------
    path : str | pathlib.Path
    id_col, score_col : str
        Column names for the trial identifier and the score.
    success_code, failure_code : str
        Score strings mapped to 1 and 0 respectively (compared case-insensitively).
    header : int
        Row index (0-based) of the header. Defaults to 1 (second row).

    Returns
    -------
    dict[str, int]
    """
    df = pd.read_excel(path, header=header)
    df = df[[id_col, score_col]].dropna(subset=[id_col])
    mapping: dict = {}
    for _, row in df.iterrows():
        tid = str(row[id_col]).strip()
        score = str(row[score_col]).strip().upper()
        if score == success_code.upper():
            mapping[tid] = 1
        elif score == failure_code.upper():
            mapping[tid] = 0
    return mapping


def save_features(df: pd.DataFrame, path) -> Path:
    """Write a feature DataFrame to CSV (index omitted). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_features(path) -> pd.DataFrame:
    """Read a feature table CSV back into a DataFrame."""
    return pd.read_csv(path)
