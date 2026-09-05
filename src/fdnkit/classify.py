"""Classification harness with honest, subject-wise cross-validation by default.

The reference study behind FDNkit found that evaluating an iEEG classifier with
row-wise splits leaked *patient identity* into the test set and inflated
accuracy. This module bakes the fix in:

* the default cross-validation is **leave-one-group-out** (subject-wise), and
  ``groups`` is *required* for it;
* trial-wise leave-one-out is available but must be requested explicitly and is
  labeled *optimistic*;
* every run can attach a **subject-level permutation test** and bootstrap
  confidence intervals.

Folds the honest-CV revalidation logic into a reusable API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedGroupKFold,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["ClassificationResult", "make_classifier", "classify", "classify_dataframe"]


@dataclass
class ClassificationResult:
    """Outcome of a cross-validated evaluation.

    Attributes
    ----------
    cv : str
        Which scheme ran (``"loso"``, ``"loo"``, ``"group_kfold"``).
    honest : bool
        True when the scheme holds out whole subjects (no identity leakage).
    n : int
        Number of trials.
    accuracy, balanced_accuracy, auc : float
        Pooled out-of-fold metrics.
    majority_baseline : float
        Accuracy of always predicting the majority class.
    permutation_p : float | None
        p-value from the subject-level permutation test (if run).
    permutation_chance : float | None
        Mean permuted score (empirical chance level).
    ci95 : tuple | None
        Bootstrap 95% CI on balanced accuracy (if requested).
    per_group : dict
        Per-group balanced accuracy (subject-wise schemes only).
    notes : str
        Human-readable caveats.
    """

    cv: str
    honest: bool
    n: int
    accuracy: float
    balanced_accuracy: float
    auc: float
    majority_baseline: float
    permutation_p: float | None = None
    permutation_chance: float | None = None
    ci95: tuple | None = None
    per_group: dict = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        """A one-block textual report."""
        lines = [
            f"FDNkit classification ({self.cv}{'' if self.honest else ', OPTIMISTIC'})",
            f"  trials              : {self.n}",
            f"  accuracy            : {self.accuracy:.3f}",
            f"  balanced accuracy   : {self.balanced_accuracy:.3f}",
            f"  ROC-AUC             : {self.auc:.3f}",
            f"  majority baseline   : {self.majority_baseline:.3f}",
        ]
        if self.permutation_p is not None:
            lines.append(
                f"  permutation test    : chance={self.permutation_chance:.3f}, "
                f"p={self.permutation_p:.4f}"
            )
        if self.ci95 is not None:
            lines.append(f"  bal-acc 95% CI      : [{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]")
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


def make_classifier(C: float = 1.0, max_iter: int = 1000):
    """Standard-scaler + logistic-regression pipeline used throughout."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))


def _bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(balanced_accuracy_score(yt, yp))
    if not scores:
        return None
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def classify(
    X,
    y,
    groups=None,
    *,
    cv: str = "loso",
    estimator=None,
    n_splits: int = 5,
    permutation: bool = True,
    n_permutations: int = 1000,
    bootstrap: bool = True,
    random_state: int = 0,
) -> ClassificationResult:
    """Cross-validate a binary classifier with honest defaults.

    Parameters
    ----------
    X : array-like, shape (n_trials, n_features)
    y : array-like, shape (n_trials,)
        Binary labels.
    groups : array-like, optional
        Group (e.g. subject) id per trial. **Required** for ``cv in
        {"loso", "group_kfold"}``.
    cv : {"loso", "group_kfold", "loo"}
        Cross-validation scheme. ``"loso"`` (leave-one-subject-out) is the
        default and the only fully honest single-holdout option. ``"loo"`` is
        trial-wise and *optimistic* (same subject can appear in train and test).
    estimator : sklearn estimator, optional
        Defaults to :func:`make_classifier`.
    n_splits : int
        Folds for ``cv="group_kfold"``.
    permutation : bool
        Run a (group-aware) permutation test.
    n_permutations : int
        Permutation count.
    bootstrap : bool
        Compute a bootstrap 95% CI on balanced accuracy.
    random_state : int
        Seed for permutation/bootstrap reproducibility.

    Returns
    -------
    ClassificationResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_trials, n_features)")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y length mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes present in y")

    est = make_classifier() if estimator is None else estimator
    cv = cv.lower()
    honest = cv in ("loso", "group_kfold")

    if cv in ("loso", "group_kfold") and groups is None:
        raise ValueError(
            f"cv='{cv}' is subject-wise and requires `groups` (one id per trial). "
            "Pass groups=..., or use cv='loo' explicitly for an optimistic trial-wise estimate."
        )
    if groups is not None:
        groups = np.asarray(groups)

    if cv == "loso":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, y, groups)
        notes = ""
    elif cv == "group_kfold":
        n_groups = len(np.unique(groups))
        k = min(n_splits, n_groups)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y, groups)
        notes = f"stratified group {k}-fold"
    elif cv == "loo":
        splitter = LeaveOneOut()
        split_iter = splitter.split(X)
        notes = "trial-wise LOO ignores subject structure and is leakage-prone"
    else:
        raise ValueError(f"unknown cv scheme: {cv!r}")

    preds = np.full(len(y), -1)
    probs = np.full(len(y), np.nan)
    for tr, te in split_iter:
        if len(np.unique(y[tr])) < 2:
            # a fold whose training set is single-class can't fit a classifier
            preds[te] = int(round(np.mean(y[tr])))
            probs[te] = np.mean(y[tr])
            continue
        model = est.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        if hasattr(model, "predict_proba"):
            probs[te] = model.predict_proba(X[te])[:, 1]
        else:
            probs[te] = preds[te]

    acc = accuracy_score(y, preds)
    bal = balanced_accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    majority = max(np.mean(y), 1 - np.mean(y))

    per_group: dict = {}
    if honest and groups is not None:
        for g in np.unique(groups):
            m = groups == g
            if m.sum() and len(np.unique(y[m])) >= 1:
                per_group[str(g)] = float(balanced_accuracy_score(y[m], preds[m])) \
                    if len(np.unique(y[m])) >= 2 else float(accuracy_score(y[m], preds[m]))

    perm_p = perm_chance = None
    if permutation:
        if cv == "loso":
            cv_obj = LeaveOneGroupOut()
            score, perm_scores, perm_p = permutation_test_score(
                est, X, y, groups=groups, cv=cv_obj, scoring="accuracy",
                n_permutations=n_permutations, random_state=random_state, n_jobs=1,
            )
            perm_chance = float(np.mean(perm_scores))
        elif cv == "group_kfold":
            cv_obj = StratifiedGroupKFold(n_splits=min(n_splits, len(np.unique(groups))),
                                          shuffle=True, random_state=random_state)
            score, perm_scores, perm_p = permutation_test_score(
                est, X, y, groups=groups, cv=cv_obj, scoring="accuracy",
                n_permutations=n_permutations, random_state=random_state, n_jobs=1,
            )
            perm_chance = float(np.mean(perm_scores))
        # LOO permutation is not group-aware; skip to avoid a misleading p-value.

    ci = _bootstrap_ci(y, preds, seed=random_state) if bootstrap else None

    return ClassificationResult(
        cv=cv, honest=honest, n=len(y), accuracy=float(acc),
        balanced_accuracy=float(bal), auc=float(auc), majority_baseline=float(majority),
        permutation_p=None if perm_p is None else float(perm_p),
        permutation_chance=perm_chance, ci95=ci, per_group=per_group, notes=notes,
    )


def classify_dataframe(
    df: pd.DataFrame,
    *,
    feature_cols=None,
    label_col: str = "label",
    group_col: str = "group",
    dropna: bool = True,
    **kwargs,
) -> ClassificationResult:
    """Run :func:`classify` directly on a feature DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``label_col``; ``group_col`` is used when present (and is
        required for the default subject-wise CV).
    feature_cols : sequence of str, optional
        Columns to use as features. Defaults to all numeric columns except the
        label/group/identifier columns.
    label_col, group_col : str
    dropna : bool
        Drop rows with missing features or label.
    **kwargs
        Forwarded to :func:`classify` (e.g. ``cv``, ``permutation``).
    """
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not in DataFrame")

    reserved = {label_col, group_col, "trial_id", "id"}
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in reserved
        ]
    if not feature_cols:
        raise ValueError("no feature columns found")

    use = df.copy()
    subset = list(feature_cols) + [label_col]
    if dropna:
        use = use.dropna(subset=subset)

    X = use[feature_cols].to_numpy()
    y = use[label_col].to_numpy()
    groups = use[group_col].to_numpy() if group_col in use.columns else None
    return classify(X, y, groups=groups, **kwargs)
