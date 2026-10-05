"""Validate row identity before splitting, joining, or evaluating predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty table")


def unique_patients(frame):
    require_columns(frame, ["patient_id"])
    ids = frame["patient_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Patient IDs must be nonempty")
    if ids.astype(str).duplicated().any():
        raise ValueError(
            "Duplicate patient IDs; resolve their provenance before training or merging"
        )


def binary_labels(labels):
    y = np.asarray(labels, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a nonempty vector of binary 0/1 values")
    return y


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return p


def metrics(labels, preds):
    y, p = binary_labels(labels), probabilities(preds)
    if p.ndim != 1 or len(y) != len(p):
        raise ValueError("Expected one prediction per label")
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "n": len(y),
    }


def patient_folds(frame, n_splits=5, seed=42):
    unique_patients(frame)
    require_columns(frame, ["label"])
    y = binary_labels(frame.label)
    counts = np.bincount(y.astype(int), minlength=2)
    if n_splits < 2 or counts.min() < n_splits:
        raise ValueError("Each class needs at least n_splits unique patients")
    # Sort first: changing manifest row order must not change fold membership.
    ordered = frame.assign(patient_id=frame.patient_id.astype(str)).sort_values("patient_id")
    folds = ordered[["patient_id", "label"]].copy()
    folds["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(splitter.split(ordered, ordered.label)):
        folds.iloc[valid, folds.columns.get_loc("fold")] = fold
    return folds.reset_index(drop=True)


def align_predictions(tables):
    """Strict one-to-one alignment. Never silently drop patients or keep first duplicates."""
    if not tables:
        raise ValueError("At least one prediction table is required")
    aligned = []
    base = None
    for name, frame in tables.items():
        unique_patients(frame)
        require_columns(frame, ["label", "pred_probability"])
        frame = frame.copy()
        frame["patient_id"] = frame.patient_id.astype(str)
        frame = frame.set_index("patient_id").sort_index()
        binary_labels(frame.label)
        probabilities(frame.pred_probability)
        if base is None:
            base = frame
        else:
            if not base.index.equals(frame.index):
                raise ValueError(f"Patient set mismatch in {name}")
            if not np.array_equal(base.label, frame.label):
                raise ValueError(f"Label mismatch in {name}")
            if "fold" in base and "fold" in frame and not np.array_equal(base.fold, frame.fold):
                raise ValueError(f"Fold mismatch in {name}; use a shared fold manifest")
        aligned.append(frame.pred_probability.to_numpy())
    return base.index.to_numpy(), base.label.to_numpy(), np.column_stack(aligned)


def submission(patient_ids, preds):
    out = pd.DataFrame({"patient_id": patient_ids, "is_pathologic": probabilities(preds)})
    unique_patients(out)
    return out.rename(columns={"patient_id": "uid"})
