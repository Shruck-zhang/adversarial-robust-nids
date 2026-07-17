"""Imbalance-aware, per-class evaluation metrics for the NIDS.

Because the data is heavily benign-dominated, every model is reported with
per-class and imbalance-aware metrics, not headline accuracy alone. This module
computes, for any classifier's predictions:

  * overall: accuracy, balanced accuracy, macro/weighted F1, Matthews corr. coef.
  * per-class: precision, recall (detection rate), F1, support, and the
    false-positive rate (FPR) — the operational cost of false alarms
  * optional (given class probabilities): per-class PR-AUC (average precision)
    and one-vs-rest ROC-AUC, which are the right ranking metrics under imbalance

Everything is dataset-agnostic: pass integer labels + class names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, average_precision_score,
    roc_auc_score,
)


def per_class_fpr(cm: np.ndarray) -> np.ndarray:
    """False-positive rate per class from a confusion matrix:
    FPR_c = FP_c / (FP_c + TN_c), where negatives are all other classes."""
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
    return fpr


def evaluate(y_true, y_pred, class_names, y_score=None) -> dict:
    """Full imbalance-aware evaluation. `y_score` (n, n_classes) enables PR-AUC/ROC-AUC."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fpr = per_class_fpr(cm)

    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = cm.sum(axis=1)

    per_class = pd.DataFrame({
        "class": class_names,
        "support": support,
        "precision": prec,
        "recall": rec,          # = detection rate / TPR
        "f1": f1,
        "fpr": fpr,
    })

    if y_score is not None:
        y_score = np.asarray(y_score)
        Y = np.eye(len(class_names))[y_true]
        ap, auc = [], []
        for c in range(len(class_names)):
            present = Y[:, c].sum() > 0
            ap.append(average_precision_score(Y[:, c], y_score[:, c]) if present else np.nan)
            try:
                auc.append(roc_auc_score(Y[:, c], y_score[:, c]) if present else np.nan)
            except ValueError:
                auc.append(np.nan)
        per_class["pr_auc"] = ap
        per_class["roc_auc"] = auc

    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_fpr": float(np.nanmean(fpr)),
    }
    if y_score is not None:
        overall["macro_pr_auc"] = float(np.nanmean(per_class["pr_auc"]))

    return {"overall": overall, "per_class": per_class, "confusion": cm}


def summary_row(name: str, res: dict) -> dict:
    """One flat row per model for a cross-model comparison table."""
    row = {"model": name}
    row.update(res["overall"])
    return row


def comparison_table(results: dict) -> pd.DataFrame:
    """results: {model_name: evaluate(...) dict} -> tidy comparison DataFrame."""
    df = pd.DataFrame([summary_row(n, r) for n, r in results.items()])
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def rare_class_report(res: dict, support_threshold: int) -> pd.DataFrame:
    """Isolate the rare classes (support below threshold) — the thesis's O1/O4 focus."""
    pc = res["per_class"]
    return pc[pc["support"] < support_threshold].reset_index(drop=True)
