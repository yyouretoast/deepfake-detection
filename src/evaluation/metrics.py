"""Evaluation metrics, calibration error, and safe statistical score calculations."""

from collections.abc import Sequence
from typing import Any, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def compute_roc_auc_safe(
    y_true: Union[np.ndarray, Sequence[float]],
    y_score: Union[np.ndarray, Sequence[float]],
    fallback: float = 0.5,
) -> float:
    """Computes ROC AUC score defensively, returning fallback if fewer than 2 distinct classes exist."""
    y_true_arr = np.asarray(y_true).flatten()
    y_score_arr = np.asarray(y_score).flatten()

    if len(np.unique(y_true_arr)) < 2:
        return fallback
    try:
        return float(roc_auc_score(y_true_arr, y_score_arr))
    except (ValueError, TypeError, RuntimeError):
        return fallback


def compute_classification_metrics(
    y_true: Union[np.ndarray, Sequence[float]],
    y_prob: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Computes binary classification metrics: AUC, F1, precision, recall, and accuracy."""
    y_true_arr = np.asarray(y_true).flatten()
    y_prob_arr = np.asarray(y_prob).flatten()
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    auc_val = compute_roc_auc_safe(y_true_arr, y_prob_arr)
    f1_val = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    prec_val = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec_val = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    acc_val = float(np.mean(y_true_arr == y_pred_arr))

    return {
        "auc": auc_val,
        "f1": f1_val,
        "precision": prec_val,
        "recall": rec_val,
        "accuracy": acc_val,
    }


def compute_ece(probs: Any, targets: Any, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    probs_arr = np.asarray(probs, dtype=np.float64).flatten()
    targets_arr = np.asarray(targets, dtype=np.float64).flatten()

    confidences = np.maximum(probs_arr, 1.0 - probs_arr)
    predictions = (probs_arr >= 0.5).astype(int)
    accuracies = (predictions == targets_arr).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == 0:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def fit_temperature_log(logits: Any, labels: Any) -> float:
    """Fits temperature scale T = exp(log_T) via NLL optimization using SciPy L-BFGS-B."""
    logits_arr = np.asarray(logits, dtype=np.float64).flatten()
    labels_arr = np.asarray(labels, dtype=np.float64).flatten()

    def nll_func(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        scaled_logits = logits_arr / t
        y_signed = 2.0 * labels_arr - 1.0
        margin = y_signed * scaled_logits
        loss = np.log1p(np.exp(-np.clip(margin, -50.0, 50.0)))
        return float(np.mean(loss))

    res = minimize(nll_func, [0.0], method="L-BFGS-B")
    return float(np.exp(res.x[0]))


def compute_eer(
    y_true: Union[np.ndarray, Sequence[float]],
    y_score: Union[np.ndarray, Sequence[float]],
) -> tuple[float, float]:
    """Computes Equal Error Rate (EER) and the corresponding decision threshold."""
    y_true_arr = np.asarray(y_true).flatten()
    y_score_arr = np.asarray(y_score).flatten()

    if len(np.unique(y_true_arr)) < 2:
        return 0.5, 0.5

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx])
    return eer, thresh
