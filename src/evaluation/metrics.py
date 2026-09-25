"""Evaluation metrics, calibration error, and safe statistical score calculations."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def compute_roc_auc_safe(
    y_true: np.ndarray | Sequence[float],
    y_score: np.ndarray | Sequence[float],
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

    bounds = [(float(np.log(0.1)), float(np.log(10.0)))]
    res = minimize(nll_func, [0.0], method="L-BFGS-B", bounds=bounds)
    fitted_t = float(np.exp(res.x[0]))
    return float(np.clip(fitted_t, 0.1, 10.0))


def fit_platt_scaling(
    logits: Any, labels: Any, sample_weights: Any | None = None
) -> tuple[float, float, float]:
    """Fits 2-parameter Platt scaling p = sigmoid(a * z + b) to absorb balanced-sampling prior shift.

    Supports optional sample_weights for stratified calibration on skewed validation cohorts.

    Returns:
        tuple (scale_a, bias_b, effective_temperature_T)
        where T = 1.0 / max(scale_a, 1e-6)
    """
    logits_arr = np.asarray(logits, dtype=np.float64).flatten()
    labels_arr = np.asarray(labels, dtype=np.float64).flatten()
    if sample_weights is not None:
        weights_arr = np.asarray(sample_weights, dtype=np.float64).flatten()
        weights_arr = weights_arr / max(1e-12, float(np.sum(weights_arr)))
    else:
        weights_arr = None

    def nll_func(params: np.ndarray) -> float:
        a = float(params[0])
        b = float(params[1])
        calibrated = a * logits_arr + b
        y_signed = 2.0 * labels_arr - 1.0
        margin = y_signed * calibrated
        loss = np.log1p(np.exp(-np.clip(margin, -50.0, 50.0)))
        if weights_arr is not None:
            return float(np.sum(weights_arr * loss))
        return float(np.mean(loss))

    # a bounded to [0.1, 10.0] (corresponding to T in [0.1, 10.0]), b in [-10.0, 10.0]
    bounds = [(0.1, 10.0), (-10.0, 10.0)]
    res = minimize(nll_func, [1.0, 0.0], method="L-BFGS-B", bounds=bounds)
    a_fit = float(np.clip(res.x[0], 0.1, 10.0))
    b_fit = float(np.clip(res.x[1], -10.0, 10.0))
    eff_t = float(1.0 / max(a_fit, 1e-6))
    return a_fit, b_fit, eff_t


def compute_eer(
    y_true: np.ndarray | Sequence[float],
    y_score: np.ndarray | Sequence[float],
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


def find_optimal_threshold(
    y_true: np.ndarray | Sequence[float],
    y_prob: np.ndarray | Sequence[float],
    criterion: str = "balanced_accuracy",
    n_thresholds: int = 81,
) -> tuple[float, float]:
    """
    Finds the optimal decision threshold tau* maximizing a chosen performance criterion.

    Supported criteria:
    - 'balanced_accuracy' / 'youden_grid': Grid search maximizing Balanced Accuracy (Youden's J = 2*BA - 1).
    - 'macro_f1': Grid search maximizing macro-averaged F1 score.
    - 'f1': Grid search maximizing positive-class binary F1 score.
    - 'youden_roc': Continuous ROC-based Youden's J statistic (TPR - FPR).

    Returns:
        (optimal_threshold, best_score)
    """
    from sklearn.metrics import balanced_accuracy_score

    y_true_arr = np.asarray(y_true).flatten().astype(int)
    y_prob_arr = np.asarray(y_prob).flatten().astype(float)

    if len(np.unique(y_true_arr)) < 2:
        return 0.5, 0.5

    if criterion == "youden_roc":
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
        valid_idx = np.isfinite(thresholds)
        if not np.any(valid_idx):
            return 0.5, 0.0
        j_scores = tpr[valid_idx] - fpr[valid_idx]
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(np.clip(thresholds[valid_idx][best_idx], 0.01, 0.99))
        return best_thresh, float(j_scores[best_idx])

    threshold_grid = np.linspace(0.1, 0.9, n_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for t in threshold_grid:
        preds = (y_prob_arr >= t).astype(int)
        if criterion in ("balanced_accuracy", "youden_grid"):
            score = float(balanced_accuracy_score(y_true_arr, preds))
        elif criterion == "macro_f1":
            score = float(f1_score(y_true_arr, preds, average="macro", zero_division=0))
        elif criterion == "f1":
            score = float(f1_score(y_true_arr, preds, zero_division=0))
        else:
            raise ValueError(f"Unknown threshold optimization criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score


def calibrate_probabilities_balanced(
    logits: Any,
    temp: float = 1.0,
    platt_a: float | None = None,
    platt_b: float | None = None,
    cal_prevalence: float | None = None,
    target_prevalence: float = 0.50,
) -> np.ndarray:
    """
    Computes prevalence-invariant calibrated probabilities.
    If Platt parameters (a, b) and calibration dataset prevalence (cal_prevalence) are provided,
    applies Saerens et al. (2002) prior-shift subtraction to eliminate majority-class bias.
    Otherwise, applies pure monotonic temperature scaling (z / T*) without intercept distortion.
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    if platt_a is not None and platt_b is not None and cal_prevalence is not None and 0.0 < cal_prevalence < 1.0:
        z_platt = float(platt_a) * logits_arr + float(platt_b)
        prior_shift = np.log(cal_prevalence / (1.0 - cal_prevalence))
        target_shift = np.log(target_prevalence / (1.0 - target_prevalence))
        z_balanced = z_platt - prior_shift + target_shift
        return 1.0 / (1.0 + np.exp(-np.clip(z_balanced, -50.0, 50.0)))

    t = max(float(temp), 1e-4)
    return 1.0 / (1.0 + np.exp(-np.clip(logits_arr / t, -50.0, 50.0)))

