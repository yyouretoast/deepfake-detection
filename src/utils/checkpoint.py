"""Checkpoint state dictionary cleaning and temperature calibration utilities."""

from typing import Any

import numpy as np
from scipy.optimize import minimize

DEFAULT_THRESHOLD: float = 0.50
DEFAULT_TEMPERATURE: float = 1.4788


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strips 'module.' and '_orig_mod.' prefixes from checkpoint keys."""
    cleaned: dict[str, Any] = {}
    for k, v in state_dict.items():
        if "lora_" in k:
            continue
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[7:]
        if new_k.startswith("_orig_mod."):
            new_k = new_k[10:]
        cleaned[new_k] = v
    return cleaned


def normalize_confidence(prob: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """Maps prediction probability to a 50.0% - 100.0% confidence scale relative to decision threshold."""
    prob_val = float(np.clip(prob, 0.0, 1.0))
    thresh = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    if prob_val >= thresh:
        return 50.0 + 50.0 * ((prob_val - thresh) / (1.0 - thresh))
    return 50.0 + 50.0 * ((thresh - prob_val) / thresh)


def fit_temperature_log(logits: Any, labels: Any) -> float:
    """Fits temperature scale T = exp(log_T) via NLL optimization using SciPy L-BFGS-B."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)

    def nll_func(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        scaled_logits = logits_arr / t
        y_signed = 2.0 * labels_arr - 1.0
        margin = y_signed * scaled_logits
        loss = np.log1p(np.exp(-np.clip(margin, -50.0, 50.0)))
        return float(np.mean(loss))

    res = minimize(nll_func, [0.0], method="L-BFGS-B")
    return float(np.exp(res.x[0]))


def compute_ece(probs: Any, targets: Any, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)

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


def compute_dual_thresholds(
    probs: Any, targets: Any, min_precision: float = 0.98
) -> tuple[float, float]:
    """
    Computes high-precision Bayesian decision thresholds (tau_real, tau_fake).
    - tau_fake: Decision threshold guaranteeing >= min_precision for synthetic classifications.
    - tau_real: Decision threshold guaranteeing >= min_precision for authentic classifications.
    Samples between [tau_real, tau_fake] form the forensic inconclusive/ambiguity zone.
    """
    probs_arr = np.asarray(probs, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.int32)
    thresholds = np.linspace(0.01, 0.99, 100)

    tau_fake = 0.5
    for t in thresholds:
        pred_fake = (probs_arr >= t).astype(int)
        tp = np.sum((pred_fake == 1) & (targets_arr == 1))
        fp = np.sum((pred_fake == 1) & (targets_arr == 0))
        if tp + fp > 0:
            prec = tp / (tp + fp)
            if prec >= min_precision:
                tau_fake = float(t)
                break

    tau_real = 0.5
    for t in reversed(thresholds):
        pred_real = (probs_arr <= t).astype(int)
        tn = np.sum((pred_real == 1) & (targets_arr == 0))
        fn = np.sum((pred_real == 1) & (targets_arr == 1))
        if tn + fn > 0:
            prec = tn / (tn + fn)
            if prec >= min_precision:
                tau_real = float(t)
                break

    if tau_real > tau_fake:
        tau_real, tau_fake = 0.40, 0.60

    return float(tau_real), float(tau_fake)


def classify_three_zone(
    prob: float, tau_real: float = 0.40, tau_fake: float = 0.60
) -> dict[str, Any]:
    """
    Classifies prediction probability into three forensic certainty zones:
    1. Confirmed Real (prob <= tau_real)
    2. Inconclusive / Perturbation Detected (tau_real < prob < tau_fake)
    3. Confirmed Fake (prob >= tau_fake)
    """
    p = float(np.clip(prob, 0.0, 1.0))
    if p >= tau_fake:
        return {
            "verdict": "Confirmed Synthetic",
            "zone": "high_confidence_fake",
            "confidence": float(p),
            "is_inconclusive": False,
        }
    if p <= tau_real:
        return {
            "verdict": "Confirmed Authentic",
            "zone": "high_confidence_real",
            "confidence": float(1.0 - p),
            "is_inconclusive": False,
        }
    return {
        "verdict": "Inconclusive (Heavy Compression / Perturbation Detected)",
        "zone": "ambiguity_zone",
        "confidence": float(max(p, 1.0 - p)),
        "is_inconclusive": True,
    }
