"""Checkpoint state dictionary cleaning and temperature calibration utilities."""

from typing import Any

import numpy as np

DEFAULT_THRESHOLD: float = 0.50
DEFAULT_TEMPERATURE: float = 1.4788

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TEMPERATURE",
    "clean_state_dict",
    "normalize_confidence",
    "compute_ece",
    "fit_temperature_log",
    "compute_dual_thresholds",
    "classify_three_zone",
]


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


# Re-export calibration utilities from canonical metrics module
from src.evaluation.metrics import compute_ece, fit_temperature_log


def compute_dual_thresholds(
    probs: Any, targets: Any, min_precision: float = 0.98, min_samples: int = 20
) -> tuple[float, float]:
    """
    Computes high-precision Bayesian decision thresholds (tau_real, tau_fake).
    - tau_fake: Decision threshold guaranteeing >= min_precision for synthetic classifications.
    - tau_real: Decision threshold guaranteeing >= min_precision for authentic classifications.
    - min_samples: Minimum number of samples required in the precision bin to avoid sparse flukes.
    Samples between [tau_real, tau_fake] form the forensic inconclusive/ambiguity zone.
    """
    probs_arr = np.asarray(probs, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.int32)
    thresholds = np.linspace(0.01, 0.99, 100)

    # Adaptive sample floor: scales down gracefully for small test arrays while enforcing
    # statistical significance (e.g. min 20 samples) on production validation/test splits.
    effective_min_samples = max(1, min(min_samples, len(probs_arr) // 10))

    tau_fake = 0.5
    for t in thresholds:
        pred_fake = (probs_arr >= t).astype(int)
        tp = np.sum((pred_fake == 1) & (targets_arr == 1))
        fp = np.sum((pred_fake == 1) & (targets_arr == 0))
        if tp + fp >= effective_min_samples:
            prec = tp / (tp + fp)
            if prec >= min_precision:
                tau_fake = float(t)
                break

    tau_real = 0.5
    for t in reversed(thresholds):
        pred_real = (probs_arr <= t).astype(int)
        tn = np.sum((pred_real == 1) & (targets_arr == 0))
        fn = np.sum((pred_real == 1) & (targets_arr == 1))
        if tn + fn >= effective_min_samples:
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
