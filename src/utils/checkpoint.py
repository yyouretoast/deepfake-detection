"""
Checkpoint loading and calibration helper utilities.
"""

from typing import Dict, Any

DEFAULT_THRESHOLD = 0.01
DEFAULT_TEMPERATURE = 1.4788


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strips 'module.' and '_orig_mod.' prefixes from PyTorch checkpoint keys
    resulting from DDP (DistributedDataParallel) or torch.compile wrappers.
    Ignored keys containing 'lora_'.
    """
    cleaned = {}
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
    """
    Maps calibrated prediction probability to a 50.0% - 100.0% user-facing confidence scale
    relative to the optimal decision threshold.
    """
    if prob > threshold:
        return 50.0 + 50.0 * ((prob - threshold) / (1.0 - threshold)) if threshold < 1.0 else 100.0
    else:
        return 50.0 + 50.0 * ((threshold - prob) / threshold) if threshold > 0.0 else 100.0


def fit_temperature_log(logits: Any, labels: Any) -> float:
    """Fits temperature T = exp(log_T) via NLL optimization using SciPy L-BFGS-B."""
    import numpy as np
    from scipy.optimize import minimize

    logits_arr = np.asarray(logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)

    def nll_func(log_t):
        t = np.exp(log_t[0])
        scaled_logits = logits_arr / t
        # Log-sum-exp trick for binary NLL: log(1 + exp(-y_signed * logit / T))
        y_signed = 2.0 * labels_arr - 1.0
        margin = y_signed * scaled_logits
        loss = np.log1p(np.exp(-np.clip(margin, -50.0, 50.0)))
        return np.mean(loss)

    res = minimize(nll_func, [0.0], method='L-BFGS-B')
    return float(np.exp(res.x[0]))


def compute_ece(probs: Any, targets: Any, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    import numpy as np

    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)

    confidences = np.maximum(probs_arr, 1.0 - probs_arr)
    predictions = (probs_arr >= 0.5).astype(int)
    accuracies = (predictions == targets_arr).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

