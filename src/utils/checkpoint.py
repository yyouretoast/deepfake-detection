"""
Checkpoint loading and calibration helper utilities.
"""

from typing import Dict, Any
import torch

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
        if new_k == "freq_extractor.conv_net.0.weight" and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] != 8:
            v = v.repeat(1, max(1, 8 // v.shape[1]), 1, 1)[:, :8, :, :] / float(max(1, 8 // v.shape[1]))
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
