"""
Production-Grade Video Temporal Aggregation Utilities for Deepfake Detection.

Provides logit/probability space frame score pooling algorithms:
  - mean_aggregation: Standard temporal expectation
  - top_k_aggregation: Top-K highest confidence fake frame pooling
  - soft_max_weighted_aggregation: Soft-Max weighted probability pooling
  - ema_aggregation: Chronologically sorted exponential moving average
  - aggregate_video_predictions: Unified production API wrapper
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _sanitize_scores(
    scores: Union[List[float], np.ndarray],
    is_logits: bool = False,
    temperature: float = 1.4788
) -> np.ndarray:
    """
    Cleans score array, filters NaN/Inf, and converts logits to probabilities if requested.
    """
    if scores is None or len(scores) == 0:
        return np.array([], dtype=np.float32)

    arr = np.array(scores, dtype=np.float32).flatten()
    valid_mask = np.isfinite(arr)
    valid_arr = arr[valid_mask]

    if len(valid_arr) == 0:
        return np.array([], dtype=np.float32)

    if is_logits:
        # Convert logits to calibrated probabilities via Temperature Scaling
        scaled_logits = valid_arr / max(1e-5, temperature)
        valid_arr = 1.0 / (1.0 + np.exp(-scaled_logits))

    # Clamp probabilities to [0, 1] bounds
    return np.clip(valid_arr, 0.0, 1.0)

def mean_aggregation(
    scores: Union[List[float], np.ndarray],
    is_logits: bool = False,
    temperature: float = 1.4788
) -> float:
    """Computes standard temporal mean across valid frame predictions."""
    clean = _sanitize_scores(scores, is_logits=is_logits, temperature=temperature)
    if len(clean) == 0:
        return 0.5
    return float(np.mean(clean))

def top_k_aggregation(
    scores: Union[List[float], np.ndarray],
    k: int = 5,
    is_logits: bool = False,
    temperature: float = 1.4788
) -> float:
    """Averages the top-K highest confidence fake frame scores."""
    clean = _sanitize_scores(scores, is_logits=is_logits, temperature=temperature)
    if len(clean) == 0:
        return 0.5
    k_effective = max(1, min(k, len(clean)))
    sorted_scores = np.sort(clean)[::-1]
    return float(np.mean(sorted_scores[:k_effective]))

def soft_max_weighted_aggregation(
    scores: Union[List[float], np.ndarray],
    tau: float = 1.0,
    is_logits: bool = False,
    temperature: float = 1.4788
) -> float:
    """
    Computes Soft-Max Temperature Weighting to emphasize high-confidence forgery anomalies
    without artificially inflating scores on 100% real videos.
    """
    clean = _sanitize_scores(scores, is_logits=is_logits, temperature=temperature)
    if len(clean) == 0:
        return 0.5
    
    # Softmax weights over score magnitude
    scaled = clean / max(1e-5, tau)
    exp_scores = np.exp(scaled - np.max(scaled))  # Log-sum-exp stability trick
    weights = exp_scores / np.sum(exp_scores)
    return float(np.sum(clean * weights))

def ema_aggregation(
    scores: Union[List[float], np.ndarray],
    frame_indices: Optional[List[int]] = None,
    alpha: float = 0.3,
    is_logits: bool = False,
    temperature: float = 1.4788
) -> float:
    """
    Computes exponential moving average (EMA) across chronologically sorted video frames.
    """
    clean = _sanitize_scores(scores, is_logits=is_logits, temperature=temperature)
    if len(clean) == 0:
        return 0.5

    if frame_indices is not None and len(frame_indices) == len(scores):
        indices = np.array(frame_indices).flatten()
        valid_mask = np.isfinite(np.array(scores, dtype=np.float32).flatten())
        indices = indices[valid_mask]
        sort_order = np.argsort(indices)
        clean = clean[sort_order]

    ema = clean[0]
    for s in clean[1:]:
        ema = alpha * s + (1.0 - alpha) * ema
    return float(ema)

def aggregate_video_predictions(
    scores: Union[List[float], np.ndarray],
    method: str = "top_k",
    k: int = 5,
    alpha: float = 0.3,
    tau: float = 1.0,
    threshold: float = 0.01,
    is_logits: bool = False,
    temperature: float = 1.4788,
    frame_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Unified production wrapper for video-level score aggregation.
    """
    clean = _sanitize_scores(scores, is_logits=is_logits, temperature=temperature)
    valid_count = len(clean)

    if method == "mean":
        score = mean_aggregation(clean, is_logits=False)
    elif method == "top_k":
        score = top_k_aggregation(clean, k=k, is_logits=False)
    elif method == "soft_max":
        score = soft_max_weighted_aggregation(clean, tau=tau, is_logits=False)
    elif method == "ema":
        score = ema_aggregation(clean, frame_indices=frame_indices, alpha=alpha, is_logits=False)
    else:
        logging.warning(f"Unknown aggregation method '{method}', defaulting to 'top_k'.")
        score = top_k_aggregation(clean, k=k, is_logits=False)

    return {
        "video_score": score,
        "is_fake": score > threshold,
        "aggregation_method": method,
        "valid_frames_count": valid_count,
        "threshold_used": threshold
    }
