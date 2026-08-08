"""
Video temporal aggregation utilities for deepfake detection.

Provides frame-level probability score pooling algorithms for aggregating
per-frame deepfake predictions into a single video-level score.

Functions:
    mean_aggregation: Standard temporal mean.
    top_k_aggregation: Top-K highest confidence frame pooling.
    soft_max_weighted_aggregation: Log-sum-exp stabilized soft-max weighted pooling.
    ema_aggregation: Chronologically sorted exponential moving average.
    aggregate_video_predictions: Unified production API dispatcher.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_scores(
    scores: Union[List[float], List[List[float]], np.ndarray],
) -> np.ndarray:
    """
    Sanitizes and flattens a frame score array.

    Handles nested lists produced by app.py's per-batch `.tolist()` calls
    (e.g. [[0.5], [0.3], ...]) via flatten(). Filters NaN/Inf and clips to [0, 1].

    Args:
        scores: Raw frame probabilities. May be a flat list, nested list, or ndarray.

    Returns:
        1-D float32 ndarray of finite, clipped probability values.
        Returns np.array([], dtype=np.float32) if all values are invalid.
    """
    if scores is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32).flatten()
    valid = arr[np.isfinite(arr)]
    return np.clip(valid, 0.0, 1.0)


def mean_aggregation(
    scores: Union[List[float], List[List[float]], np.ndarray],
) -> float:
    """
    Computes the temporal mean of frame-level fake probabilities.

    Args:
        scores: Frame-level probabilities in [0, 1].

    Returns:
        Mean probability. Returns 0.5 if no valid frames.
    """
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    return float(np.mean(valid))


def top_k_aggregation(
    scores: Union[List[float], List[List[float]], np.ndarray],
    k: int = 5,
) -> float:
    """
    Averages the top-K highest confidence fake frame probabilities.

    Uses np.partition (O(N)) instead of full sort (O(N log N)).
    K is clamped to the number of valid frames if fewer than K exist.

    Args:
        scores: Frame-level probabilities in [0, 1].
        k: Number of top frames to average.

    Returns:
        Mean of top-K probabilities. Returns 0.5 if no valid frames.
    """
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    k_eff = max(1, min(k, len(valid)))
    top_k = np.partition(valid, -k_eff)[-k_eff:]
    return float(np.mean(top_k))


def soft_max_weighted_aggregation(
    scores: Union[List[float], List[List[float]], np.ndarray],
    tau: float = 1.0,
) -> float:
    """
    Computes a soft-max weighted average of frame probabilities.

    High-confidence frames receive exponentially more weight, emphasising
    localised manipulation artefacts without inflating scores on clean videos.
    Log-sum-exp shift applied for numerical stability.

    Args:
        scores: Frame-level probabilities in [0, 1].
        tau: Temperature for soft-max sharpness. Smaller tau → sharper focus
             on the highest-scoring frames.

    Returns:
        Soft-max weighted probability. Returns 0.5 if no valid frames.
    """
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    scaled = valid / max(tau, 1e-8)
    shifted = scaled - np.max(scaled)          # log-sum-exp stability
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return float(np.dot(valid, weights))


def ema_aggregation(
    scores: Union[List[float], List[List[float]], np.ndarray],
    frame_indices: Optional[List[int]] = None,
    alpha: float = 0.3,
) -> float:
    """
    Computes a sequential exponential moving average over chronologically
    ordered video frames.

    If frame_indices are provided the scores are sorted chronologically first.
    Uses a plain Python for loop — for N ≤ 100 video frames the overhead is
    immeasurable and clarity is more valuable than a false vectorization.

    EMA recurrence: S_0 = p_0,  S_t = alpha * p_t + (1 - alpha) * S_{t-1}

    Args:
        scores: Frame-level probabilities in [0, 1].
        frame_indices: Optional integer indices for chronological ordering.
                       Must be the same length as the original scores list.
        alpha: EMA smoothing factor in (0, 1]. Higher → more weight on recent frames.

    Returns:
        Final EMA value. Returns 0.5 if no valid frames.
    """
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5

    if frame_indices is not None:
        raw = np.asarray(scores, dtype=np.float32).flatten()
        idx = np.asarray(frame_indices, dtype=np.int64).flatten()
        # Only sort the frames that survived sanitization (finite values)
        finite_mask = np.isfinite(raw)
        idx = idx[finite_mask]
        order = np.argsort(idx)
        valid = valid[order]

    s = float(valid[0])
    for p in valid[1:]:
        s = alpha * float(p) + (1.0 - alpha) * s
    return s


def aggregate_video_predictions(
    scores: Union[List[float], List[List[float]], np.ndarray],
    method: str = "soft_max",
    k: int = 5,
    alpha: float = 0.3,
    tau: float = 1.0,
    threshold: float = 0.01,
    frame_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Unified production dispatcher for video-level score aggregation.

    Args:
        scores: Frame-level probabilities in [0, 1].
        method: One of 'mean', 'top_k', 'soft_max', 'ema'.
        k: Top-K parameter for 'top_k' method.
        alpha: EMA smoothing factor for 'ema' method.
        tau: Temperature for 'soft_max' method.
        threshold: Decision threshold for is_fake.
        frame_indices: Optional chronological frame indices for 'ema' method.

    Returns:
        Dict with keys:
            video_score (float): Aggregated score in [0, 1].
            is_fake (bool): video_score > threshold.
            aggregation_method (str): Method used.
            valid_frames_count (int): Number of valid frames used.
            threshold_used (float): Decision threshold applied.
    """
    valid = _sanitize_scores(scores)
    valid_count = len(valid)

    if method == "mean":
        score = mean_aggregation(scores)
    elif method == "top_k":
        score = top_k_aggregation(scores, k=k)
    elif method == "soft_max":
        score = soft_max_weighted_aggregation(scores, tau=tau)
    elif method == "ema":
        score = ema_aggregation(scores, frame_indices=frame_indices, alpha=alpha)
    else:
        logger.warning("Unknown aggregation method '%s', defaulting to 'soft_max'.", method)
        score = soft_max_weighted_aggregation(scores, tau=tau)

    return {
        "video_score": score,
        "is_fake": score > threshold,
        "aggregation_method": method,
        "valid_frames_count": valid_count,
        "threshold_used": threshold,
    }
