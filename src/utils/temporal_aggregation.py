"""Video temporal aggregation utilities for deepfake detection frame-level score pooling."""

import logging
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_scores(
    scores: Union[list[float], list[list[float]], np.ndarray],
) -> np.ndarray:
    """Sanitizes and flattens frame score input into a clipped 1D float32 array in [0, 1]."""
    if scores is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32).flatten()
    valid = arr[np.isfinite(arr)]
    return np.clip(valid, 0.0, 1.0)


def mean_aggregation(
    scores: Union[list[float], list[list[float]], np.ndarray],
) -> float:
    """Computes temporal mean of frame-level fake probabilities."""
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    return float(np.mean(valid))


def top_k_aggregation(
    scores: Union[list[float], list[list[float]], np.ndarray],
    k: int = 5,
) -> float:
    """Averages the top-K highest confidence fake frame probabilities using partition."""
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    k_eff = max(1, min(k, len(valid)))
    top_k = np.partition(valid, -k_eff)[-k_eff:]
    return float(np.mean(top_k))


def soft_max_weighted_aggregation(
    scores: Union[list[float], list[list[float]], np.ndarray],
    tau: float = 1.0,
) -> float:
    """Computes soft-max weighted average of frame probabilities with log-sum-exp stabilization."""
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5
    scaled = valid / max(tau, 1e-8)
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return float(np.dot(valid, weights))


def ema_aggregation(
    scores: Union[list[float], list[list[float]], np.ndarray],
    frame_indices: Optional[list[int]] = None,
    alpha: float = 0.3,
) -> float:
    """Computes sequential EMA over chronologically ordered video frames (S_t = alpha*p_t + (1-alpha)*S_{t-1})."""
    valid = _sanitize_scores(scores)
    if len(valid) == 0:
        return 0.5

    if frame_indices is not None:
        raw = np.asarray(scores, dtype=np.float32).flatten()
        idx = np.asarray(frame_indices, dtype=np.int64).flatten()
        finite_mask = np.isfinite(raw)
        idx = idx[finite_mask]
        order = np.argsort(idx)
        valid = valid[order]

    s = float(valid[0])
    for p in valid[1:]:
        s = alpha * float(p) + (1.0 - alpha) * s
    return s


def aggregate_video_predictions(
    scores: Union[list[float], list[list[float]], np.ndarray],
    method: str = "soft_max",
    k: int = 5,
    alpha: float = 0.3,
    tau: float = 1.0,
    threshold: float = 0.01,
    frame_indices: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Unified production dispatcher for video-level score aggregation."""
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
