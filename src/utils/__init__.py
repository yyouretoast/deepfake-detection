"""Checkpointing, model interpretability, and post-processing public API."""

from src.utils.checkpoint import (
    clean_state_dict,
    compute_ece,
    fit_temperature_log,
    load_detector_checkpoint,
    normalize_confidence,
)
from src.utils.interpretability import (
    ConvNeXtGradCAM,
    generate_face_diagnostics,
)
from src.utils.temporal_aggregation import (
    aggregate_video_predictions,
    ema_aggregation,
    mean_aggregation,
    soft_max_weighted_aggregation,
    top_k_aggregation,
)
from src.utils.visualization import (
    render_temporal_anomaly_timeline,
)

__all__ = [
    "clean_state_dict",
    "load_detector_checkpoint",
    "normalize_confidence",
    "fit_temperature_log",
    "compute_ece",
    "ConvNeXtGradCAM",
    "generate_face_diagnostics",
    "aggregate_video_predictions",
    "mean_aggregation",
    "top_k_aggregation",
    "soft_max_weighted_aggregation",
    "ema_aggregation",
    "render_temporal_anomaly_timeline",
]
