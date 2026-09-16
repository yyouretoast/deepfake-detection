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
    "ConvNeXtGradCAM",
    "aggregate_video_predictions",
    "clean_state_dict",
    "compute_ece",
    "ema_aggregation",
    "fit_temperature_log",
    "generate_face_diagnostics",
    "load_detector_checkpoint",
    "mean_aggregation",
    "normalize_confidence",
    "render_temporal_anomaly_timeline",
    "soft_max_weighted_aggregation",
    "top_k_aggregation",
]
