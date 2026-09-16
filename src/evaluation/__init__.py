"""Evaluation engine and metric calculation utilities public API."""

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_ece,
    compute_eer,
    compute_roc_auc_safe,
    find_optimal_threshold,
    fit_temperature_log,
)

__all__ = [
    "ModelEvaluator",
    "compute_classification_metrics",
    "compute_ece",
    "compute_eer",
    "compute_roc_auc_safe",
    "find_optimal_threshold",
    "fit_temperature_log",
]
