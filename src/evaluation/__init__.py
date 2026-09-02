"""Evaluation engine and metric calculation utilities public API."""

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_ece,
    compute_roc_auc_safe,
    fit_temperature_log,
)

__all__ = [
    "ModelEvaluator",
    "compute_roc_auc_safe",
    "compute_classification_metrics",
    "compute_ece",
    "fit_temperature_log",
]
