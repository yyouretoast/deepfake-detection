"""Training engine, loss functions, EMA, and optimization utilities public API."""

from src.training.ema import ExponentialMovingAverage
from src.training.loss import FocalLossWithLogits, MaskedBCEWithLogits
from src.training.optimization import create_scheduler, get_differential_param_groups
from src.training.trainer import DualStreamTrainer

__all__ = [
    "DualStreamTrainer",
    "ExponentialMovingAverage",
    "FocalLossWithLogits",
    "MaskedBCEWithLogits",
    "create_scheduler",
    "get_differential_param_groups",
]
