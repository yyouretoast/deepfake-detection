"""Exponential Moving Average (EMA) of model parameter weights."""

from contextlib import contextmanager
from typing import Generator
import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters with context manager support."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model parameters: s = decay * s + (1 - decay) * p."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data.to(dtype=self.shadow[name].dtype, device=self.shadow[name].device), 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Copy shadow weights to model, returning backup of current parameters."""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Generator[None, None, None]:
        """Context manager temporarily applying EMA shadow weights for validation or checkpointing."""
        backup = self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore(model, backup)
