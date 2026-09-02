"""Loss functions for deepfake classification with corrupt sample masking support."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with dynamic positive class weighting and unreduced mask support.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        return focal_factor * bce_loss


class MaskedBCEWithLogits(nn.Module):
    """Standard binary cross-entropy with pos_weight and corrupt sample valid_flag reduction."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid_flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        unreduced = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if valid_flags is not None:
            return (unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
        return unreduced.mean()
