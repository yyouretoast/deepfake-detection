"""Optimization parameter groups and learning rate scheduler builders."""

from typing import Any
import torch
import torch.nn as nn


def get_differential_param_groups(
    model: nn.Module,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-2,
) -> list[dict[str, Any]]:
    """
    Splits parameters into differential learning rate groups:
      Group 0: Spatial Backbone parameters (lower LR for pretrained feature preservation).
      Group 1: Head & Frequency branch weights with standard weight decay.
      Group 2: Biases and normalization layers with zero weight decay.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "spatial_backbone" in name:
            backbone_params.append(param)
        elif name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": decay_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr_head, "weight_decay": 0.0},
    ]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 1,
    total_epochs: int = 5,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Builds a Sequential Warmup + Cosine Annealing learning rate scheduler."""
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
