"""Feature fusion modules and classification heads for dual-stream architectures."""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization for spatial convolutional feature maps [B, C, H, W]."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class GatedResidualFusion(nn.Module):
    """
    Symmetric Gated Residual Fusion:
      g = Sigmoid(Linear([f_s || f_f])) in R^512
      f_fused = [(1 - g) * f_s || g * f_f] in R^1024
    """

    def __init__(self, in_features: int = 1024, out_features: int = 512) -> None:
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concat_feat = torch.cat([f_s, f_f], dim=1)
        gate = self.gate_fc(concat_feat)
        fused = torch.cat([f_s * (1.0 - gate), f_f * gate], dim=1)
        return fused, gate


class ClassificationHead(nn.Module):
    """Standard 2-layer MLP classifier head with ReLU and Dropout."""

    def __init__(self, in_features: int = 1024, hidden_features: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
