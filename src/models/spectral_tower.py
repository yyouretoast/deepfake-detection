"""Residual Squeeze-and-Excitation Spectral Tower for 20-channel FFT magnitude/phase features."""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for heterogeneous frequency/phase bands."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class SpectralResBlock(nn.Module):
    """Residual convolutional block with SE channel attention for 2D FFT features."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)


class ResSESpectralTower(nn.Module):
    """
    4-Stage Residual Spectral Tower for 20-channel Magnitude/Phase inputs.
    Preserves radial and angular concentric frequency rings through progressive downsampling.
    Includes a dedicated auxiliary classifier head to eliminate gradient starvation.
    """

    def __init__(
        self,
        in_channels: int = 20,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        # Stem: downsample from 256x256 to 128x128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        # Stage 1: Central Low-Frequency Rings [c1, 128, 128]
        self.stage1 = SpectralResBlock(c1, c1, stride=1)
        # Stage 2: Intermediate Texture Annulus [c2, 64, 64]
        self.stage2 = SpectralResBlock(c1, c2, stride=2)
        # Stage 3: High-Frequency Boundary Seams [c3, 32, 32]
        self.stage3 = SpectralResBlock(c2, c3, stride=2)
        # Stage 4: Nyquist Checkerboard Harmonics [c4, 16, 16]
        self.stage4 = SpectralResBlock(c3, c4, stride=2)

        # Global Pooling and Projection Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c4, embed_dim),
            nn.ReLU(inplace=True),
        )
        # Dedicated Auxiliary Classifier Head
        self.aux_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        feat = self.fc(self.pool(out).flatten(1))
        aux_logit = self.aux_head(feat)
        return feat, aux_logit
