"""Steganographic and constrained convolutional filters for noise residual extraction."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMConv2d(nn.Module):
    """Spatial Rich Model (SRM) fixed high-pass filter bank for noise residual extraction."""

    def __init__(self) -> None:
        super().__init__()
        srm1 = (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            / 4.0
        )
        srm2 = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )
        srm3 = (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            / 2.0
        )

        filters = np.stack([srm1, srm2, srm3], axis=0)[:, np.newaxis, :, :]  # [3, 1, 5, 5]
        filters = np.tile(filters, (3, 1, 1, 1))  # [9, 1, 5, 5]
        self.register_buffer("weights", torch.from_numpy(filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, 3, H, W]
        Output: SRM residual maps [B, 9, H, W]
        """
        w = (
            self.weights
            if self.weights.device == x.device and self.weights.dtype == x.dtype
            else self.weights.to(dtype=x.dtype, device=x.device)
        )
        return F.conv2d(x, w, stride=1, padding=2, groups=3)


class BayarConv2d(nn.Module):
    """Bayar-Stamm adaptive constrained convolution layer for forgery artifact detection."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 5, 5))

    def _get_constrained_kernel(self) -> torch.Tensor:
        w = self.kernel  # [out_c, in_c, 5, 5]
        mask = torch.ones_like(w)
        mask[:, :, 2, 2] = 0.0
        w_masked = w * mask
        sum_w = w_masked.sum(dim=(2, 3), keepdim=True)
        sign_w = torch.sign(sum_w)
        sign_w = torch.where(sign_w == 0, torch.ones_like(sign_w), sign_w)
        sum_w_safe = sign_w * sum_w.abs().clamp(min=1e-5)
        w_norm = w_masked / sum_w_safe

        center_mask = torch.zeros_like(w)
        center_mask[:, :, 2, 2] = -1.0
        return w_norm + center_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, 3, H, W]
        Output: Bayar residual map [B, 1, H, W]
        """
        w_constrained = self._get_constrained_kernel().to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, w_constrained, stride=1, padding=2)
