"""
Dual-Stream Hybrid Deepfake Detector Engine.

Spatial Stream: ConvNeXt-Small backbone extracted feature vector f_s in R^512.
Frequency Stream: SRM spatial residual filters (9-ch) + Bayar constrained conv (1-ch) -> 10-ch noise residuals.
2D FFT Spectral Decomposition: Magnitude & Phase maps (20-ch) -> freq ConvNeXt-style projection f_f in R^512.
Gated Residual Fusion:
    g = Sigmoid(Linear([f_s || f_f])) in R^512
    f_fused = [f_s || (f_f (o) g)] in R^1024
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SRMConv2d(nn.Module):
    """Spatial Rich Model (SRM) fixed high-pass filter bank for noise residual extraction."""

    def __init__(self) -> None:
        super().__init__()
        srm1 = np.array(
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        ) / 4.0
        srm2 = np.array(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            dtype=np.float32,
        ) / 12.0
        srm3 = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        ) / 2.0

        filters = np.stack([srm1, srm2, srm3], axis=0)[:, np.newaxis, :, :]  # [3, 1, 5, 5]
        filters = np.tile(filters, (3, 1, 1, 1))  # [9, 1, 5, 5]
        self.register_buffer("weights", torch.from_numpy(filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, 3, H, W]
        Output: SRM residual maps [B, 9, H, W]
        """
        w = self.weights if self.weights.device == x.device and self.weights.dtype == x.dtype else self.weights.to(dtype=x.dtype, device=x.device)
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


class RealFFT2DModule(nn.Module):
    """2D FFT spectral decomposition into log-magnitude and normalized phase components."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, C, H, W]  (e.g., C=10 noise maps)
        Output: Spectral features [B, 2*C, H, W] (e.g., C_out=20 concatenated mag & phase)
        """
        device_type = x.device.type if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            fft = torch.fft.fft2(x_fp32, norm="ortho")
            fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
            eps = 1e-6
            abs_fft = torch.abs(fft_shift)
            mag = torch.log1p(torch.clamp(abs_fft, min=1e-7))

            mask = abs_fft < eps
            safe_real = torch.where(mask, torch.ones_like(fft_shift.real), fft_shift.real)
            safe_imag = torch.where(mask, torch.zeros_like(fft_shift.imag), fft_shift.imag)
            phase_raw = torch.atan2(safe_imag, safe_real) / torch.pi
            phase = torch.where(mask, torch.zeros_like(phase_raw), phase_raw)

            mag = torch.nan_to_num(mag, nan=0.0, posinf=10.0, neginf=-10.0)
            phase = torch.nan_to_num(phase, nan=0.0, posinf=1.0, neginf=-1.0)
            out_fp32 = torch.cat([mag, phase], dim=1)
        return out_fp32.to(dtype=x.dtype, device=x.device)


class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector.
    Fuses Spatial Stream (ConvNeXt-Small, 512-d) and Frequency Stream (SRM + Bayar 2D FFT, 512-d)
    via Sigmoid Gated Residual Fusion: f_fused = [f_s || (f_f (o) g)] in R^1024.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        use_fft_branch: bool = True,
        dropout: float = 0.3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if config is not None and isinstance(config, dict) and "model" in config:
            model_cfg = config["model"]
            backbone_name = model_cfg.get("backbone", backbone_name)
            pretrained = model_cfg.get("pretrained", pretrained)
            use_fft_branch = model_cfg.get("use_fft_branch", use_fft_branch)
            dropout = model_cfg.get("dropout", dropout)

        self.use_fft_branch = use_fft_branch

        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_small(weights=weights)
        self.spatial_backbone = convnext.features
        # Extract the LayerNorm2d that ConvNeXt applies before its classifier head.
        # Omitting it leaves spatial features unscaled, which causes volatile gradients
        # and poor convergence, especially when fused with the normalized frequency branch.
        self.spatial_norm = convnext.classifier[0]  # nn.LayerNorm2d(768)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_fc = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if self.use_fft_branch:
            self.srm = SRMConv2d()
            self.bayar = BayarConv2d(in_channels=3, out_channels=1)
            self.fft = RealFFT2DModule()
            self.freq_conv = nn.Sequential(
                nn.Conv2d(20, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.freq_fc = nn.Sequential(nn.Linear(128, 512), nn.ReLU())
            self.gate_fc = nn.Sequential(nn.Linear(1024, 512), nn.Sigmoid())
            classifier_in = 1024
        else:
            classifier_in = 512

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 4D image input tensor [B, 3, H, W].
        Returns unscaled classification logits [B, 1].
        """
        if self.training and not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if (x < -0.1).any() or (x > 1.1).any():
                import logging
                logging.getLogger(__name__).warning(
                    "Input tensor x has values outside [0, 1] range: min=%.3f, max=%.3f. "
                    "SRM and Bayar filters expect unnormalized [0, 1] inputs.",
                    float(x.min()), float(x.max())
                )

        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x_spatial = (x - mean) / std
        # Apply backbone features then LayerNorm2d (critical for stable distributions)
        feat_maps = self.spatial_backbone(x_spatial)   # [B, 768, H', W']
        feat_maps = self.spatial_norm(feat_maps)        # LayerNorm2d normalisation
        f_s = self.spatial_pool(feat_maps).flatten(1)   # [B, 768]
        f_s = self.spatial_fc(f_s)                      # [B, 512]

        if self.use_fft_branch:
            srm_out = self.srm(x)  # [B, 9, H, W]
            bayar_out = self.bayar(x)  # [B, 1, H, W]
            noise_combined = torch.cat([srm_out, bayar_out], dim=1)  # [B, 10, H, W]
            freq_maps = self.fft(noise_combined)  # [B, 20, H, W]
            f_f = self.freq_conv(freq_maps).flatten(1)  # [B, 128]
            f_f = self.freq_fc(f_f)  # [B, 512]

            concat_feat = torch.cat([f_s, f_f], dim=1)  # [B, 1024]
            gate = self.gate_fc(concat_feat)  # [B, 512]
            # Symmetric gated fusion: both streams are gated so neither has a free
            # gradient path to the classifier. This prevents early training from
            # saturating the gate to 0 and permanently starving the frequency branch.
            fused = torch.cat([f_s * (1.0 - gate), f_f * gate], dim=1)  # [B, 1024]
        else:
            fused = f_s  # [B, 512]

        return self.classifier(fused)  # [B, 1]

    def forward_sequence(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        Forward pass for 5D video sequence input tensor [B, T, 3, H, W].
        Processes frames in chunks of `chunk_size` to avoid OOM when B*T is large,
        then averages logits across the temporal dimension.
        Returns pooled sequence logits [B, 1].

        Args:
            x: Tensor of shape [B, T, 3, H, W].
            chunk_size: Number of frames per GPU forward chunk. Reduce if OOM occurs.
        """
        batch_size, seq_len, c, h, w = x.shape
        x_flat = x.view(batch_size * seq_len, c, h, w)  # [B*T, 3, H, W]

        # Process in chunks to avoid OOM from flattening large B*T batches
        logit_chunks: list[torch.Tensor] = []
        for start in range(0, batch_size * seq_len, chunk_size):
            chunk = x_flat[start : start + chunk_size]
            logit_chunks.append(self.forward(chunk))

        frame_logits = torch.cat(logit_chunks, dim=0)  # [B*T, 1]
        frame_logits = frame_logits.view(batch_size, seq_len)  # [B, T]
        return frame_logits.mean(dim=1, keepdim=True)  # [B, 1]



