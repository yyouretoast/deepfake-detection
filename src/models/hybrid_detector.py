from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from src.config import load_config


class SRMConv2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        srm1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        srm2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        srm3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
        
        filters = np.stack([srm1, srm2, srm3], axis=0)[:, np.newaxis, :, :]
        filters = np.tile(filters, (3, 1, 1, 1))
        self.register_buffer("weights", torch.from_numpy(filters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, w, stride=1, padding=2, groups=3)


class BayarConv2d(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 5, 5))

    def _get_constrained_kernel(self) -> torch.Tensor:
        w = self.kernel
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
        return F.conv2d(x, self._get_constrained_kernel(), stride=1, padding=2)


class RealFFT2DModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float()
            fft = torch.fft.rfft2(x_fp32, norm='ortho')
            mag = torch.log1p(torch.abs(fft))
            phase = torch.clamp(torch.angle(fft) / torch.pi, -1.0, 1.0)
            
            mag = torch.nan_to_num(mag, nan=0.0, posinf=1.0, neginf=-1.0)
            phase = torch.nan_to_num(phase, nan=0.0, posinf=1.0, neginf=-1.0)
            
            mag_resized = F.interpolate(mag, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            phase_resized = F.interpolate(phase, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out_fp32 = torch.cat([mag_resized, phase_resized], dim=1)
        return out_fp32.to(dtype=x.dtype)


class HybridDeepfakeDetector(nn.Module):
    """
    Canonical Dual-Stream Hybrid Deepfake Detector.
    Fuses Spatial Backbone (ConvNeXt-Small, 512-d) and Frequency Stream (SRM + Bayar 2D FFT, 512-d)
    via Sigmoid Gated Residual Fusion ($f_{\\text{fused}} = [f_s \\,\\|\\, (f_f \\odot g)] \\in \\mathbb{R}^{1024}$).
    """
    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        use_fft_branch: bool = True,
        dropout: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.use_fft_branch = use_fft_branch

        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_small(weights=weights)
        self.spatial_backbone = convnext.features
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU()
        )
        self.imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
                nn.AdaptiveAvgPool2d(1)
            )
            self.freq_fc = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU()
            )
            self.gate_fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Sigmoid()
            )
            classifier_in = 1024
        else:
            classifier_in = 512

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 4D image input tensor [B, 3, H, W].
        """
        # ImageNet normalization without torchvision data-dependent branch guards
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_spatial = (x - mean) / std
        f_s = self.spatial_pool(self.spatial_backbone(x_spatial)).flatten(1)
        f_s = self.spatial_fc(f_s)

        if self.use_fft_branch:
            srm_out = self.srm(x)
            bayar_out = self.bayar(x)
            noise_combined = torch.cat([srm_out, bayar_out], dim=1)
            freq_maps = self.fft(noise_combined)
            f_f = self.freq_conv(freq_maps).flatten(1)
            f_f = self.freq_fc(f_f)

            concat_feat = torch.cat([f_s, f_f], dim=1)
            gate = self.gate_fc(concat_feat)
            gated_freq = f_f * gate
            fused = torch.cat([f_s, gated_freq], dim=1)
        else:
            fused = f_s

        return self.classifier(fused)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 5D video sequence input tensor [B, T, 3, H, W].
        Performs frame-level forward passes followed by logit-space temporal average pooling.
        """
        batch_size, seq_len, c, h, w = x.shape
        x_flat = x.view(batch_size * seq_len, c, h, w)
        frame_logits = self.forward(x_flat)
        frame_logits = frame_logits.view(batch_size, seq_len)
        return frame_logits.mean(dim=1, keepdim=True)


