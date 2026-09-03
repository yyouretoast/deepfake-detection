"""Dual-Stream Hybrid Deepfake Detector Engine.

Spatial Stream: ConvNeXt-Small backbone extracted feature vector f_s in R^512.
Frequency Stream: SRM spatial residual filters (9-ch) + Bayar constrained conv (1-ch) -> 10-ch noise residuals.
2D FFT Spectral Decomposition: Magnitude & Phase maps (20-ch) -> freq projection f_f in R^512.
Symmetric Gated Residual Fusion:
    g = Sigmoid(Linear([f_s || f_f])) in R^512
    f_fused = [(1 - g) * f_s || g * f_f] in R^1024
"""

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torchvision import models

from src.models.fusion import ClassificationHead, GatedResidualFusion, LayerNorm2d
from src.models.spectral import RealFFT2DModule
from src.models.spectral_tower import ResSESpectralTower
from src.models.steganography import BayarConv2d, SRMConv2d

logger = logging.getLogger(__name__)

__all__ = [
    "SRMConv2d",
    "BayarConv2d",
    "RealFFT2DModule",
    "LayerNorm2d",
    "GatedResidualFusion",
    "ClassificationHead",
    "HybridDeepfakeDetector",
]


class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector.
    Fuses Spatial Stream (ConvNeXt-Small, 512-d) and Frequency Stream (SRM + Bayar 2D FFT, 512-d)
    via Symmetric Gated Residual Fusion: f_fused = [(1 - g) * f_s || g * f_f] in R^1024.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        use_fft_branch: bool = True,
        dropout: float = 0.3,
        config: Optional[dict[str, Any]] = None,
        frequency_backbone: str = "legacy",
    ) -> None:
        super().__init__()
        if config is not None:
            model_cfg = config.get("model", {})
            backbone_name = model_cfg.get("backbone", backbone_name)
            pretrained = model_cfg.get("pretrained", pretrained)
            use_fft_branch = model_cfg.get("use_fft_branch", use_fft_branch)
            dropout = model_cfg.get("dropout", dropout)
            frequency_backbone = model_cfg.get("frequency_backbone", frequency_backbone)

        self.use_fft_branch = use_fft_branch
        self.backbone_name = backbone_name
        self.frequency_backbone = frequency_backbone

        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_small(weights=weights)
        self.spatial_backbone = convnext.features
        self.spatial_norm = convnext.classifier[0]  # nn.LayerNorm2d(768)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_fc = nn.Sequential(nn.Linear(768, 512), nn.ReLU())

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.use_fft_branch:
            self.srm = SRMConv2d()
            self.bayar = BayarConv2d(in_channels=3, out_channels=1)
            self.fft = RealFFT2DModule()
            if self.frequency_backbone == "resse":
                self.freq_tower = ResSESpectralTower(in_channels=20, embed_dim=512)
            else:
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

    def _switch_to_legacy_frequency_branch(self) -> None:
        """Dynamically switches frequency stream to legacy 90k CNN to load legacy checkpoints."""
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
        if hasattr(self, "freq_tower"):
            delattr(self, "freq_tower")
        self.freq_conv = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        ).to(device=device, dtype=dtype)
        self.freq_fc = nn.Sequential(nn.Linear(128, 512), nn.ReLU()).to(device=device, dtype=dtype)
        self.frequency_backbone = "legacy"

    def _switch_to_resse_frequency_branch(self) -> None:
        """Dynamically switches frequency stream to ResSE-Spectral Tower."""
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
        if hasattr(self, "freq_conv"):
            delattr(self, "freq_conv")
        if hasattr(self, "freq_fc"):
            delattr(self, "freq_fc")
        self.freq_tower = ResSESpectralTower(in_channels=20, embed_dim=512).to(device=device, dtype=dtype)
        self.frequency_backbone = "resse"

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False):
        """Auto-detects whether incoming state_dict is from legacy or ResSE-Spectral architecture."""
        has_legacy = any(k.startswith("freq_conv.") for k in state_dict)
        has_resse = any(k.startswith("freq_tower.") for k in state_dict)
        if has_legacy and hasattr(self, "freq_tower") and not hasattr(self, "freq_conv"):
            self._switch_to_legacy_frequency_branch()
        elif has_resse and hasattr(self, "freq_conv") and not hasattr(self, "freq_tower"):
            self._switch_to_resse_frequency_branch()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim fused feature embedding for downstream spatiotemporal sequence modeling."""
        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x_spatial = (x - mean) / std

        feat_maps = self.spatial_backbone(x_spatial)
        feat_maps = self.spatial_norm(feat_maps)
        f_s = self.spatial_pool(feat_maps).flatten(1)
        f_s = self.spatial_fc(f_s)

        if self.use_fft_branch:
            srm_out = self.srm(x)
            bayar_out = self.bayar(x)
            noise_combined = torch.cat([srm_out, bayar_out], dim=1)
            freq_maps = self.fft(noise_combined)
            if hasattr(self, "freq_tower"):
                f_f, _ = self.freq_tower(freq_maps)
            else:
                f_f = self.freq_conv(freq_maps).flatten(1)
                f_f = self.freq_fc(f_f)

            concat_feat = torch.cat([f_s, f_f], dim=1)
            gate = self.gate_fc(concat_feat)
            return (1.0 - gate) * f_s + gate * f_f
        return f_s

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for 4D image input tensor [B, 3, H, W].
        Returns unscaled classification logits [B, 1], or (logits, aux_logit) if return_aux=True.
        """
        if self.training and not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if (x < -0.1).any() or (x > 1.1).any():
                logger.warning(
                    "Input tensor x has values outside [0, 1] range: min=%.3f, max=%.3f. "
                    "SRM and Bayar filters expect unnormalized [0, 1] inputs.",
                    float(x.min()),
                    float(x.max()),
                )

        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        x_spatial = (x - mean) / std

        feat_maps = self.spatial_backbone(x_spatial)
        feat_maps = self.spatial_norm(feat_maps)
        f_s = self.spatial_pool(feat_maps).flatten(1)
        f_s = self.spatial_fc(f_s)

        aux_logit = None
        if self.use_fft_branch:
            srm_out = self.srm(x)
            bayar_out = self.bayar(x)
            noise_combined = torch.cat([srm_out, bayar_out], dim=1)
            freq_maps = self.fft(noise_combined)
            if hasattr(self, "freq_tower"):
                f_f, aux_logit = self.freq_tower(freq_maps)
            else:
                f_f = self.freq_conv(freq_maps).flatten(1)
                f_f = self.freq_fc(f_f)
                aux_logit = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

            concat_feat = torch.cat([f_s, f_f], dim=1)
            gate = self.gate_fc(concat_feat)
            fused = torch.cat([f_s * (1.0 - gate), f_f * gate], dim=1)
        else:
            fused = f_s
            aux_logit = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

        logits = self.classifier(fused)
        if return_aux:
            return logits, aux_logit
        return logits

    def forward_sequence(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        Forward pass for 5D video frame sequence tensor [B, T, 3, H, W].
        Processes frames through chunked sub-batches to bound peak VRAM allocation.
        """
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        logits_list = []
        for i in range(0, b * t, chunk_size):
            chunk = x_reshaped[i : i + chunk_size]
            logits_list.append(self.forward(chunk))
        logits = torch.cat(logits_list, dim=0)
        frame_logits = logits.view(b, t)
        return frame_logits.mean(dim=1, keepdim=True)
