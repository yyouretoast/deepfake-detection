from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import timm
from torch.utils.checkpoint import checkpoint

from src.config import load_config

class FFTFrequencyExtractor(nn.Module):
    """
    Dual-Domain FFT & SRM Noise Residual Frequency Extractor.
    Combines 3 fixed linear Steganographic Rich Model (SRM) high-pass noise kernels with 
    1 learnable Bayar-Stamm constrained high-pass convolution layer.
    Extracts 8-channel spectral maps (4 log-magnitude + 4 phase angle) into 512-d embeddings.
    Supports native 512x512 full-resolution inputs with adaptive Nyquist grid pooling.
    """
    def __init__(self, out_features: int = 512) -> None:
        super().__init__()
        self.out_features = out_features
        
        self.rgb_to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.rgb_to_gray.weight.data = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]], dtype=torch.float32)
        self.rgb_to_gray.weight.requires_grad = False

        # 1. Fixed SRM 3x3 Linear Residual High-Pass Filter Kernels (3 filters)
        self.srm_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        srm_weights = torch.tensor([
            [[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]],        # 1st order Laplacian
            [[[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]],    # 2nd order Edge
            [[[-1., 2., -1.], [2., -4., 2.], [0., 0., 0.]]]       # 3rd order Directional
        ], dtype=torch.float32) / 4.0
        with torch.no_grad():
            self.srm_conv.weight.copy_(srm_weights)
        self.srm_conv.weight.requires_grad = False

        # 2. Learnable Bayar-Stamm Constrained High-Pass Filter (1 filter)
        self.bayar_conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayar_conv.weight)

        # 8 Input channels: (3 SRM + 1 Bayar) * (1 Magnitude + 1 Phase) = 8 Channels
        self.conv_net = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _get_constrained_bayar_weights(self) -> torch.Tensor:
        """Enforces Bayar-Stamm sum-to-zero and center=-1 constraint without in-place tensor mutations."""
        w = self.bayar_conv.weight
        mask = torch.ones_like(w)
        mask[:, :, 2, 2] = 0.0
        w_masked = w * mask
        sum_w = w_masked.sum(dim=(2, 3), keepdim=True)
        w_norm = w_masked / (sum_w + 1e-7)
        
        center_mask = torch.zeros_like(w)
        center_mask[:, :, 2, 2] = -1.0
        return w_norm + center_mask

    def _extract_norm_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 8-channel FP32 frequency spectrum feature map with NaN/Inf numerical safeguards."""
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.to(torch.float32)
            raw_x = (x_fp32 * self.std + self.mean).clamp(0.0, 1.0)
            gray = self.rgb_to_gray(raw_x)

            # Apply SRM and Constrained Bayar Filters (Linear floats — no ReLU)
            srm_out = self.srm_conv(gray)  # [B, 3, H, W]
            bayar_w = self._get_constrained_bayar_weights()
            bayar_out = F.conv2d(gray, bayar_w, padding=2)  # [B, 1, H, W]

            multi_channel_spatial = torch.cat([srm_out, bayar_out], dim=1)  # [B, 4, H, W]

            fft_2d = torch.fft.rfft2(multi_channel_spatial, norm="ortho")
            magnitude = torch.abs(fft_2d)
            log_spectrum = torch.log(magnitude + 1e-5)
            log_spectrum = torch.fft.fftshift(log_spectrum, dim=-2)
            
            mean = log_spectrum.mean(dim=(-2, -1), keepdim=True)
            std = log_spectrum.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
            norm_magnitude = (log_spectrum - mean) / std
            norm_magnitude = torch.nan_to_num(norm_magnitude, nan=0.0, posinf=1.0, neginf=-1.0)

            phase_angle = torch.angle(fft_2d) / torch.pi
            phase_angle = torch.fft.fftshift(phase_angle, dim=-2)
            phase_angle = torch.nan_to_num(phase_angle, nan=0.0, posinf=1.0, neginf=-1.0)

            eight_channel_spectrum = torch.cat([norm_magnitude, phase_angle], dim=1)  # [B, 8, H, W]

        return self.conv_net(eight_channel_spectrum.to(x.dtype))

    def forward_grid(self, x: torch.Tensor, target_h: int = 8, target_w: int = 8) -> torch.Tensor:
        """Extracts spatial frequency feature grid dynamically adaptive-pooled to (target_h, target_w)."""
        conv_features = self._extract_norm_spectrum(x)
        return F.adaptive_avg_pool2d(conv_features, (target_h, target_w))

class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector with Residual Gated Multi-Head Cross-Attention.
    Fuses Spatial Backbone (ConvNeXt-Base, 1024-d) and Frequency Stream (SRM + Bayar 2D FFT, 512-d)
    into a balanced 1024-d equalized feature representation ($512 + 512 = 1024$).
    """
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        use_fft_branch: bool = True,
        dropout: float = 0.3,
        use_checkpointing: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        if config is None:
            config = load_config()

        model_cfg = config.get("model", {})
        if backbone_name is None:
            backbone_name = model_cfg.get("backbone", "convnext_small")

        self.use_fft_branch = use_fft_branch
        self.use_checkpointing = use_checkpointing or model_cfg.get("use_checkpointing", False)
        self.sequence_trained = False

        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features
        freq_embed_dim = model_cfg.get("freq_embed_dim", 512)

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=freq_embed_dim)
            self.spatial_proj = nn.Linear(spatial_in_features, 512)
            self.freq_proj = nn.Linear(freq_embed_dim, 512)
            self.gate_net = nn.Sequential(
                nn.Linear(spatial_in_features + freq_embed_dim, 512),
                nn.Sigmoid()
            )
            self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self.attn_out_proj = nn.Linear(512, 512)
            self.gamma = nn.Parameter(torch.tensor(0.1))
            fusion_dim = 1024
        else:
            self.freq_extractor = None
            self.spatial_proj = None
            self.freq_proj = None
            self.gate_net = None
            self.cross_attn = None
            self.gamma = None
            fusion_dim = spatial_in_features

        self.fusion_dim = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """Adapter hook to seamlessly duplicate legacy checkpoint weights if shape dimensions match."""
        key = "freq_extractor.conv_net.0.weight"
        if key in state_dict and self.use_fft_branch and self.freq_extractor is not None:
            w = state_dict[key]
            if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[1] != self.freq_extractor.conv_net[0].weight.shape[1]:
                target_in_ch = self.freq_extractor.conv_net[0].weight.shape[1]
                repeat_factor = max(1, target_in_ch // w.shape[1])
                new_w = w.repeat(1, repeat_factor, 1, 1)[:, :target_in_ch, :, :] / float(repeat_factor)
                state_dict[key] = new_w
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extract_features(
        self,
        x: torch.Tensor,
        x_full: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature representations (spatial, frequency, fused).
        Performs GPU-side bilinear downscaling to 256x256 for ConvNeXt while evaluating 
        native full-resolution FFT frequency spectra.
        """
        if x_full is None:
            x_full = x
            
        if x.ndim == 4 and (x.shape[2] != 256 or x.shape[3] != 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            spatial_grid = checkpoint(self.spatial_backbone.forward_features, x, use_reentrant=False)
        else:
            spatial_grid = self.spatial_backbone.forward_features(x)

        if spatial_grid.ndim == 4 and spatial_grid.shape[1] != self.spatial_backbone.num_features:
            spatial_grid = spatial_grid.permute(0, 3, 1, 2)
        spatial_raw = self.spatial_backbone.forward_head(spatial_grid, pre_logits=True)

        if self.use_fft_branch and self.freq_extractor is not None and self.cross_attn is not None:
            B, C_s, H_s, W_s = spatial_grid.shape
            spatial_tokens = spatial_grid.view(B, C_s, H_s * W_s).transpose(1, 2)
            
            freq_grid = self.freq_extractor.forward_grid(x_full, target_h=H_s, target_w=W_s)
            freq_tokens = freq_grid.view(B, freq_grid.shape[1], H_s * W_s).transpose(1, 2)

            s_q = self.spatial_proj(spatial_tokens)
            f_kv = self.freq_proj(freq_tokens)
            attn_out, _ = self.cross_attn(query=s_q, key=f_kv, value=f_kv)

            freq_enhanced_tokens = freq_tokens + self.gamma * self.attn_out_proj(attn_out)
            freq_raw_512 = freq_enhanced_tokens.mean(dim=1)
            spatial_raw_512 = self.spatial_proj(spatial_raw)

            # Residual Gated Fusion: f_fused = [f_spatial_512 || f_freq_512 + g * f_spatial_512] (1024-d)
            g = self.gate_net(torch.cat([spatial_raw, freq_raw_512], dim=1))
            fused_freq = freq_raw_512 + g * spatial_raw_512
            fused = torch.cat([spatial_raw_512, fused_freq], dim=1)
            freq_raw = freq_raw_512
        else:
            freq_raw = torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)
            fused = spatial_raw

        return {
            "spatial": spatial_raw,
            "frequency": freq_raw,
            "fused": fused
        }

    def forward(self, x: torch.Tensor, x_full: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 5:
            return self.forward_sequence(x)
        features = self.extract_features(x, x_full=x_full)
        logits = self.classifier(features["fused"])
        return logits.view(-1)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes 5D video sequence tensors [B, T, 3, H, W].
        Performs calibrated logit-space temporal pooling over sequence frames.
        """
        if x_seq.ndim == 4:
            return self.forward(x_seq)

        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        logits_flat = self.forward(x_flat)
        logits_seq = logits_flat.view(B, T)
        if padding_mask is not None:
            valid = (~padding_mask).float()
            return (logits_seq * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return logits_seq.mean(dim=1)

def build_model(
    use_fft: bool = True,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    compile_model: bool = False,
    backbone_name: str = "convnext_small",
    use_checkpointing: bool = False,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    """Factory function to build and optionally compile model."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HybridDeepfakeDetector(
        backbone_name=backbone_name,
        pretrained=pretrained,
        use_fft_branch=use_fft,
        use_checkpointing=use_checkpointing,
        config=config
    )
    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model

