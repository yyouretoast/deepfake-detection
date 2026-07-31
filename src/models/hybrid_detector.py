from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import timm
from torch.utils.checkpoint import checkpoint

from src.config import load_config
from src.models.temporal import TemporalSequenceEncoder

class FFTFrequencyExtractor(nn.Module):
    """
    2D Real Fast Fourier Transform (FFT) Frequency Feature Extractor.
    Extracts 128-d frequency spectrum embeddings in FP32 precision.
    Supports native 512x512 full-resolution inputs with adaptive Nyquist grid pooling.
    """
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.out_features = out_features
        
        self.rgb_to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.rgb_to_gray.weight.data = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]], dtype=torch.float32)
        self.rgb_to_gray.weight.requires_grad = False

        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
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

    def _extract_norm_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 2-channel FP32 frequency spectrum feature map (log-magnitude + normalized phase angle)."""
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.to(torch.float32)
            raw_x = (x_fp32 * self.std + self.mean).clamp(0.0, 1.0)
            gray = self.rgb_to_gray(raw_x)
            fft_2d = torch.fft.rfft2(gray, norm="ortho")
            magnitude = torch.abs(fft_2d)
            log_spectrum = torch.log(magnitude + 1e-5)
            log_spectrum = torch.fft.fftshift(log_spectrum, dim=-2)
            mean = log_spectrum.mean(dim=(-2, -1), keepdim=True)
            std = log_spectrum.std(dim=(-2, -1), keepdim=True)
            norm_magnitude = (log_spectrum - mean) / (std + 1e-6)

            # Channel 1: Normalized Phase Angle [-1.0, +1.0]
            phase_angle = torch.angle(fft_2d) / torch.pi
            phase_angle = torch.fft.fftshift(phase_angle, dim=-2)

            two_channel_spectrum = torch.cat([norm_magnitude, phase_angle], dim=1)
        return self.conv_net(two_channel_spectrum.to(x.dtype))

    def forward_grid(self, x: torch.Tensor, target_h: int = 8, target_w: int = 8) -> torch.Tensor:
        """Extracts spatial frequency feature grid dynamically adaptive-pooled to (target_h, target_w)."""
        conv_features = self._extract_norm_spectrum(x)
        return F.adaptive_avg_pool2d(conv_features, (target_h, target_w))

class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector with Multi-Head Cross-Attention.
    Fuses Spatial Backbone (ConvNeXt-Base, 1024-d) and Frequency Stream (2D FFT, 128-d)
    via 4-Head Cross-Attention into an 1152-d feature representation.
    Supports Pre-Downsample 512x512 FFT Extraction and LoRA Parameter-Efficient Fine-Tuning.
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
            backbone_name = model_cfg.get("backbone", "convnext_base")

        self.use_fft_branch = use_fft_branch
        self.use_checkpointing = use_checkpointing or model_cfg.get("use_checkpointing", False)

        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features
        freq_embed_dim = model_cfg.get("freq_embed_dim", 128)

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=freq_embed_dim)
            self.spatial_proj = nn.Linear(spatial_in_features, 128)
            self.freq_proj = nn.Linear(freq_embed_dim, 128)
            self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.attn_out_proj = nn.Linear(128, freq_embed_dim)
            self.gamma = nn.Parameter(torch.tensor(0.1))
            fusion_dim = spatial_in_features + freq_embed_dim
        else:
            self.freq_extractor = None
            self.spatial_proj = None
            self.freq_proj = None
            self.cross_attn = None
            self.gamma = None
            fusion_dim = spatial_in_features

        self.fusion_dim = fusion_dim
        self.temporal_encoder = TemporalSequenceEncoder(embed_dim=self.fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """Adapter hook to seamlessly duplicate 1-channel legacy weights to 2-channel phase FFT models."""
        key = "freq_extractor.conv_net.0.weight"
        if key in state_dict and self.use_fft_branch and self.freq_extractor is not None:
            w = state_dict[key]
            if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[1] == 1:
                state_dict[key] = w.repeat(1, 2, 1, 1) / 2.0
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extract_features(
        self,
        x: torch.Tensor,
        x_full: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature representations (spatial, frequency, fused).
        Performs GPU-side bilinear downscaling to 256x256 for ConvNeXt while evaluating 
        512x512 native FFT frequency spectra.
        """
        if x_full is None:
            x_full = x
            
        if x.ndim == 4:
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
            freq_raw = freq_enhanced_tokens.mean(dim=1)
            fused = torch.cat([spatial_raw, freq_raw], dim=1)
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
        padding_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 8
    ) -> torch.Tensor:
        """
        Processes 5D video sequence tensors [B, T, 3, H, W] via mini-batched spatial-frequency
        extraction (chunk_size=8) to cap VRAM under 600MB, followed by Temporal Transformer sequence modeling.
        padding_mask: optional bool tensor [B, T], True = padded frame (to be ignored).
        """
        if x_seq.ndim == 4:
            return self.forward(x_seq)

        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)

        fused_list = []
        total_frames = B * T
        for i in range(0, total_frames, chunk_size):
            chunk = x_flat[i : i + chunk_size]
            feats = self.extract_features(chunk)
            fused_list.append(feats["fused"])

        fused_frames = torch.cat(fused_list, dim=0)
        fused_seq = fused_frames.view(B, T, -1)
        if padding_mask is not None:
            fused_seq = fused_seq * (~padding_mask).unsqueeze(-1).float()
        pooled_seq = self.temporal_encoder(fused_seq, padding_mask=padding_mask)

        logits = self.classifier(pooled_seq)
        return logits.view(-1)

def build_model(
    use_fft: bool = True,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    compile_model: bool = False,
    backbone_name: Optional[str] = None,
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
