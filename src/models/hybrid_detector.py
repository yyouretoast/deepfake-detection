from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import timm

from src.config import load_config
from src.models.temporal import TemporalSequenceEncoder
from src.models.lora import apply_lora_to_model, merge_all_lora_weights

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
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
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
        """Extracts normalized FP32 log-magnitude frequency spectrum feature map."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.to(torch.float32)
            raw_x = (x_fp32 * self.std + self.mean).clamp(0.0, 1.0)
            gray = self.rgb_to_gray(raw_x)
            fft_2d = torch.fft.rfft2(gray, norm="ortho")
            magnitude = torch.abs(fft_2d)
            log_spectrum = torch.log(magnitude + 1e-5)
            norm_spectrum = log_spectrum / 10.0
        return self.conv_net(norm_spectrum.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self._extract_norm_spectrum(x)
        return conv_features.mean(dim=(-2, -1))

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
        use_lora: bool = False,
        lora_rank: int = 8,
        dropout: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        if config is None:
            config = load_config()

        model_cfg = config.get("model", {})
        if backbone_name is None:
            backbone_name = model_cfg.get("backbone", "convnext_base")

        self.use_fft_branch = use_fft_branch
        self.use_lora = use_lora or model_cfg.get("use_lora", False)
        self.lora_rank = lora_rank if lora_rank != 8 else model_cfg.get("lora_rank", 8)

        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features
        freq_embed_dim = model_cfg.get("freq_embed_dim", 128)

        if self.use_lora:
            apply_lora_to_model(self.spatial_backbone, rank=self.lora_rank)

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=freq_embed_dim)
            self.spatial_proj = nn.Linear(spatial_in_features, 128)
            self.freq_proj = nn.Linear(freq_embed_dim, 128)
            self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            fusion_dim = spatial_in_features + freq_embed_dim
        else:
            self.freq_extractor = None
            self.spatial_proj = None
            self.freq_proj = None
            self.cross_attn = None
            fusion_dim = spatial_in_features

        self.temporal_encoder = TemporalSequenceEncoder(embed_dim=fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def merge_lora_weights(self) -> None:
        """Folds all LoRA weights into base parameters for 0ms inference latency penalty."""
        if self.use_lora:
            merge_all_lora_weights(self.spatial_backbone)

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
            if x.ndim == 4 and x.shape[-1] > 256:
                x_full = x
                x = F.interpolate(x_full, size=(256, 256), mode='bilinear', align_corners=False)
            else:
                x_full = x
        else:
            if x.shape[-1] > 256 and (x_full is None or x_full.shape[-1] == x.shape[-1]):
                x_spatial = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
                x_full = x
                x = x_spatial

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

            freq_enhanced_tokens = freq_tokens + 0.1 * attn_out
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
        features = self.extract_features(x, x_full=x_full)
        logits = self.classifier(features["fused"])
        return logits.view(-1)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Processes 5D video sequence tensors [B, T, 3, H, W] via 1-pass spatial-frequency
        extraction and Temporal Transformer sequence modeling.
        """
        if x_seq.ndim == 4:
            return self.forward(x_seq)

        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)

        feats = self.extract_features(x_flat)
        fused_frames = feats["fused"]

        fused_seq = fused_frames.view(B, T, -1)
        pooled_seq = self.temporal_encoder(fused_seq)

        logits = self.classifier(pooled_seq)
        return logits.view(-1)

def build_model(
    use_fft: bool = True,
    use_lora: bool = False,
    lora_rank: int = 8,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    compile_model: bool = False,
    backbone_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """Factory function to build and optionally compile model."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HybridDeepfakeDetector(
        backbone_name=backbone_name,
        pretrained=pretrained,
        use_fft_branch=use_fft,
        use_lora=use_lora,
        lora_rank=lora_rank,
        config=config
    )
    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model
