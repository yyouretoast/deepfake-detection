from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.fft
import timm

from src.config import load_config

class FFTFrequencyExtractor(nn.Module):
    """
    2D Real Fast Fourier Transform (FFT) Frequency Feature Extractor.
    Extracts 128-d frequency spectrum embeddings in FP32 precision.
    Uses 1x1 Conv2d grayscale projection for single-kernel GPU execution.
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
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force FP32 computation via autocast(enabled=False) to prevent cuFFT FP16 power-of-two size errors (224x224)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.to(torch.float32)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            raw_x = (x_fp32 * std + mean).clamp(0.0, 1.0)
            
            gray = self.rgb_to_gray(raw_x)
            fft_2d = torch.fft.rfft2(gray, norm="ortho")
            
            magnitude = torch.abs(fft_2d)
            eps = 1e-5
            log_spectrum = torch.log(magnitude + eps)
            norm_spectrum = log_spectrum / 10.0

        norm_spectrum = norm_spectrum.to(x.dtype)
        conv_features = self.conv_net[:-1](norm_spectrum)
        feat = self.conv_net[-1](conv_features)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

    def forward_grid(self, x: torch.Tensor, target_h: int = 8, target_w: int = 8) -> torch.Tensor:
        """Extracts spatial frequency feature grid dynamically adaptive-pooled to (target_h, target_w)."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.to(torch.float32)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
            raw_x = (x_fp32 * std + mean).clamp(0.0, 1.0)
            
            gray = self.rgb_to_gray(raw_x)
            fft_2d = torch.fft.rfft2(gray, norm="ortho")
            
            magnitude = torch.abs(fft_2d)
            eps = 1e-5
            log_spectrum = torch.log(magnitude + eps)
            norm_spectrum = log_spectrum / 10.0

        norm_spectrum = norm_spectrum.to(x.dtype)
        conv_features = self.conv_net[:-1](norm_spectrum)
        grid_pooled = nn.functional.adaptive_avg_pool2d(conv_features, (target_h, target_w))
        return grid_pooled

class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector with Multi-Head Cross-Attention.
    Fuses Spatial Backbone (ConvNeXt-Base, 1024-d) and Frequency Stream (2D FFT, 128-d)
    via 4-Head Cross-Attention into an 1152-d feature representation.
    """
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        use_fft_branch: bool = True,
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
        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features
        freq_embed_dim = model_cfg.get("freq_embed_dim", 128)

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=freq_embed_dim)
            self.spatial_proj = nn.Linear(spatial_in_features, 128)
            self.freq_proj = nn.Linear(freq_embed_dim, 128)
            self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.gate_fc = None
            fusion_dim = spatial_in_features + freq_embed_dim
        else:
            self.freq_extractor = None
            self.spatial_proj = None
            self.freq_proj = None
            self.cross_attn = None
            self.gate_fc = None
            fusion_dim = spatial_in_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts intermediate feature representations (spatial, frequency, fused)."""
        spatial_grid = self.spatial_backbone.forward_features(x)
        if spatial_grid.ndim == 4 and spatial_grid.shape[1] != self.spatial_backbone.num_features:
            spatial_grid = spatial_grid.permute(0, 3, 1, 2)
        spatial_raw = self.spatial_backbone.forward_head(spatial_grid, pre_logits=True)

        if self.use_fft_branch and self.freq_extractor is not None and self.cross_attn is not None:
            B, C_s, H_s, W_s = spatial_grid.shape
            spatial_tokens = spatial_grid.flatten(2).transpose(1, 2)
            
            freq_grid = self.freq_extractor.forward_grid(x, target_h=H_s, target_w=W_s)
            freq_tokens = freq_grid.flatten(2).transpose(1, 2)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features["fused"])
        return logits.view(-1)

def build_model(
    use_fft: bool = True,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    compile_model: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """Factory function to build, wrap in DataParallel, and optionally JIT compile model."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HybridDeepfakeDetector(pretrained=pretrained, use_fft_branch=use_fft, config=config)
    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model
