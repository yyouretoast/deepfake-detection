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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 and un-normalize ImageNet scaling back to raw [0, 1] RGB
        x_fp32 = x.to(torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
        raw_x = (x_fp32 * std + mean).clamp(0.0, 1.0)
        
        gray = self.rgb_to_gray(raw_x)
        fft_2d = torch.fft.rfft2(gray)
        
        magnitude = torch.abs(fft_2d)
        eps = 1e-5
        log_spectrum = torch.log(magnitude + eps)

        flat_spectrum = log_spectrum.flatten(1)
        min_val = flat_spectrum.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_val = flat_spectrum.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        
        norm_spectrum = (log_spectrum - min_val) / (max_val - min_val + eps)
        norm_spectrum = norm_spectrum.to(x.dtype)

        feat = self.conv_net(norm_spectrum)
        return self.fc(feat)

class HybridDeepfakeDetector(nn.Module):
    """
    Dual-Stream Hybrid Deepfake Detector.
    Fuses Spatial Backbone (ConvNeXt-Small, 768-d) and Frequency Stream (2D FFT, 128-d)
    via Adaptive Gated Fusion into an 896-d feature representation for binary classification.
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
            backbone_name = model_cfg.get("backbone", "convnext_small")

        self.use_fft_branch = use_fft_branch
        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features

        freq_embed_dim = model_cfg.get("freq_embed_dim", 128)

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=freq_embed_dim)
            self.gate_fc = nn.Linear(spatial_in_features + freq_embed_dim, freq_embed_dim)
            nn.init.constant_(self.gate_fc.bias, -2.0)  # Sigmoid(-2.0) ≈ 0.119 (Safe noise-free warmup)
            fusion_dim = spatial_in_features + freq_embed_dim
        else:
            self.freq_extractor = None
            self.gate_fc = None
            fusion_dim = spatial_in_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts intermediate feature representations (spatial, frequency, fused)."""
        spatial_feat = self.spatial_backbone(x)
        if self.use_fft_branch and self.freq_extractor is not None and self.gate_fc is not None:
            freq_raw = self.freq_extractor(x)
            gate = torch.sigmoid(self.gate_fc(torch.cat([spatial_feat, freq_raw], dim=1)))
            freq_feat = freq_raw * gate
            fused = torch.cat([spatial_feat, freq_feat], dim=1)
        else:
            freq_feat = torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)
            fused = spatial_feat

        return {
            "spatial": spatial_feat,
            "frequency": freq_feat,
            "fused": fused
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features["fused"])
        return logits.squeeze(-1)

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
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model
