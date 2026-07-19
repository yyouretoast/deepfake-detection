from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.fft
import timm

class FFTFrequencyExtractor(nn.Module):
    """
    2D Real Fast Fourier Transform (FFT) Frequency Feature Extractor.
    Extracts 128-d frequency spectrum embeddings in FP32 precision to prevent AMP FP16 underflow.
    
    Tensor Flow:
      [B, 3, H, W] -> Grayscale [B, 1, H, W] -> 2D FFT & Shift [B, 1, H, W] 
      -> Log Spectrum [B, 1, H, W] -> Conv2d Stack -> [B, 128]
    """
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.out_features = out_features
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
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        gray_fp32 = gray.to(torch.float32)
        
        fft_2d = torch.fft.fft2(gray_fp32)
        fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))
        
        magnitude = torch.abs(fft_shift)
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
    into an 896-d feature representation for final binary classification.
    
    Tensor Geometry:
      Input: [B, 3, 224, 224]
      Spatial Features: [B, 768]
      Frequency Features: [B, 128]
      Fused Features: [B, 896]
      Output Logits: [B]
    """
    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        use_fft_branch: bool = True,
        dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.use_fft_branch = use_fft_branch
        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features: int = self.spatial_backbone.num_features

        if self.use_fft_branch:
            self.freq_extractor = FFTFrequencyExtractor(out_features=128)
            fusion_dim = spatial_in_features + 128
        else:
            self.freq_extractor = None
            fusion_dim = spatial_in_features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature representations without passing through classifier head.
        Returns dictionary containing 'spatial', 'frequency', and 'fused' feature tensors.
        """
        spatial_feat = self.spatial_backbone(x)
        if self.use_fft_branch and self.freq_extractor is not None:
            freq_feat = self.freq_extractor(x)
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
    pretrained: bool = True
) -> nn.Module:
    """Factory function to build and wrap model in DataParallel if multi-GPU is present."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=pretrained, use_fft_branch=use_fft)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model.to(device)
