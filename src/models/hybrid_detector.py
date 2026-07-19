import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import timm

class FFTFrequencyExtractor(nn.Module):
    """
    2D Fast Fourier Transform (FFT) Frequency Feature Stream.
    Computes log-magnitude frequency spectrum in FP32 precision to prevent AMP FP16 underflow/NaNs.
    """
    def __init__(self, out_features=128):
        super().__init__()
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

    def forward(self, x):
        # Convert RGB to Grayscale
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        # Enforce FP32 precision for FFT math to avoid FP16 underflow & NaNs under AMP
        gray_fp32 = gray.to(torch.float32)
        
        # Compute 2D Complex FFT & 2D Centered Shift
        fft_2d = torch.fft.fft2(gray_fp32)
        fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))
        
        magnitude = torch.abs(fft_shift)
        eps = 1e-5
        log_spectrum = torch.log(magnitude + eps)

        # Per-sample Min-Max Normalization to [0, 1]
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
    """
    def __init__(self, backbone_name="convnext_small", pretrained=True, use_fft_branch=True, dropout=0.3):
        super().__init__()
        self.use_fft_branch = use_fft_branch
        self.spatial_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        spatial_in_features = self.spatial_backbone.num_features

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

    def forward(self, x):
        spatial_feat = self.spatial_backbone(x)

        if self.use_fft_branch and self.freq_extractor is not None:
            freq_feat = self.freq_extractor(x)
            fused = torch.cat([spatial_feat, freq_feat], dim=1)
        else:
            fused = spatial_feat

        logits = self.classifier(fused)
        return logits.squeeze(-1)
