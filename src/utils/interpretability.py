"""Model interpretability utilities including ConvNeXtGradCAM and 4-panel face diagnostic generator."""

import logging
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hybrid_detector import HybridDeepfakeDetector

logger = logging.getLogger(__name__)


class ConvNeXtGradCAM:
    """Grad-CAM Heatmap Engine for ConvNeXt Spatial Backbone."""

    def __init__(self, model: HybridDeepfakeDetector) -> None:
        self.model = model
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer = self.model.spatial_backbone[-1]
        self.forward_handle = target_layer.register_forward_hook(self._save_feature_maps)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.feature_maps = output

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, img_size: int = 512) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            logits = self.model(input_tensor)
            scalar_logit = logits.squeeze()
            scalar_logit.backward()

        if self.feature_maps is None or self.gradients is None:
            return np.zeros((img_size, img_size), dtype=np.float32)

        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i]

        cam = F.relu(cam).detach().cpu().numpy()
        denom = float(cam.max() - cam.min())
        if denom > 1e-6:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        return cv2.resize(cam, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    def remove_hooks(self) -> None:
        try:
            self.forward_handle.remove()
            self.backward_handle.remove()
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.debug("Failed to remove hooks: %s", e)


def generate_face_diagnostics(
    model: HybridDeepfakeDetector,
    face_rgb: np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generates 4-panel interpretability representations (RGB, SRM, FFT, Grad-CAM)."""
    img_size = face_rgb.shape[0]
    img_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        srm_out = model.srm(img_tensor)
        bayar_out = model.bayar(img_tensor)
        noise_combined = torch.cat([srm_out, bayar_out], dim=1)
        freq_maps = model.fft(noise_combined)

    # Panel B: SRM High-Pass Residual Noise Map
    srm_map = srm_out[0].abs().mean(dim=0).cpu().numpy()
    srm_denom = max(float(srm_map.max() - srm_map.min()), 1e-6)
    srm_norm = (srm_map - srm_map.min()) / srm_denom
    srm_uint8 = (srm_norm * 255.0).astype(np.uint8)
    srm_colored = cv2.applyColorMap(srm_uint8, cv2.COLORMAP_VIRIDIS)
    srm_rgb = cv2.cvtColor(srm_colored, cv2.COLOR_BGR2RGB)

    # Panel C: Centered 2D Real FFT Log-Magnitude Spectrum
    mag_maps = freq_maps[0, :10].cpu().numpy()
    mean_mag = np.mean(mag_maps, axis=0)
    fft_centered = np.fft.fftshift(mean_mag)
    fft_denom = max(float(fft_centered.max() - fft_centered.min()), 1e-6)
    fft_norm = (fft_centered - fft_centered.min()) / fft_denom
    fft_uint8 = (fft_norm * 255.0).astype(np.uint8)
    fft_colored = cv2.applyColorMap(fft_uint8, cv2.COLORMAP_MAGMA)
    fft_rgb = cv2.cvtColor(fft_colored, cv2.COLOR_BGR2RGB)

    # Panel D: Grad-CAM Heatmap Overlay
    grad_cam = ConvNeXtGradCAM(model)
    cam_map = grad_cam.generate_heatmap(img_tensor.clone(), img_size=img_size)
    grad_cam.remove_hooks()

    cam_uint8 = (cam_map * 255.0).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(face_rgb, 0.6, cam_rgb, 0.4, 0)

    return {
        "original": face_rgb,
        "srm_residual": srm_rgb,
        "fft_spectrum": fft_rgb,
        "gradcam_overlay": cam_overlay,
    }
