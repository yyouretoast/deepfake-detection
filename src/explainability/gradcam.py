from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchGradCAM:
    """
    Gradient-Weighted Class Activation Mapping (Grad-CAM) for Dual-Stream Deepfake Detector.
    Visualizes spatial feature maps responsible for manipulation classification decisions.
    Fully vectorized for GPU batch execution. Scales heatmaps to input tensor resolution (512x512).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model.eval()
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        if target_layer is None:
            spatial_backbone = getattr(self.model, "spatial_backbone", None)
            if spatial_backbone is not None and hasattr(spatial_backbone, "stages"):
                target_layer = spatial_backbone.stages[-1]
            elif spatial_backbone is not None and hasattr(spatial_backbone, "conv_head"):
                target_layer = spatial_backbone.conv_head
            else:
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module

        if target_layer is None:
            raise ValueError("Could not automatically locate target Conv2d layer for Grad-CAM.")

        self.target_layer = target_layer
        self.fh = self.target_layer.register_forward_hook(self._save_feature_maps)
        self.bh = self.target_layer.register_full_backward_hook(self._save_gradients)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "fh") and self.fh is not None:
            self.fh.remove()
        if hasattr(self, "bh") and self.bh is not None:
            self.bh.remove()

    def _save_feature_maps(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if output.ndim == 4 and output.shape[1] != output.shape[-1] and output.shape[-1] == output.shape[-2]:
            self.feature_maps = output
        elif output.ndim == 4 and output.shape[-1] != output.shape[1]:
            self.feature_maps = output.permute(0, 3, 1, 2)
        else:
            self.feature_maps = output

    def _save_gradients(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        grad = grad_output[0]
        if grad.ndim == 4 and grad.shape[1] != grad.shape[-1] and grad.shape[-1] == grad.shape[-2]:
            self.gradients = grad
        elif grad.ndim == 4 and grad.shape[-1] != grad.shape[1]:
            self.gradients = grad.permute(0, 3, 1, 2)
        else:
            self.gradients = grad

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Generates single 2D Grad-CAM heatmap array [H, W] normalized to [0.0, 1.0]."""
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        heatmaps = self.generate_heatmaps_batch(input_tensor, target_classes=[target_class])
        return heatmaps[0]

    def generate_heatmaps_batch(
        self,
        input_tensor_batch: torch.Tensor,
        target_classes: Optional[Union[List[int], int]] = None,
        target_class: int = 1
    ) -> List[np.ndarray]:
        """
        Generates Grad-CAM heatmaps for a batch of images [B, 3, H, W] in a single GPU pass.
        Returns a list of 2D numpy arrays [H, W] normalized to [0.0, 1.0].
        """
        self.gradients = None
        self.feature_maps = None

        with torch.enable_grad():
            self.model.eval()
            self.model.zero_grad()
            input_tensors = input_tensor_batch.clone().detach().requires_grad_(True)
            outputs = self.model(input_tensors)
            loss = torch.sum(outputs)
            loss.backward()

        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture feature maps or gradients.")

        target_h, target_w = input_tensor_batch.shape[2], input_tensor_batch.shape[3]

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cams = F.relu(torch.sum(weights * self.feature_maps, dim=1, keepdim=True))  # [B, 1, H, W]
        cams_upsampled = F.interpolate(
            cams,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )

        cams_np = cams_upsampled.squeeze(1).detach().cpu().numpy()
        
        heatmaps = []
        for b in range(cams_np.shape[0]):
            cam = cams_np[b]
            c_min, c_max = cam.min(), cam.max()
            norm_cam = (cam - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else np.zeros_like(cam)
            heatmaps.append(norm_cam)

        return heatmaps

    @staticmethod
    def overlay_heatmap(
        rgb_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Overlays Grad-CAM heatmap onto RGB image."""
        if heatmap.shape[:2] != rgb_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return overlay

overlay_cam = PyTorchGradCAM.overlay_heatmap
