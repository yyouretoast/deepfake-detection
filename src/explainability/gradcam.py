from typing import Optional, Any, Tuple, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchGradCAM:
    """
    Grad-CAM Heatmap Generator for PyTorch ConvNeXt / EfficientNet models.
    Supports Python context manager protocol (`with PyTorchGradCAM(model) as gradcam:`)
    and batched heatmap generation for ultra-fast multi-frame rendering.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.backward_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        if target_layer is None:
            target_layer = self._find_last_conv_layer()
            
        self.target_layer: nn.Module = target_layer

    def __enter__(self: "PyTorchGradCAM") -> "PyTorchGradCAM":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._remove_hooks()
        self.feature_maps = None
        self.gradients = None

    def _find_last_conv_layer(self) -> nn.Module:
        """
        Target layer auto-detection restricted strictly to spatial_backbone.
        Prevents hook misdirection to 2D FFT frequency extractor layers.
        """
        target_root = getattr(self.model, "spatial_backbone", self.model)
        if hasattr(self.model, "module"):
            target_root = getattr(self.model.module, "spatial_backbone", self.model.module)

        last_layer: Optional[nn.Module] = None
        for name, module in target_root.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_layer = module
        if last_layer is None:
            raise ValueError("No Conv2d layer found in spatial backbone for Grad-CAM")
        return last_layer

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _remove_hooks(self) -> None:
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def _forward_hook(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.feature_maps = output.detach()

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Generates normalized Grad-CAM heatmap array [H, W] in range [0, 1] for a single image."""
        heatmaps = self.generate_heatmaps_batch(input_tensor, target_classes=[target_class])
        return heatmaps[0]

    def generate_heatmaps_batch(
        self,
        input_tensor_batch: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Batched Grad-CAM heatmap generation over a batch of images [B, 3, H, W]
        in 1 single forward/backward pass.
        """
        if input_tensor_batch.ndim == 3:
            input_tensor_batch = input_tensor_batch.unsqueeze(0)

        batch_size = input_tensor_batch.shape[0]
        if target_classes is None:
            target_classes = [1] * batch_size

        self._register_hooks()
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                tensor_clone = input_tensor_batch.detach().clone()
                tensor_clone.requires_grad = True
                
                outputs = self.model(tensor_clone)
                scores = outputs if outputs.ndim == 1 else outputs[:, 0]
                
                loss_weights = torch.tensor(
                    [1.0 if tc == 1 else -1.0 for tc in target_classes],
                    device=input_tensor_batch.device,
                    dtype=scores.dtype
                )
                loss = torch.sum(scores * loss_weights)

                loss.backward()
                self.model.zero_grad()

                if self.gradients is None or self.feature_maps is None:
                    return [np.zeros((input_tensor_batch.shape[2], input_tensor_batch.shape[3]), dtype=np.float32) for _ in range(batch_size)]

                heatmaps: List[np.ndarray] = []
                for b in range(batch_size):
                    weights = torch.mean(self.gradients[b], dim=(1, 2), keepdim=True)
                    cam = torch.sum(weights * self.feature_maps[b], dim=0)
                    cam = F.relu(cam)
                    
                    cam_np = cam.cpu().numpy()
                    cam_np = cam_np - np.min(cam_np)
                    cam_np = cam_np / (np.max(cam_np) + 1e-8)
                    heatmaps.append(cam_np)

                return heatmaps
        finally:
            self._remove_hooks()
            self.feature_maps = None
            self.gradients = None

    def __del__(self) -> None:
        self._remove_hooks()

def overlay_cam(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlays normalized [0, 1] heatmap on RGB uint8 image using OpenCV Jet colormap."""
    h, w, _ = image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    return blended
