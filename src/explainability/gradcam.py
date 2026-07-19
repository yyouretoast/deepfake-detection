from typing import Optional, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchGradCAM:
    """
    Grad-CAM Heatmap Generator for PyTorch ConvNeXt / EfficientNet models.
    Supports Python context manager protocol (`with PyTorchGradCAM(model) as gradcam:`)
    to guarantee hook cleanup and prevent RAM/VRAM leaks.
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
        last_layer: Optional[nn.Module] = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_layer = module
        if last_layer is None:
            raise ValueError("No Conv2d layer found in model for Grad-CAM")
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
        """
        Generates normalized Grad-CAM heatmap array [H, W] in range [0, 1].
        
        Args:
            input_tensor: Input tensor of shape [1, 3, H, W].
            target_class: Target class index (0: Fake, 1: Real).
            
        Returns:
            Normalized 2D float32 heatmap array [H, W].
        """
        self._register_hooks()
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                tensor_clone = input_tensor.detach().clone()
                tensor_clone.requires_grad = True
                
                output = self.model(tensor_clone)
                score = output[0] if output.ndim == 1 else output[0, 0]
                
                loss = score if target_class == 1 else -score
                loss.backward()
                self.model.zero_grad()

                if self.gradients is None or self.feature_maps is None:
                    return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

                weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)
                cam = torch.sum(weights * self.feature_maps[0], dim=0)
                cam = F.relu(cam)
                
                cam_np = cam.cpu().numpy()
                cam_np = cam_np - np.min(cam_np)
                cam_np = cam_np / (np.max(cam_np) + 1e-8)
                
                return cam_np
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
