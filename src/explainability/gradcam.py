import cv2
import numpy as np
import torch
import torch.nn.functional as F

class PyTorchGradCAM:
    """
    Grad-CAM Heatmap Generator for PyTorch ConvNeXt / EfficientNet models.
    Supports clean hook management to prevent memory leaks on cached models.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None

        if target_layer is None:
            # Auto-detect last Conv2d stage layer in backbone
            target_layer = self._find_last_conv_layer()
            
        self.target_layer = target_layer

    def _find_last_conv_layer(self):
        last_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_layer = module
        if last_layer is None:
            raise ValueError("No Conv2d layer found in model for Grad-CAM")
        return last_layer

    def _register_hooks(self):
        self._remove_hooks()
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def _forward_hook(self, module, input, output):
        self.feature_maps = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap array [H, W] in range [0, 1].
        Cleans up forward/backward hook handles immediately to prevent memory leaks.
        """
        self._register_hooks()
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                tensor_clone = input_tensor.detach().clone()
                tensor_clone.requires_grad = True
                
                output = self.model(tensor_clone)
                score = output[0] if output.ndim == 1 else output[0, 0]
                
                if target_class == 0:
                    loss = -score
                else:
                    loss = score

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

    def __del__(self):
        self._remove_hooks()

def overlay_cam(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays normalized [0, 1] heatmap on RGB uint8 image.
    """
    h, w, _ = image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    return blended
