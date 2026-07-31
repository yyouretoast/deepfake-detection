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
    Fully vectorized for GPU batch execution. Scales heatmaps to input tensor resolution.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, target_stream: str = "spatial") -> None:
        self.model = model
        self.target_stream = target_stream
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model

        if target_layer is None:
            if target_stream == "spatial":
                stream_module = getattr(unwrapped, "spatial_backbone", None)
                if stream_module is not None and hasattr(stream_module, "stages"):
                    target_layer = stream_module.stages[-1]
                elif stream_module is not None and hasattr(stream_module, "conv_head"):
                    target_layer = stream_module.conv_head
            elif target_stream == "frequency":
                stream_module = getattr(unwrapped, "freq_extractor", None)
                if stream_module is not None and hasattr(stream_module, "conv_net"):
                    target_layer = stream_module.conv_net[-1]

            if target_layer is not None:
                # Find leaf Conv2d
                last_conv = None
                for name, module in target_layer.named_modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                if last_conv is not None:
                    target_layer = last_conv

            if target_layer is None:
                # Fallback to last conv in the whole unwrapped model
                for name, module in unwrapped.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module

        if target_layer is None:
            raise ValueError("Could not automatically locate target Conv2d layer for Grad-CAM.")

        self.target_layer = target_layer
        self.fh = self.target_layer.register_forward_hook(self._save_feature_maps)
        self.bh = self.target_layer.register_full_backward_hook(self._save_gradients)

    def remove(self):
        if hasattr(self, "fh") and self.fh is not None:
            self.fh.remove()
            self.fh = None
        if hasattr(self, "bh") and self.bh is not None:
            self.bh.remove()
            self.bh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def __del__(self):
        self.remove()

    def _save_feature_maps(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if output.ndim == 4 and output.shape[1] != output.shape[-1] and output.shape[-1] == output.shape[-2]:
            self.feature_maps = output
        elif output.ndim == 4 and output.shape[-1] != output.shape[1]:
            self.feature_maps = output.permute(0, 3, 1, 2)
        else:
            self.feature_maps = output

    def _save_gradients(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
        if grad_output is None or len(grad_output) == 0 or grad_output[0] is None: return
        grad = grad_output[0]
        if grad.ndim == 4 and grad.shape[1] != grad.shape[-1] and grad.shape[-1] == grad.shape[-2]:
            self.gradients = grad
        elif grad.ndim == 4 and grad.shape[-1] != grad.shape[1]:
            self.gradients = grad.permute(0, 3, 1, 2)
        else:
            self.gradients = grad

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Generates single Grad-CAM heatmap array normalized to [0.0, 1.0]."""
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 4 and input_tensor.shape[0] != 1 and input_tensor.shape[1] != 3:
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
        Generates Grad-CAM heatmaps for a batch of images/videos.
        Returns a list of numpy arrays normalized to [0.0, 1.0].
        """
        was_training = self.model.training
        self.gradients = None
        self.feature_maps = None
        
        batch_size = input_tensor_batch.shape[0]
        if target_classes is None:
            target_classes = [target_class] * batch_size
        elif isinstance(target_classes, int):
            target_classes = [target_classes] * batch_size
            
        if len(target_classes) != batch_size:
            raise ValueError(f"len(target_classes)={len(target_classes)} must match batch_size={batch_size}")

        try:
            with torch.enable_grad():
                self.model.eval()
                unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
                
                # Infer device
                model_device = next(unwrapped.parameters()).device
                
                input_tensors = input_tensor_batch.to(model_device).clone().detach().requires_grad_(True)
                unwrapped.zero_grad()
                
                outputs = unwrapped(input_tensors)
                
                # Handling batch outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                scores = []
                for i in range(batch_size):
                    z_i = outputs[i]
                    if target_classes[i] == 1:
                        scores.append(z_i)
                    else:
                        scores.append(-z_i)
                        
                scores = torch.stack(scores)
                loss = torch.sum(scores)
                loss.backward()

            if self.feature_maps is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture feature maps or gradients.")

            target_h, target_w = input_tensor_batch.shape[-2], input_tensor_batch.shape[-1]
            is_video = input_tensor_batch.ndim == 5
            T = input_tensor_batch.shape[1] if is_video else 1
            
            fmaps = self.feature_maps.to(torch.float32)
            grads = self.gradients.to(torch.float32)

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            cams = F.relu(torch.sum(weights * fmaps, dim=1, keepdim=True))  # [B, 1, H, W]
            cams_upsampled = F.interpolate(
                cams,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

            cams_np = cams_upsampled.squeeze(1).detach().cpu().numpy()
            if is_video and cams_np.shape[0] == batch_size * T:
                cams_np = cams_np.reshape(batch_size, T, target_h, target_w)
                
            heatmaps = []
            for b in range(batch_size):
                cam = cams_np[b]
                c_min, c_max = cam.min(), cam.max()
                if (c_max - c_min) > 1e-6:
                    norm_cam = (cam - c_min) / (c_max - c_min)
                else:
                    norm_cam = np.zeros_like(cam)
                    
                norm_cam = np.nan_to_num(norm_cam, nan=0.0)
                if is_video and norm_cam.ndim == 2:
                    norm_cam = np.stack([norm_cam] * T, axis=0)
                heatmaps.append(norm_cam)

            return heatmaps
        finally:
            self.gradients = None
            self.feature_maps = None
            self.model.train(was_training)

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

        heatmap = np.nan_to_num(heatmap, nan=0.0)
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            if rgb_image.max() <= 1.0:
                rgb_image = np.uint8(255 * rgb_image)
            else:
                rgb_image = np.uint8(rgb_image)

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return overlay

overlay_cam = PyTorchGradCAM.overlay_heatmap
