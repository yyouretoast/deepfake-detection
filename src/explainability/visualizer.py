import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List, Tuple
from sklearn.calibration import calibration_curve

class PyTorchGradCAM:
    """
    Grad-CAM context manager for VRAM-safe explainability on PyTorch spatial backbones.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._fwd_hook = None
        self._bwd_hook = None

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().cpu()

    def save_activation(self, module, input, output):
        self.activations = output.detach().cpu()

    def __enter__(self):
        self._bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        self._fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
        if self._bwd_hook is not None:
            self._bwd_hook.remove()
        self.gradients = None
        self.activations = None
        torch.cuda.empty_cache()

    def generate(self) -> np.ndarray:
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Forward and backward passes must be run within the context.")
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.numpy()

def plot_fft_spectrum_residuals(real_img: np.ndarray, fake_img: np.ndarray, save_path: Optional[str] = None):
    """
    Compute 2D Real FFT log-magnitude spectra log(|rfft2(gray)| + 1e-5) for Real vs. Fake images 
    and plot/save high-frequency residual heatmap comparisons.
    """
    def get_log_magnitude(img):
        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and img.shape[0] == 3:
            # Assuming CHW rgb
            img_hwc = np.transpose(img, (1, 2, 0))
            gray = cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        fft = np.fft.rfft2(gray)
        shifted = np.fft.fftshift(fft, axes=0)
        return np.log(np.abs(shifted) + 1e-5)

    real_mag = get_log_magnitude(real_img)
    fake_mag = get_log_magnitude(fake_img)
    residual = np.abs(real_mag - fake_mag)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    im1 = ax.imshow(real_mag, cmap='viridis')
    ax.set_title("Real FFT Log-Magnitude")
    fig.colorbar(im1, ax=ax)
    
    ax = axes[1]
    im2 = ax.imshow(fake_mag, cmap='viridis')
    ax.set_title("Fake FFT Log-Magnitude")
    fig.colorbar(im2, ax=ax)
    
    ax = axes[2]
    im3 = ax.imshow(residual, cmap='hot')
    ax.set_title("High-Frequency Residual")
    fig.colorbar(im3, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, y_prob_calibrated: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """
    Plot 15-bin Confidence vs. Accuracy bar chart overlaying uncalibrated and temperature-calibrated ECE scores.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    def plot_curve(true_labels, probs, label, color):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, probs, n_bins=15, strategy='uniform'
        )
        
        # Calculate ECE
        ece = 0
        bin_edges = np.linspace(0, 1, 16)
        for i in range(15):
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if i == 14:
                bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
            if np.sum(bin_mask) > 0:
                acc = np.mean(true_labels[bin_mask])
                conf = np.mean(probs[bin_mask])
                weight = np.sum(bin_mask) / len(probs)
                ece += weight * np.abs(acc - conf)
                
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{label} (ECE: {ece:.4f})", color=color)
        return fraction_of_positives, mean_predicted_value

    plot_curve(y_true, y_prob, "Uncalibrated", "blue")
    
    if y_prob_calibrated is not None:
        plot_curve(y_true, y_prob_calibrated, "Calibrated", "green")
        
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_ylabel("Accuracy (Fraction of Positives)")
    ax.set_xlabel("Confidence (Mean Predicted Value)")
    ax.set_title("Reliability Diagram (15 Bins)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)

def plot_explainability_grid(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, num_samples: int = 4, save_path: Optional[str] = None):
    """
    Generate a side-by-side figure (Original Image, Grad-CAM Heatmap, 2D FFT Magnitude Residual) for Real vs. Fake test samples.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    model.eval()
    
    target_layer = None
    if hasattr(model, 'spatial_backbone'):
        target_layer = list(model.spatial_backbone.children())[-1]
    else:
        target_layer = list(model.children())[-1]

    samples_processed = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(images.size(0)):
            if samples_processed >= num_samples:
                break
                
            img_tensor = images[i].unsqueeze(0)
            img_np = img_tensor.cpu().numpy().squeeze()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = np.transpose(img_np, (1, 2, 0)) # CHW to HWC
            
            with PyTorchGradCAM(model, target_layer) as cam:
                out = model(img_tensor)
                loss = out[0, out.argmax(dim=1).item()]
                loss.backward()
                heatmap = cam.generate()
            
            ax_orig = axes[samples_processed, 0]
            ax_cam = axes[samples_processed, 1]
            ax_fft = axes[samples_processed, 2]
            
            # 1. Original Image
            if img_np.shape[-1] == 1:
                ax_orig.imshow(img_np.squeeze(), cmap='gray')
            else:
                ax_orig.imshow(img_np)
            label_str = "Real" if labels[i].item() == 0 else "Fake"
            ax_orig.set_title(f"Original ({label_str})")
            ax_orig.axis('off')
            
            # 2. Grad-CAM Heatmap
            if img_np.shape[-1] == 1:
                ax_cam.imshow(img_np.squeeze(), cmap='gray')
            else:
                ax_cam.imshow(img_np)
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            ax_cam.imshow(heatmap_resized, cmap='jet', alpha=0.5)
            ax_cam.set_title("Grad-CAM Heatmap")
            ax_cam.axis('off')
            
            # 3. 2D FFT Magnitude Residual
            # For demonstration in the grid we'll just show the FFT of the current image
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            elif img_np.ndim == 3:
                gray = img_np.squeeze()
            else:
                gray = img_np
                
            fft = np.fft.rfft2(gray)
            shifted = np.fft.fftshift(fft, axes=0)
            mag = np.log(np.abs(shifted) + 1e-5)
            
            ax_fft.imshow(mag, cmap='viridis')
            ax_fft.set_title("2D FFT Magnitude")
            ax_fft.axis('off')
            
            samples_processed += 1
            
        if samples_processed >= num_samples:
            break
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
