"""Visual Interpretability & Attention Map Generator for Dual-Stream Deepfake Detector."""

import argparse
import json
import logging
import os
import random
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.dataset.loader import dedupe_split
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMG_SIZE = 256
DEFAULT_THRESHOLD = 0.01
DEFAULT_TEMPERATURE = 1.4788


class ConvNeXtGradCAM:
    def __init__(self, model: HybridDeepfakeDetector) -> None:
        self.model = model
        self.feature_maps: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer = self.model.spatial_backbone[-1]
        self.forward_handle = target_layer.register_forward_hook(self._save_feature_maps)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.feature_maps = output

    def _save_gradients(
        self, module: torch.nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            logits = self.model(input_tensor)
            scalar_logit = logits.squeeze()
            scalar_logit.backward()

        if self.feature_maps is None or self.gradients is None:
            logging.warning("Grad-CAM feature maps or gradients not captured.")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i]

        cam = F.relu(cam).detach().cpu().numpy()
        denom = cam.max() - cam.min()
        if denom > 1e-6:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def generate_4panel_figure(
    model: HybridDeepfakeDetector,
    grad_cam: ConvNeXtGradCAM,
    rgb_uint8: np.ndarray,
    label_str: str,
    output_path: str,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    model.eval()

    img_tensor = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        srm_out = model.srm(img_tensor)
        bayar_out = model.bayar(img_tensor)
        noise_combined = torch.cat([srm_out, bayar_out], dim=1)
        freq_maps = model.fft(noise_combined)

        mean = model.imagenet_mean.to(dtype=img_tensor.dtype)
        std = model.imagenet_std.to(dtype=img_tensor.dtype)
        x_spatial = (img_tensor - mean) / std
        f_s = model.spatial_pool(model.spatial_backbone(x_spatial)).flatten(1)
        f_s = model.spatial_fc(f_s)

        f_f = model.freq_conv(freq_maps).flatten(1)
        f_f = model.freq_fc(f_f)

        concat_feat = torch.cat([f_s, f_f], dim=1)
        gate = model.gate_fc(concat_feat)
        gate_mean = float(gate.mean().item())

        logits = model(img_tensor).squeeze(-1).float()
        raw_logit = float(logits.item())
        calibrated_prob = float(torch.sigmoid(logits / temperature).item())
        pred_label = "FAKE" if calibrated_prob > threshold else "REAL"

    cam_map = grad_cam.generate_heatmap(img_tensor.clone())

    srm_map = srm_out[0].abs().mean(dim=0).cpu().numpy()
    srm_norm = (srm_map - srm_map.min()) / max(srm_map.max() - srm_map.min(), 1e-6)

    mag_maps = freq_maps[0, :10].cpu().numpy()
    mean_mag = np.mean(mag_maps, axis=0)
    fft_centered = np.fft.fftshift(mean_mag)

    cam_uint8 = np.uint8(255 * cam_map)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    gradcam_overlay = cv2.addWeighted(rgb_uint8, 0.6, heatmap_rgb, 0.4, 0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)

    axes[0, 0].imshow(rgb_uint8)
    axes[0, 0].set_title("(a) Input RGB Face Crop (256x256)", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(srm_norm, cmap="magma")
    axes[0, 1].set_title("(b) SRM Noise Residual Map (9 Filters)", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    im_c = axes[1, 0].imshow(fft_centered, cmap="viridis")
    axes[1, 0].set_title(f"(c) 2D FFT Magnitude Spectrum (Gate={gate_mean:.3f})", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im_c, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(gradcam_overlay)
    axes[1, 1].set_title("(d) ConvNeXt Grad-CAM Attention Overlay", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    correct_symbol = "✓" if pred_label == label_str else "✗"
    title_color = "#16A34A" if pred_label == label_str else "#DC2626"
    fig.suptitle(
        f"Ground Truth: {label_str}  |  Pred: {pred_label} {correct_symbol}  "
        f"(p = {calibrated_prob:.4f}, logit = {raw_logit:+.2f}, T* = {temperature:.4f}, thresh = {threshold:.2f})",
        fontsize=11, fontweight="bold", color=title_color, y=0.98
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info("Saved attention map figure -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 4-panel visual interpretability diagnostic figures."
    )
    parser.add_argument("--checkpoint", default="dual_stream_calibrated.pth", help="Path to trained model weights checkpoint")
    parser.add_argument("--data_root", default="data/cropped", help="Dataset root containing splits.json")
    parser.add_argument("--output_dir", default="figures/attention_maps", help="Output directory for rendered figures")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of sample diagnostic figures to generate")
    parser.add_argument("--image_path", type=str, default=None, help="Optional single image path for one-off visualization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    config = load_config()
    backbone = config.get("model", {}).get("backbone", "convnext_small")
    model = HybridDeepfakeDetector(
        backbone_name=backbone, pretrained=False, use_fft_branch=True, config=config
    )

    threshold = DEFAULT_THRESHOLD
    temperature = DEFAULT_TEMPERATURE

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        threshold = float(checkpoint.get("optimal_threshold", DEFAULT_THRESHOLD))
        temperature = float(checkpoint.get("temperature", DEFAULT_TEMPERATURE))
        logging.info("Loaded checkpoint '%s' (Threshold=%.4f, Temp=%.4f)", args.checkpoint, threshold, temperature)
    else:
        logging.warning("Checkpoint '%s' not found. Running with initial model weights.", args.checkpoint)

    model.to(device).eval()
    grad_cam = ConvNeXtGradCAM(model)

    if args.image_path is not None:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found at {args.image_path}")
        bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        out_name = os.path.join(args.output_dir, "single_image_attention.png")
        generate_4panel_figure(
            model, grad_cam, rgb, "UNKNOWN", out_name, device, threshold, temperature
        )
        grad_cam.remove_hooks()
        return

    splits_path = os.path.join(args.data_root, "splits.json")
    samples = []
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        test_samples = dedupe_split(splits.get("test", []))
        reals = [s for s in test_samples if s[1] == 0]
        fakes = [s for s in test_samples if s[1] == 1]

        n_each = args.n_samples // 2
        random.seed(42)
        sampled_reals = random.sample(reals, min(n_each, len(reals)))
        sampled_fakes = random.sample(fakes, min(n_each, len(fakes)))
        samples = [(path, "REAL" if label == 0 else "FAKE") for path, label in sampled_reals + sampled_fakes]

    if not samples:
        logging.warning("No dataset splits found. Generating synthetic diagnostic sample figures.")
        for idx in range(args.n_samples):
            label_str = "FAKE" if idx % 2 == 1 else "REAL"
            img = np.random.randint(40, 220, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            cv2.circle(img, (128, 128), 80, (200, 180, 160), -1)
            cv2.circle(img, (95, 105), 12, (50, 50, 50), -1)
            cv2.circle(img, (161, 105), 12, (50, 50, 50), -1)
            cv2.ellipse(img, (128, 160), (35, 15), 0, 0, 180, (150, 50, 50), 4)
            out_path = os.path.join(args.output_dir, f"attention_map_{idx+1:02d}_{label_str.lower()}.png")
            generate_4panel_figure(
                model, grad_cam, img, label_str, out_path, device, threshold, temperature
            )
    else:
        for idx, (rel_path, label_str) in enumerate(samples):
            full_path = os.path.join(args.data_root, rel_path)
            if not os.path.exists(full_path):
                continue
            bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
                rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(args.output_dir, f"attention_map_{idx+1:02d}_{label_str.lower()}.png")
            generate_4panel_figure(
                model, grad_cam, rgb, label_str, out_path, device, threshold, temperature
            )

    grad_cam.remove_hooks()
    logging.info("All diagnostic figures rendered to %s/", args.output_dir)


if __name__ == "__main__":
    main()
