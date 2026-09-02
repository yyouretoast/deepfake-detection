"""Robustness evaluation script for Dual-Stream Deepfake Detector under real-world distortions."""

import argparse
from collections.abc import Callable
import json
import logging
import os
import sys
from typing import Optional

import cv2
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import dedupe_split
from src.dataset.resolver import resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import compute_roc_auc_safe
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
RobustnessDataset = FaceCropDataset


def jpeg_fn(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    """JPEG compression degradation at specified quality level (0-100)."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return fn


def blur_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Gaussian blur degradation at specified kernel sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        ksize = int(6 * sigma + 1) | 1
        return cv2.GaussianBlur(rgb, (ksize, ksize), sigma)

    return fn


def noise_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Additive Gaussian noise degradation at specified pixel-space sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*rgb.shape).astype(np.float32) * sigma
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return fn


def downscale_fn(scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """Downscale and re-upsample image back to original resolution."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = cv2.resize(
            rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return fn


def run_eval(
    model: torch.nn.Module,
    samples: list[tuple[str, float]],
    root_dir: str,
    degradation_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    threshold: float,
    temperature: float,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
) -> tuple[float, float, int]:
    """Run an inference pass over evaluation samples under specified degradation."""
    if max_samples:
        reals = [s for s in samples if s[1] == 0]
        fakes = [s for s in samples if s[1] == 1]
        n_each = max_samples // 2
        capped = reals[:n_each] + fakes[:n_each]
    else:
        capped = samples

    dataset = FaceCropDataset(capped, root_dir, is_train=False, degradation_fn=degradation_fn)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    evaluator = ModelEvaluator(model, device=device)
    logits, targets, valid = evaluator.predict_loader(loader)

    mask = valid > 0.0
    logits = logits[mask]
    targets = targets[mask]

    if len(np.unique(targets)) < 2:
        logger.warning("Only one class present in evaluated samples - AUC undefined.")
        return float("nan"), float("nan"), len(targets)

    probs = 1.0 / (1.0 + np.exp(-(logits / temperature)))
    preds = (probs >= threshold).astype(int)

    auc = compute_roc_auc_safe(targets, probs)
    f1 = float(f1_score(targets, preds, zero_division=0))
    return auc, f1, len(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation for Dual-Stream Deepfake Detector.")
    parser.add_argument("--checkpoint", required=True, help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--data_root", required=True, help="Dataset root containing splits.json")
    parser.add_argument("--output_json", default="robustness_results.json", help="Output path for JSON results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap test samples for fast local runs (e.g. 500)")
    args = parser.parse_args()

    splits_path = resolve_splits_path(data_root=args.data_root)
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_samples = dedupe_split(splits["test"])
    logger.info("Loaded %d test samples from %s", len(test_samples), splits_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    threshold = float(ckpt.get("optimal_threshold", 0.50))
    temperature = float(ckpt.get("temperature", 1.0))
    logger.info("Loaded threshold=%.4f, temperature=%.4f from checkpoint", threshold, temperature)

    model = HybridDeepfakeDetector(pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state), strict=False)
    model.eval()

    results: dict[str, dict[str, dict[str, float]]] = {}

    # Baseline (no degradation)
    logger.info("--- Baseline Evaluation (No Degradation) ---")
    b_auc, b_f1, n = run_eval(model, test_samples, args.data_root, None, threshold, temperature, device, args.batch_size, args.max_samples)
    results["baseline"] = {"clean": {"auc": b_auc, "f1": b_f1, "samples": float(n)}}
    logger.info("Baseline -> AUC: %.4f | F1: %.4f (N=%d)", b_auc, b_f1, n)

    # 1. JPEG Compression
    logger.info("--- 1. JPEG Compression Sweep ---")
    results["jpeg_compression"] = {}
    for q in [90, 80, 70, 60, 50, 40, 30]:
        auc, f1, n = run_eval(model, test_samples, args.data_root, jpeg_fn(q), threshold, temperature, device, args.batch_size, args.max_samples)
        results["jpeg_compression"][f"q_{q}"] = {"auc": auc, "f1": f1, "samples": float(n)}
        logger.info("  JPEG Q=%2d -> AUC: %.4f | F1: %.4f", q, auc, f1)

    # 2. Gaussian Blur
    logger.info("--- 2. Gaussian Blur Sweep ---")
    results["gaussian_blur"] = {}
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        auc, f1, n = run_eval(model, test_samples, args.data_root, blur_fn(sigma), threshold, temperature, device, args.batch_size, args.max_samples)
        results["gaussian_blur"][f"sigma_{sigma}"] = {"auc": auc, "f1": f1, "samples": float(n)}
        logger.info("  Blur sigma=%.1f -> AUC: %.4f | F1: %.4f", sigma, auc, f1)

    # 3. Gaussian Noise
    logger.info("--- 3. Gaussian Noise Sweep ---")
    results["gaussian_noise"] = {}
    for sigma in [5.0, 10.0, 15.0, 20.0, 30.0]:
        auc, f1, n = run_eval(model, test_samples, args.data_root, noise_fn(sigma), threshold, temperature, device, args.batch_size, args.max_samples)
        results["gaussian_noise"][f"sigma_{sigma}"] = {"auc": auc, "f1": f1, "samples": float(n)}
        logger.info("  Noise sigma=%4.1f -> AUC: %.4f | F1: %.4f", sigma, auc, f1)

    # 4. Downscaling
    logger.info("--- 4. Downscale Sweep ---")
    results["downscale"] = {}
    for scale in [0.75, 0.50, 0.33, 0.25]:
        auc, f1, n = run_eval(model, test_samples, args.data_root, downscale_fn(scale), threshold, temperature, device, args.batch_size, args.max_samples)
        results["downscale"][f"scale_{scale}"] = {"auc": auc, "f1": f1, "samples": float(n)}
        logger.info("  Scale=%.2fx -> AUC: %.4f | F1: %.4f", scale, auc, f1)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Robustness results written to %s", args.output_json)


if __name__ == "__main__":
    main()
