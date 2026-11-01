"""Stage 8 of the Release-1 pipeline: robustness stress testing (JPEG, blur,
noise, downscaling) on N=1,000 balanced crops per regime.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from scripts.release1.common import subsample_balanced
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logger = logging.getLogger("release1_runner")

def run_robustness_stress_test(
    calibrated_path: str,
    data_root: str,
    splits_path: str,
    output_dir: str | None = None,
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robustness_path = os.path.join(output_dir, "robustness_results.json") if output_dir else None
    if not force_rerun and robustness_path and os.path.exists(robustness_path):
        logger.info("Found existing robustness results: %s. Skipping robustness.", robustness_path)
        with open(robustness_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if smoke_test:
        logger.info("--- [SMOKE TEST] Stress-Testing Model Resilience (10 Samples, 1 Setting per Distortion) ---")
    else:
        logger.info("--- Stress-Testing Model Resilience to Real-World Degradations ---")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_samples = splits["test"]

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    calib_meta = ckpt.get("calibration", {"optimal_temperature": 1.0})
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()
    scale_a = calib_meta.get("platt_scale_a", 1.0 / max(1e-6, calib_meta.get("optimal_temperature", 1.0)))
    bias_b = calib_meta.get("platt_bias_b", 0.0)

    # Sample a reproducible subset (e.g. 1,000 crops; 10 in smoke mode)
    n_samples = 10 if smoke_test else min(1000, len(test_samples))
    sample_subset = subsample_balanced(test_samples, n_samples, seed=42)

    def evaluate_degraded_images(degrade_fn: Any) -> float:
        y_true, y_probs = [], []
        for path, label in sample_subset:
            full_p = os.path.join(data_root, path)
            img = cv2.imread(full_p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_deg = degrade_fn(img)
            img_deg = cv2.resize(img_deg, (256, 256))
            tensor = torch.from_numpy(img_deg).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(tensor)
                if isinstance(out, tuple):
                    out = out[0]
                prob = torch.sigmoid(scale_a * out + bias_b).item()
            y_true.append(int(label))
            y_probs.append(prob)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_probs))

    # 1. JPEG
    jpeg_results = {}
    jpeg_qs = [70] if smoke_test else [90, 70, 50, 30]
    for q in jpeg_qs:
        def apply_jpeg(im: np.ndarray, quality: int = q) -> np.ndarray:
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        jpeg_results[f"q_{q}"] = evaluate_degraded_images(apply_jpeg)
        logger.info("Robustness JPEG Q=%d: AUC = %.4f", q, jpeg_results[f"q_{q}"])

    # 2. Gaussian Blur
    blur_results = {}
    blur_sigmas = [2.0] if smoke_test else [1.0, 2.0, 3.0, 4.0]
    for sigma in blur_sigmas:
        ksize = int(6 * sigma + 1) | 1
        def apply_blur(im: np.ndarray, k: int = ksize, s: float = sigma) -> np.ndarray:
            return cv2.GaussianBlur(im, (k, k), s)
        blur_results[f"sigma_{sigma}"] = evaluate_degraded_images(apply_blur)
        logger.info("Robustness Blur sigma=%.1f: AUC = %.4f", sigma, blur_results[f"sigma_{sigma}"])

    # 3. Additive Gaussian Noise
    noise_results = {}
    noise_sigmas = [10.0] if smoke_test else [5.0, 10.0, 15.0, 20.0]
    for sigma_n in noise_sigmas:
        def apply_noise(im: np.ndarray, s_n: float = sigma_n) -> np.ndarray:
            noise = np.random.normal(0, s_n, im.shape)
            return np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noise_results[f"sigma_{sigma_n}"] = evaluate_degraded_images(apply_noise)
        logger.info("Robustness Noise sigma=%.1f: AUC = %.4f", sigma_n, noise_results[f"sigma_{sigma_n}"])

    # 4. Downscaling
    downscale_results = {}
    downscale_scales = [4] if smoke_test else [2, 4, 8]
    for scale in downscale_scales:
        def apply_downscale(im: np.ndarray, sc: int = scale) -> np.ndarray:
            h, w = im.shape[:2]
            small = cv2.resize(im, (w // sc, h // sc), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        downscale_results[f"scale_{scale}x"] = evaluate_degraded_images(apply_downscale)
        logger.info("Robustness Downscale %dx: AUC = %.4f", scale, downscale_results[f"scale_{scale}x"])

    robustness_summary = {
        "jpeg": jpeg_results,
        "gaussian_blur": blur_results,
        "gaussian_noise": noise_results,
        "downscaling": downscale_results,
    }
    if robustness_path:
        with open(robustness_path, "w", encoding="utf-8") as f:
            json.dump(robustness_summary, f, indent=2)

    return robustness_summary


# ---------------------------------------------------------------------------
# Stage 8: CUDA Latency Profiling (Table 5)
# ---------------------------------------------------------------------------

