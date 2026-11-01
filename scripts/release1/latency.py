"""Stage 9 of the Release-1 pipeline: single-T4 CUDA-synchronized latency
profiling (batch sizes 1 and 32).

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
import torch

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logger = logging.getLogger("release1_runner")

def run_latency_profiling(
    calibrated_path: str,
    output_dir: str | None = None,
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latency_path = os.path.join(output_dir, "latency_results.json") if output_dir else None
    if not force_rerun and latency_path and os.path.exists(latency_path):
        logger.info("Found existing latency results: %s. Skipping latency.", latency_path)
        with open(latency_path, "r", encoding="utf-8") as f:
            return json.load(f)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    is_tesla_t4 = "T4" in gpu_name.upper()
    logger.info("--- Running CUDA Latency Profiling on Device: %s (Is Tesla T4: %s) ---", gpu_name, is_tesla_t4)

    if not is_tesla_t4 and not smoke_test:
        logger.warning("Active device '%s' is NOT an NVIDIA Tesla T4. Refusing to write invalid latency metrics for Table 5.", gpu_name)
        results = {
            "device": gpu_name,
            "is_tesla_t4": False,
            "status": "SKIPPED_NOT_TESLA_T4",
            "message": "Execution device is not NVIDIA Tesla T4. Latency profiling requires a dedicated single T4 GPU.",
        }
        if latency_path:
            with open(latency_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return results

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()

    warmup_iters = 5 if smoke_test else 50
    timed_iters = 10 if smoke_test else 200

    results = {
        "device": gpu_name,
        "is_tesla_t4": is_tesla_t4,
    }
    for b_sz in [1, 32]:
        dummy = torch.randn(b_sz, 3, 256, 256, device=device)
        # Warmup
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed runs
        timings = []
        with torch.inference_mode():
            for _ in range(timed_iters):
                t_start = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t_start) * 1000.0)

        mean_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        fps = float(b_sz / (mean_ms / 1000.0))
        results[f"batch_size_{b_sz}"] = {
            "mean_latency_ms": mean_ms,
            "std_latency_ms": std_ms,
            "fps": fps,
        }
        logger.info("Latency B=%d on %s: %.2f ms +/- %.2f ms (%.1f FPS)", b_sz, gpu_name, mean_ms, std_ms, fps)

    if latency_path:
        with open(latency_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Stage 9: Master Turnkey Execution & Verification
# ---------------------------------------------------------------------------

