"""Inference Latency & Throughput Benchmark Script for Dual-Stream Deepfake Detector Engine."""

import argparse
import gc
import os
import sys
import time
from typing import Any, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.resolver import find_weights_path
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict


def benchmark_inference(
    weights_path: Optional[str] = None,
    img_size: int = 256,
    batch_size: int = 32,
    device_str: Optional[str] = None,
) -> dict[str, Any]:
    """Benchmark model inference latency and throughput for batch sizes 1 and batch_size."""
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Multi-thread"
    print(f"[1/2] Instantiating PyTorch Model on {device_name} (Resolution: {img_size}x{img_size})...")

    model = HybridDeepfakeDetector(pretrained=False).to(device)
    model.eval()

    try:
        resolved_weights = find_weights_path(weights_path)
        ckpt = torch.load(resolved_weights, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        print(f" Loaded weights from {resolved_weights}")
    except FileNotFoundError:
        print(" Running with uninitialized random weights (dry-run mode).")

    use_amp = device.type == "cuda"
    autocast_dtype = torch.float16 if use_amp else torch.float32

    # Benchmark Batch Size 1 (Real-time Single-Frame Latency)
    if device.type == "cuda":
        bs1_input = torch.randn(1, 3, img_size, img_size).pin_memory().to(device, non_blocking=True)
    else:
        bs1_input = torch.randn(1, 3, img_size, img_size, device=device)

    warmup_runs = 15 if device.type == "cuda" else 3
    for _ in range(warmup_runs):
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                _ = model(bs1_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_runs = 30 if device.type == "cuda" else 5
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                for _ in range(n_runs):
                    _ = model(bs1_input)
        end_event.record()
        torch.cuda.synchronize()
        total_time_s = start_event.elapsed_time(end_event) / 1000.0
    else:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_runs):
                _ = model(bs1_input)
        total_time_s = time.perf_counter() - t0

    latency_bs1_ms = (total_time_s / n_runs) * 1000.0
    fps_bs1 = n_runs / total_time_s
    print(f"  Single-frame (BS=1):  {latency_bs1_ms:.2f} ms/frame  ({fps_bs1:.1f} FPS)")

    # Benchmark Batch Size N (Batch Throughput)
    if device.type == "cuda":
        bsN_input = torch.randn(batch_size, 3, img_size, img_size).pin_memory().to(device, non_blocking=True)
    else:
        bsN_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for _ in range(warmup_runs):
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                _ = model(bsN_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                for _ in range(n_runs):
                    _ = model(bsN_input)
        end_event.record()
        torch.cuda.synchronize()
        total_time_s = start_event.elapsed_time(end_event) / 1000.0
    else:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_runs):
                _ = model(bsN_input)
        total_time_s = time.perf_counter() - t0

    throughput_fps = (batch_size * n_runs) / total_time_s
    latency_per_sample_ms = (total_time_s / (batch_size * n_runs)) * 1000.0
    print(f"  Batched (BS={batch_size}):      {throughput_fps:.1f} FPS  ({latency_per_sample_ms:.2f} ms/frame)")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device_name,
        "latency_bs1_ms": latency_bs1_ms,
        "fps_bs1": fps_bs1,
        "throughput_fps": throughput_fps,
        "latency_per_sample_ms": latency_per_sample_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Dual-Stream detector inference latency and throughput.")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint weights")
    parser.add_argument("--img_size", type=int, default=256, help="Input resolution (default: 256)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for throughput testing")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    args = parser.parse_args()

    benchmark_inference(
        weights_path=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
