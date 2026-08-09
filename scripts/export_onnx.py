"""
Inference Latency & Throughput Benchmark Script for Dual-Stream Deepfake Detector Engine.
Measures single-crop latency (ms/crop) and throughput (FPS) at full 512x512 resolution.
"""

import os
import sys
import time
import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config

CONFIG = load_config()
IMG_SIZE = CONFIG.get("preprocessing", {}).get("img_size", 512)

def benchmark_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Multi-thread"
    print(f"[1/2] Instantiating PyTorch Model on {device_name} (Resolution: {IMG_SIZE}x{IMG_SIZE})...")
    
    model = HybridDeepfakeDetector().to(device)
    model.eval()

    weights_path = os.path.join(REPO_ROOT, "dual_stream_calibrated.pth")
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f" Loaded weights from {weights_path}")

    use_amp = (device.type == 'cuda')
    autocast_dtype = torch.float16 if use_amp else torch.float32

    # -----------------------------------------------------------------------
    # Benchmark Batch Size 1 (Real-time Single-Frame Latency)
    # -----------------------------------------------------------------------
    bs1_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    
    # Warmup Phase (5 passes)
    for _ in range(5):
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                _ = model(bs1_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_runs = 20
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(n_runs):
                    _ = model(bs1_input)
        end_event.record()
        torch.cuda.synchronize()
        bs1_latency = start_event.elapsed_time(end_event) / n_runs
    else:
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_runs):
                _ = model(bs1_input)
        bs1_latency = (time.perf_counter() - start_time) / n_runs * 1000.0

    bs1_fps = 1000.0 / bs1_latency

    # -----------------------------------------------------------------------
    # Benchmark Batch Size 32 (High-Throughput Batch Processing)
    # -----------------------------------------------------------------------
    bs32_input = torch.randn(32, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(3):
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
                _ = model(bs32_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_runs_batch = 5
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                for _ in range(n_runs_batch):
                    _ = model(bs32_input)
        end_event.record()
        torch.cuda.synchronize()
        bs32_total_time_ms = start_event.elapsed_time(end_event) / n_runs_batch
    else:
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_runs_batch):
                _ = model(bs32_input)
        bs32_total_time_ms = (time.perf_counter() - start_time) / n_runs_batch * 1000.0

    bs32_per_crop_ms = bs32_total_time_ms / 32.0
    bs32_fps = 1000.0 / bs32_per_crop_ms

    print("\n[2/2] Benchmark Performance Results:")
    print(f"  Device Hardware:             {device_name}")
    print(f"  Input Resolution:            {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Single-Crop Latency (BS=1):  {bs1_latency:.2f} ms/crop ({bs1_fps:.1f} FPS)")
    print(f"  Batch Throughput (BS=32):    {bs32_per_crop_ms:.2f} ms/crop ({bs32_fps:.1f} FPS)")

    return {
        "device": device_name,
        "img_size": IMG_SIZE,
        "bs1_latency_ms": bs1_latency,
        "bs1_fps": bs1_fps,
        "bs32_per_crop_ms": bs32_per_crop_ms,
        "bs32_fps": bs32_fps
    }

if __name__ == '__main__':
    benchmark_inference()
