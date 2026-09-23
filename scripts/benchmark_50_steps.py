"""Benchmark training steps of Dual-Stream detector with batch size 32."""

import os
import sys
import time
import torch
from torch import nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.optimization import get_differential_param_groups

def run_benchmark(batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 30 if device.type == "cuda" else 3
    print(f"Benchmarking on device: {device} for {num_steps} steps with batch_size={batch_size}")
    img_size = 256
    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse").to(device)
    optimizer = torch.optim.AdamW(
        get_differential_param_groups(model, lr_backbone=1e-5, lr_head=1e-4),
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(f"Testing forward + backward with batch_size={batch_size} (3 x {img_size} x {img_size})...")
    step_times = []
    start_total = time.time()
    for step in range(num_steps):
        t0 = time.time()
        x = torch.rand(batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out, aux = model(x, return_aux=True)
            loss = criterion(out, y) + 0.3 * criterion(aux, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        if (step + 1) % 10 == 0 or step < 5:
            print(f"  Step {step+1}/{num_steps}: {dt*1000.0:.1f} ms")
    
    total_time = time.time() - start_total
    avg_step_ms = (sum(step_times) / len(step_times)) * 1000.0
    throughput = (batch_size * num_steps) / total_time
    
    print(f"{num_steps} Steps Completed in {total_time:.2f}s")
    print(f"Average Step Latency: {avg_step_ms:.1f} ms/step")
    print(f"Training Throughput: {throughput:.1f} crops/second")
    if device.type == "cuda":
        curr_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        curr_res = torch.cuda.memory_reserved() / (1024 ** 2)
        peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"GPU Memory Analysis:")
        print(f"  - Peak Memory Allocated (Activations + Weights + Gradients): {peak_alloc:.1f} MB")
        print(f"  - End-of-Step Persistent Allocated (Weights + Optimizer States): {curr_alloc:.1f} MB")
        print(f"  - Peak Memory Reserved (CUDA Caching Allocator Pool): {peak_res:.1f} MB")
        print(f"  - Current Memory Reserved: {curr_res:.1f} MB")
        
    return avg_step_ms, throughput

if __name__ == "__main__":
    b_sz = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    run_benchmark(batch_size=b_sz)
