from typing import Optional
import argparse
import os
import time
import numpy as np
import torch

from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.onnx_exporter import export_to_onnx, ONNXDeepfakePredictor, HAS_ONNX

def benchmark_inference(iterations: int = 50, batch_size: int = 1, img_size: int = 256) -> None:
    """
    Inference latency benchmarking script comparing PyTorch Native vs ONNX Runtime.
    """
    config = load_config()
    print(f"\n==================================================")
    print(f"INFERENCE LATENCY BENCHMARK SUITE")
    print(f"==================================================")
    print(f"• Iterations: {iterations}")
    print(f"• Batch Size: {batch_size}")
    print(f"• Image Size: {img_size}x{img_size}")
    print(f"• PyTorch Version: {torch.__version__}")
    print(f"--------------------------------------------------\n")

    # 1. Initialize PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Initializing model on device: {device}...")
    backbone_name = config.get("model", {}).get("backbone", "convnext_base")
    pytorch_model = HybridDeepfakeDetector(backbone_name=backbone_name, pretrained=False, use_fft_branch=True, config=config)
    pytorch_model.to(device)
    pytorch_model.eval()

    dummy_torch = torch.randn(batch_size, 3, img_size, img_size, device=device)
    dummy_numpy = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)

    # Warmup PyTorch
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy_torch)

    # Measure PyTorch Latency
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = pytorch_model(dummy_torch)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    pytorch_total_ms = (end_time - start_time) * 1000.0
    pytorch_avg_ms = pytorch_total_ms / (iterations * batch_size)
    print(f"  PyTorch Native ({device.type.upper()}): {pytorch_avg_ms:.2f} ms / frame")

    # 2. Export & Benchmark ONNX Model
    onnx_path = "models/deepfake_convnext_v2.onnx"
    onnx_avg_ms: Optional[float] = None
    provider_str = "CPUExecutionProvider"
    if HAS_ONNX:
        try:
            if not os.path.exists(onnx_path):
                print(f"[ONNX] Exporting model to {onnx_path}...")
                os.makedirs("models", exist_ok=True)
                export_to_onnx(pytorch_model, save_path=onnx_path, img_size=img_size)

            onnx_predictor = ONNXDeepfakePredictor(onnx_path)
            provider_str = onnx_predictor.session.get_providers()[0]
            
            # Warmup ONNX
            for _ in range(5):
                _ = onnx_predictor.predict_batch(dummy_numpy)

            # Measure ONNX Latency
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = onnx_predictor.predict_batch(dummy_numpy)
            end_time = time.perf_counter()

            onnx_total_ms = (end_time - start_time) * 1000.0
            onnx_avg_ms = onnx_total_ms / (iterations * batch_size)
            speedup = pytorch_avg_ms / onnx_avg_ms if onnx_avg_ms > 0 else 1.0
            print(f"  ONNX Runtime ({provider_str}): {onnx_avg_ms:.2f} ms / frame (Speedup Factor: {speedup:.2f}x)")
        except Exception as e:
            print(f"  ONNX Benchmark Warning: {e}")
    else:
        print("  ONNX Runtime not installed. Skipping ONNX benchmark.")

    print(f"\n==================================================")
    print(f"SUMMARY LATENCY BENCHMARK RESULTS")
    print(f"==================================================")
    print(f"| Engine / Framework | Device / Provider | Latency per Frame | Speedup Factor |")
    print(f"| :--- | :---: | :---: | :---: |")
    print(f"| PyTorch Native | {device.type.upper()} | {pytorch_avg_ms:.2f} ms | 1.00x |")
    if onnx_avg_ms is not None:
        speedup = pytorch_avg_ms / onnx_avg_ms
        print(f"| ONNX Runtime | {provider_str} | {onnx_avg_ms:.2f} ms | **{speedup:.2f}x** |")
    print(f"==================================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Detector Latency Benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Benchmark batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    args = parser.parse_args()

    benchmark_inference(iterations=args.iterations, batch_size=args.batch_size, img_size=args.img_size)

if __name__ == "__main__":
    main()
