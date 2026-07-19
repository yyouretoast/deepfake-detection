import os
import time
import numpy as np
import torch

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.onnx_exporter import export_to_onnx, ONNXDeepfakePredictor, HAS_ONNX

def benchmark_inference(iterations: int = 50, batch_size: int = 1, img_size: int = 224):
    """
    Empirical inference latency benchmarking script comparing PyTorch Native vs ONNX Runtime.
    """
    print(f"\n==================================================")
    print(f"🚀 DEEPFAKE DETECTION ENGINE - BENCHMARK SUITE")
    print(f"==================================================")
    print(f"• Iterations: {iterations}")
    print(f"• Batch Size: {batch_size}")
    print(f"• Image Size: {img_size}x{img_size}")
    print(f"• PyTorch Version: {torch.__version__}")
    print(f"--------------------------------------------------\n")

    # 1. Initialize PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] Initializing model on device: {device}...")
    pytorch_model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=False, use_fft_branch=True)
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
    print(f"  ✓ PyTorch Native ({device.type.upper()}): {pytorch_avg_ms:.2f} ms / frame")

    # 2. Export & Benchmark ONNX Model
    onnx_path = "deepfake_convnext_v2.onnx"
    onnx_avg_ms = None
    if HAS_ONNX:
        try:
            if not os.path.exists(onnx_path):
                print(f"[ONNX] Exporting model to {onnx_path}...")
                export_to_onnx(pytorch_model, save_path=onnx_path, img_size=img_size)

            onnx_predictor = ONNXDeepfakePredictor(onnx_path)
            
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
            print(f"  ✓ ONNX Runtime Acceleration: {onnx_avg_ms:.2f} ms / frame ({speedup:.1f}x Speedup ⚡)")
        except Exception as e:
            print(f"  ⚠️ ONNX Benchmark Warning: {e}")
    else:
        print("  ⚠️ ONNX Runtime not installed. Skipping ONNX benchmark.")

    print(f"\n==================================================")
    print(f"📊 SUMMARY LATENCY BENCHMARK RESULTS")
    print(f"==================================================")
    print(f"| Engine / Framework | Device | Latency per Frame | Speedup |")
    print(f"| :--- | :---: | :---: | :---: |")
    print(f"| PyTorch Native | {device.type.upper()} | {pytorch_avg_ms:.2f} ms | 1.0x |")
    if onnx_avg_ms is not None:
        speedup = pytorch_avg_ms / onnx_avg_ms
        print(f"| ONNX Runtime | {device.type.upper()} | {onnx_avg_ms:.2f} ms | **{speedup:.1f}x Faster** ⚡ |")
    print(f"==================================================\n")

if __name__ == "__main__":
    benchmark_inference(iterations=50, batch_size=1, img_size=224)
