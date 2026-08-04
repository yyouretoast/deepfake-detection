"""
ONNX Model Exporter, INT8 Quantizer, Sidecar Metadata Saver & Inference Verifier for Dual-Stream Deepfake Detector.

Exports calibrated PyTorch model state_dict to dynamic-axes ONNX format (opset 17),
saves calibration sidecar metadata JSON (temperature T*, threshold), performs INT8 dynamic quantization,
and verifies ONNX Runtime prediction agreement for both FP32 and INT8 formats.

Usage:
    python scripts/export_onnx.py --checkpoint models/dual_stream_calibrated.pth --output models/dual_stream_detector.onnx --quantize
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import logging
import torch
import numpy as np
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config()

IMG_SIZE = CONFIG.get("preprocessing", {}).get("img_size", 256)

def export_to_onnx(checkpoint_path: str, output_path: str, quantize: bool = False):
    device = torch.device("cpu")
    model = HybridDeepfakeDetector().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'!")

    ckpt = torch.load(checkpoint_path, map_location=device)
    threshold = 0.5
    temp = 1.0

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = float(ckpt.get("optimal_threshold", 0.5))
        temp = float(ckpt.get("temperature", 1.0))
    else:
        model.load_state_dict(ckpt)

    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save Sidecar Calibration Metadata JSON
    meta_path = output_path.replace(".onnx", ".json")
    metadata = {
        "optimal_threshold": threshold,
        "temperature": temp,
        "img_size": IMG_SIZE,
        "backbone": CONFIG.get("model", {}).get("backbone", "convnext_small"),
        "use_fft_branch": CONFIG.get("model", {}).get("use_fft_branch", True)
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("💾 Saved calibration sidecar metadata to '%s' (Threshold=%.2f, Temp=%.4f)", meta_path, threshold, temp)

    logging.info("🚀 Exporting PyTorch model (resolution %dx%d) to ONNX format (Opset 17) at '%s'...", IMG_SIZE, IMG_SIZE, output_path)

    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
    except Exception as e:
        logging.error("❌ ONNX Export failed during PyTorch C++ graph tracing: %s", e)
        raise e

    fp32_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info("✅ ONNX FP32 export successful! File size: %.2f MB", fp32_size_mb)

    # Verify ONNX Runtime FP32 agreement
    try:
        import onnxruntime as ort
        logging.info("🔍 Verifying ONNX Runtime FP32 logit agreement with PyTorch...")
        
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_outs = model(dummy_input).numpy()

        diff_fp32 = np.max(np.abs(torch_outs - ort_outs))
        logging.info("📊 Maximum Logit Difference (PyTorch vs ONNX FP32): %.6f", diff_fp32)

        if diff_fp32 < 1e-3:
            logging.info("✅ ONNX Runtime FP32 verification PASSED!")
        else:
            logging.warning("⚠️ High difference between PyTorch and ONNX predictions: %.6f", diff_fp32)

        # INT8 Dynamic Quantization & Verification
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = output_path.replace(".onnx", "_int8.onnx")
            logging.info("⚡ Performing INT8 Dynamic Quantization -> '%s'...", quant_path)
            
            quantize_dynamic(
                model_input=output_path,
                model_output=quant_path,
                weight_type=QuantType.QUInt8
            )

            int8_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
            logging.info("✅ INT8 Quantization successful! Reduced size: %.2f MB (%.1f%% reduction)", 
                         int8_size_mb, (1.0 - int8_size_mb / fp32_size_mb) * 100)

            # Verify INT8 Runtime agreement
            session_int8 = ort.InferenceSession(quant_path, providers=['CPUExecutionProvider'])
            int8_outs = session_int8.run(None, ort_inputs)[0]
            diff_int8 = np.max(np.abs(torch_outs - int8_outs))
            logging.info("📊 Maximum Logit Error Margin (PyTorch vs INT8 ONNX): %.6f", diff_int8)

    except ImportError:
        logging.warning("onnxruntime package not installed; skipping ONNX Runtime prediction verification and INT8 quantization.")

def main():
    parser = argparse.ArgumentParser(description="ONNX Exporter for Dual-Stream Deepfake Detector")
    parser.add_argument("--checkpoint", type=str, default="models/dual_stream_calibrated.pth", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="models/dual_stream_detector.onnx", help="Output path for ONNX file")
    parser.add_argument("--quantize", action="store_true", help="Perform INT8 dynamic quantization")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, quantize=args.quantize)

if __name__ == "__main__":
    main()
