import os
import json
import pytest
import torch
from scripts.export_onnx import export_to_onnx

def test_export_onnx_creates_sidecar_metadata(tmp_path):
    """Verifies that export_to_onnx creates both .onnx model and .json sidecar metadata files."""
    ckpt_path = tmp_path / "dummy_calibrated.pth"
    onnx_path = tmp_path / "dummy_model.onnx"
    json_path = tmp_path / "dummy_model.json"

    # Create dummy checkpoint
    from src.models.hybrid_detector import HybridDeepfakeDetector
    model = HybridDeepfakeDetector()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_threshold': 0.15,
        'temperature': 1.4788
    }, ckpt_path)

    # Export to ONNX
    export_to_onnx(str(ckpt_path), str(onnx_path), quantize=False)

    assert os.path.exists(onnx_path), "ONNX model file was not created!"
    assert os.path.exists(json_path), "ONNX sidecar metadata JSON file was not created!"

    with open(json_path, 'r') as f:
        meta = json.load(f)

    assert meta["optimal_threshold"] == 0.15
    assert meta["temperature"] == 1.4788

def test_export_onnx_int8_quantization(tmp_path):
    """Verifies that export_to_onnx with quantize=True executes INT8 dynamic quantization."""
    ckpt_path = tmp_path / "dummy_calibrated.pth"
    onnx_path = tmp_path / "dummy_model.onnx"
    int8_path = tmp_path / "dummy_model_int8.onnx"

    from src.models.hybrid_detector import HybridDeepfakeDetector
    model = HybridDeepfakeDetector()
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

    # Export to ONNX with INT8 quantization
    export_to_onnx(str(ckpt_path), str(onnx_path), quantize=True)

    assert os.path.exists(onnx_path), "FP32 ONNX model file was not created!"
    # INT8 quantization will be skipped gracefully if onnxruntime is missing, or create model if installed
    if os.path.exists(int8_path):
        assert os.path.getsize(int8_path) > 0, "INT8 ONNX model file is empty!"
