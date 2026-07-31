import os
import tempfile
import numpy as np
import torch
import pytest

from src.models.hybrid_detector import build_model
from src.models.onnx_exporter import export_to_onnx, HAS_ONNX, ONNXDeepfakePredictor

@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
def test_onnx_export_4d_single_frame():
    model = build_model(use_fft=False, pretrained=False)
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test_model_4d.onnx")
        export_to_onnx(model, save_path=onnx_path, img_size=256)
        
        assert os.path.exists(onnx_path)
        predictor = ONNXDeepfakePredictor(onnx_path)
        
        dummy_np = np.random.randn(2, 3, 256, 256).astype(np.float32)
        probs = predictor.predict_batch(dummy_np)
        assert probs.shape == (2,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime not installed")
def test_onnx_export_5d_sequence():
    model = build_model(use_fft=False, pretrained=False)
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test_model_5d.onnx")
        dummy_5d = torch.randn(1, 4, 3, 256, 256)
        export_to_onnx(model, save_path=onnx_path, img_size=256, dummy_input=dummy_5d)
        
        assert os.path.exists(onnx_path)
        predictor = ONNXDeepfakePredictor(onnx_path)
        
        # Test 5D video sequence ONNX batch inference (T=4)
        dummy_seq4 = np.random.randn(1, 4, 3, 256, 256).astype(np.float32)
        probs4 = predictor.predict_batch(dummy_seq4)
        assert probs4.shape == (1,)
        assert (probs4 >= 0.0).all() and (probs4 <= 1.0).all()

