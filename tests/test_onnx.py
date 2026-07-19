import os
import tempfile
import torch
import numpy as np
import pytest
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.onnx_exporter import export_to_onnx, ONNXDeepfakePredictor, HAS_ONNX

@pytest.mark.skipif(not HAS_ONNX, reason="onnx / onnxruntime not installed in environment")
def test_onnx_export_and_parity():
    model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=False, use_fft_branch=True)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "test_model.onnx")
        export_to_onnx(model, save_path=onnx_path, img_size=224)

        assert os.path.exists(onnx_path)

        predictor = ONNXDeepfakePredictor(onnx_path)

        dummy_tensor = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            torch_logits = model(dummy_tensor)
            torch_probs = torch.sigmoid(torch_logits).cpu().numpy()

        onnx_probs = predictor.predict_batch(dummy_tensor.numpy())

        # Test output parity between PyTorch and ONNX runtime
        np.testing.assert_allclose(torch_probs, onnx_probs, atol=1e-4, rtol=1e-3)
