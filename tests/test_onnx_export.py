import os
import json
import pytest
import torch
from scripts.export_onnx import export_to_onnx

def test_export_onnx_raises_unsupported_for_pure_fft2d(tmp_path):
    """
    Verifies that export_to_onnx correctly raises an exception when attempting
    to export pure 2D FFT (torch.fft.rfft2) models to ONNX, safeguarding pure spectral math integrity.
    """
    ckpt_path = tmp_path / "dummy_calibrated.pth"
    onnx_path = tmp_path / "dummy_model.onnx"

    from src.models.hybrid_detector import HybridDeepfakeDetector
    model = HybridDeepfakeDetector()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_threshold': 0.15,
        'temperature': 1.4788
    }, ckpt_path)

    # Pure 2D FFT models cannot be exported to ONNX without approximation.
    # torch.onnx.export raises RuntimeError for unsupported ops.
    with pytest.raises((RuntimeError, torch.onnx.errors.OnnxExporterError)):
        export_to_onnx(str(ckpt_path), str(onnx_path), quantize=False)
