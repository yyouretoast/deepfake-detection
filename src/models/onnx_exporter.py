from typing import List, Optional
import os
import logging
import numpy as np
import torch

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)

def export_to_onnx(
    model: torch.nn.Module,
    save_path: str = "deepfake_convnext_v2.onnx",
    img_size: int = 256,
    dummy_input: Optional[torch.Tensor] = None
) -> str:
    """Exports PyTorch HybridDeepfakeDetector model to ONNX format with dynamic batching and dynamic sequence length support."""
    model.eval()
    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model

    device = next(unwrapped.parameters()).device
    if dummy_input is None:
        dummy_input = torch.randn(2, 3, img_size, img_size, device=device)

    if dummy_input.ndim == 5:
        dynamic_axes = {
            'input': {0: 'batch_size', 1: 'sequence_length', 3: 'height', 4: 'width'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }

    torch.onnx.export(
        unwrapped,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    if HAS_ONNX:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

    return save_path

def quantize_onnx_model(
    onnx_path: str = "deepfake_convnext_v2.onnx",
    quant_path: Optional[str] = None
) -> str:
    """
    Applies INT8 Dynamic Quantization to an ONNX model, reducing memory footprint
    and enhancing CPU inference throughput via QInt8 signed integer quantization.
    """
    if not HAS_ONNX:
        raise ImportError("onnxruntime is required for quantization.")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if quant_path is None:
        base, ext = os.path.splitext(onnx_path)
        quant_path = f"{base}_quant{ext}"

    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8
    )
    logger.info("Saved quantized INT8 model to: %s", quant_path)
    return quant_path

class ONNXDeepfakePredictor:
    """
    High-performance ONNX Runtime inference engine.
    Supports 3x-5x faster CPU/GPU predictions.
    """
    def __init__(self, onnx_path: str) -> None:
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required for ONNXDeepfakePredictor.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

        available_providers: List[str] = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

    def predict_batch(self, numpy_batch: np.ndarray) -> np.ndarray:
        if numpy_batch.ndim == 3:
            numpy_batch = np.expand_dims(numpy_batch, axis=0)

        numpy_batch = np.asarray(numpy_batch, dtype=np.float32)
        logits = self.session.run([self.output_name], {self.input_name: numpy_batch})[0]
        
        clipped_logits = np.clip(logits, -88.0, 88.0)
        probs = 1.0 / (1.0 + np.exp(-clipped_logits))
        return probs.squeeze(-1) if probs.ndim > 1 else probs
