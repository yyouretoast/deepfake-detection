import os
import numpy as np
import torch

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

def export_to_onnx(model: torch.nn.Module, save_path: str = "deepfake_convnext_v2.onnx", img_size: int = 224) -> str:
    """
    Exports PyTorch HybridDeepfakeDetector model to ONNX runtime format
    with dynamic batch size support.
    """
    model.eval()
    if hasattr(model, 'module'):
        model = model.module

    dummy_input = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)

    # Dynamic axes allow inference on arbitrary batch sizes [B, 3, 224, 224]
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    if HAS_ONNX:
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model successfully exported and verified: {save_path}")

    return save_path

class ONNXDeepfakePredictor:
    """
    High-performance ONNX Runtime inference engine.
    Supports 3x-5x faster CPU/GPU predictions.
    """
    def __init__(self, onnx_path: str):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required for ONNXDeepfakePredictor. Run pip install onnxruntime.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_batch(self, numpy_batch: np.ndarray) -> np.ndarray:
        """
        Runs ONNX inference on a numpy batch [B, 3, 224, 224].
        Returns array of probabilities [B] in range [0, 1].
        """
        if numpy_batch.ndim == 3:
            numpy_batch = np.expand_dims(numpy_batch, axis=0)

        numpy_batch = numpy_batch.astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: numpy_batch})[0]
        
        # Clip logits to eliminate overflow in NumPy sigmoid exponentiation
        clipped_logits = np.clip(logits, -88.0, 88.0)
        probs = 1.0 / (1.0 + np.exp(-clipped_logits))
        return probs.squeeze(-1) if probs.ndim > 1 else probs
