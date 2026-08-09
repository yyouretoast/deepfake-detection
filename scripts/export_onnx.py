"""Export trained Dual-Stream Deepfake Detector to ONNX format."""

import io
import os
import sys
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.services.video_engine import load_prediction_engine  # noqa: E402


def export_onnx(output_path: str = "models/dual_stream_detector.onnx") -> str:
    """Instantiate the trained model and export it to ONNX format.

    Args:
        output_path: Path where the output ONNX binary file will be saved.

    Returns:
        The output file path.
    """
    pytorch_model, _, _, _, _ = load_prediction_engine()
    pytorch_model.eval()

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    device = next(pytorch_model.parameters()).device
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32, device=device)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        abs_output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_rgb"],
        output_names=["logits"],
        dynamic_axes={
            "input_rgb": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model successfully exported to {abs_output_path}")
    return abs_output_path


if __name__ == "__main__":
    export_onnx()


