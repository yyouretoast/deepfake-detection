"""Export trained Dual-Stream Deepfake Detector to ONNX format."""

import argparse
import io
import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.services.video_engine import load_prediction_engine


def export_onnx(
    output_path: str = "models/dual_stream_detector.onnx",
    weights_path: str = None,
    img_size: int = 256,
) -> str:
    """Instantiate the trained model and export it to ONNX format."""
    pytorch_model, _, _, _, _ = load_prediction_engine(weights_path=weights_path)
    pytorch_model.eval()

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    device = next(pytorch_model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        abs_output_path,
        export_params=True,
        opset_version=18,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained Dual-Stream Detector to ONNX format.")
    parser.add_argument("--output", default="models/dual_stream_detector.onnx", help="Path to output ONNX file")
    parser.add_argument("--weights", default=None, help="Path to PyTorch model weights")
    parser.add_argument("--img_size", type=int, default=256, help="Input image resolution")
    args = parser.parse_args()

    export_onnx(output_path=args.output, weights_path=args.weights, img_size=args.img_size)


if __name__ == "__main__":
    main()
