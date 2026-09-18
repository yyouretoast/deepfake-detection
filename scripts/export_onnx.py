"""Export trained Dual-Stream Deepfake Detector to ONNX format."""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.services.video_engine import load_prediction_engine


def fuse_conv_bn_eval_recursive(module: torch.nn.Module) -> None:
    """Recursively fuse Conv2d and BatchNorm2d pairs in eval mode to avoid Dynamo decomposition tuples."""
    children = list(module.named_children())
    for i in range(len(children) - 1):
        name1, child1 = children[i]
        name2, child2 = children[i + 1]
        if isinstance(child1, torch.nn.Conv2d) and isinstance(child2, torch.nn.BatchNorm2d):
            fused = torch.nn.utils.fuse_conv_bn_eval(child1, child2)
            setattr(module, name1, fused)
            setattr(module, name2, torch.nn.Identity())
    for _, child in module.named_children():
        fuse_conv_bn_eval_recursive(child)


def export_onnx(
    output_path: str = "models/dual_stream_detector.onnx",
    weights_path: str | None = None,
    img_size: int = 256,
) -> str:
    """Instantiate the trained model and export it to ONNX format."""
    pytorch_model, _, _, _, _ = load_prediction_engine(weights_path=weights_path)
    pytorch_model.eval()

    # Pre-pass: Fuse Conv2d and BatchNorm2d layers in eval mode
    fuse_conv_bn_eval_recursive(pytorch_model)

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    device = next(pytorch_model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref_output = pytorch_model(dummy_input).detach().cpu().numpy()

    torch.onnx.export(
        pytorch_model,
        (dummy_input,),
        abs_output_path,
        opset_version=18,
        dynamo=True,
    )

    # Consolidate external data into a single self-contained ONNX file if needed
    try:
        import onnx

        data_file = abs_output_path + ".data"
        if os.path.exists(data_file):
            model_proto = onnx.load(abs_output_path, load_external_data=True)
            onnx.save_model(model_proto, abs_output_path, save_as_external_data=False)
            if os.path.exists(data_file):
                os.remove(data_file)
    except Exception as e:
        print(f"Note: ONNX consolidation skipped ({e})")

    print(f"ONNX model successfully exported to {abs_output_path}")

    # Parity verification with onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(abs_output_path, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        ort_output = sess.run(None, {in_name: dummy_input.detach().cpu().numpy()})[0]
        max_diff = float(np.max(np.abs(ref_output - ort_output)))
        print(f"ONNX Runtime parity verification: max absolute error = {max_diff:.6e}")
        np.testing.assert_allclose(ref_output, ort_output, atol=1e-3, rtol=1e-3)
        print(" Parity check passed within tolerance (atol=1e-3).")
    except ImportError as e:
        print(f"Skipping ONNX Runtime parity check: optional dependency not installed ({e}).")

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
