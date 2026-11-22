"""Unit tests for executable script helpers in scripts/*.py."""

import json
import os
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_loto_experiment import filter_loto_split_strict, matches_holdout_domain
from src.dataset.degradations import blur_fn, downscale_fn, jpeg_fn, noise_fn


class TestLOTOScriptHelpers:
    def test_matches_holdout_domain_celeb(self) -> None:
        assert matches_holdout_domain("fake/celeb_df_v2/id0_id16/001.png", "celeb") is True
        assert matches_holdout_domain("fake/ff_c23/000_003/001.png", "celeb") is False

    def test_matches_holdout_domain_generators(self) -> None:
        assert matches_holdout_domain("fake/050_053/001.png", "deepfakes") is True
        assert matches_holdout_domain("fake/150_153/001.png", "face2face") is True
        assert matches_holdout_domain("fake/250_253/001.png", "face2face") is True
        assert matches_holdout_domain("fake/450_453/001.png", "faceswap") is True
        assert matches_holdout_domain("fake/650_653/001.png", "neuraltextures") is True

    def test_filter_loto_split_strict(self) -> None:
        samples = [
            ("fake/celeb_df_v2/id0_id16/001.png", 1.0),
            ("real/celeb_df_v2/id0_id16/002.png", 0.0),
            ("fake/ff_c23/000_003/003.png", 1.0),
        ]
        retained, held_out = filter_loto_split_strict(samples, "celeb")
        assert len(held_out) == 1
        assert held_out[0][0] == "fake/celeb_df_v2/id0_id16/001.png"
        assert len(retained) == 2


class TestRobustnessDegradations:
    @pytest.fixture
    def sample_rgb_image(self) -> np.ndarray:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 256, dtype=np.uint8)
        img[:, :, 1] = 128
        img[:, :, 2] = 200
        return img

    def test_jpeg_fn(self, sample_rgb_image: np.ndarray) -> None:
        fn = jpeg_fn(quality=50)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8

    def test_blur_fn(self, sample_rgb_image: np.ndarray) -> None:
        fn = blur_fn(sigma=2.0)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8

    def test_noise_fn(self, sample_rgb_image: np.ndarray) -> None:
        fn = noise_fn(sigma=15.0)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8
        assert np.min(degraded) >= 0
        assert np.max(degraded) <= 255

    def test_downscale_fn(self, sample_rgb_image: np.ndarray) -> None:
        fn = downscale_fn(scale=0.5)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8


class TestExportONNX:
    @pytest.mark.slow
    def test_onnx_export_and_file_creation(self, tmp_path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx optional dependency is not installed")

        try:
            import onnxscript  # noqa: F401
        except ImportError:
            pytest.skip("onnxscript optional dependency is not installed")

        import torch

        from src.models.hybrid_detector import HybridDeepfakeDetector

        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        model.eval()

        dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
        onnx_out = str(tmp_path / "test_model.onnx")

        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_out,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                dynamo=True,
                input_names=["input_rgb"],
                output_names=["logits"],
                dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
            )
        except (ModuleNotFoundError, ImportError) as e:
            pytest.skip(f"ONNX export dependency missing in this environment: {e}")

        assert os.path.exists(onnx_out), "Exported ONNX file does not exist"
        assert os.path.getsize(onnx_out) > 1000, "Exported ONNX file is empty"


class TestPredictCLI:
    """Verifies standalone predict.py CLI argument parsing and execution."""

    def test_predict_cli_positional_and_json(self, tmp_path, monkeypatch, capsys):
        import torch

        import predict

        img_path = str(tmp_path / "dummy_face.png")
        cv2.imwrite(img_path, np.full((256, 256, 3), 128, dtype=np.uint8))

        dummy_weights = str(tmp_path / "dummy_weights.pth")
        torch.save({"model_state_dict": {}}, dummy_weights)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "predict.py",
                img_path,
                "--weights",
                dummy_weights,
                "--is_crop",
                "--device",
                "cpu",
                "--json",
            ],
        )
        predict.main()
        stdout_clean = capsys.readouterr().out.strip()
        json_start = stdout_clean.find("{")
        assert json_start != -1, f"No JSON object found in output: {stdout_clean}"
        data = json.loads(stdout_clean[json_start:])
        assert "calibrated_posterior_probability" in data
        assert "triage" in data
        assert data["triage"]["binary_decision"] in ["AUTHENTIC", "SYNTHETIC"]

    def test_predict_numerical_parity_with_model(self, tmp_path):
        import torch

        import predict
        from src.models.hybrid_detector import HybridDeepfakeDetector

        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        model.eval()

        weights_path = str(tmp_path / "parity_weights.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "calibration": {
                    "platt_scale_a": 0.278311,
                    "platt_bias_b": 0.408914,
                    "effective_temperature": 3.5931,
                    "optimal_threshold": 0.2600,
                },
            },
            weights_path,
        )

        img_path = str(tmp_path / "face.png")
        crop_np = np.full((256, 256, 3), 128, dtype=np.uint8)
        cv2.imwrite(img_path, crop_np)

        device = torch.device("cpu")
        detector, cal = predict.load_detector(weights_path, device=device)
        cropper = predict.DynamicFaceCropper(target_size=256)
        res = predict.predict_image(img_path, detector, cropper, cal, device=device, is_crop=True)

        crop_tensor = torch.from_numpy(crop_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            gt_logit = float(detector(crop_tensor).item())

        a = cal.get("platt_scale_a", 1.0)
        b = cal.get("platt_bias_b", 0.0)
        expected_prob = float(1.0 / (1.0 + np.exp(-(a * gt_logit + b))))

        assert abs(res["raw_logit"] - gt_logit) < 1e-4
        assert abs(res["calibrated_posterior_probability"] - expected_prob) < 1e-4

