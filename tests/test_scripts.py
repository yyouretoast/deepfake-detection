"""Unit tests for executable script helpers in scripts/*.py."""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_robustness import blur_fn, downscale_fn, jpeg_fn, noise_fn
from scripts.train_loto_experiment import filter_loto_split_strict, matches_holdout_domain


class TestLOTOScriptHelpers:
    def test_matches_holdout_domain_celeb(self) -> None:
        assert matches_holdout_domain("fake/celeb_df_v2/id0_id16/001.png", "celeb") is True
        assert matches_holdout_domain("fake/ff_c23/000_003/001.png", "celeb") is False

    def test_matches_holdout_domain_generators(self) -> None:
        assert matches_holdout_domain("fake/050_053/001.png", "deepfakes") is True
        assert matches_holdout_domain("fake/150_153/001.png", "face2face") is True
        assert matches_holdout_domain("fake/250_253/001.png", "faceswap") is True
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

        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
        onnx_out = str(tmp_path / "test_model.onnx")

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_out,
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
        except (ModuleNotFoundError, ImportError) as e:
            pytest.skip(f"ONNX exporter dependency missing: {e}")

        assert os.path.exists(onnx_out), "Exported ONNX file does not exist"
        assert os.path.getsize(onnx_out) > 1000, "Exported ONNX file is empty"
