"""Unit tests for app.py helper and inference functions."""

import numpy as np
import torch

from app import (
    clean_state_dict,
    normalize_confidence,
    preprocess_tensors_batch,
    process_video_frames,
)
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import compute_ece, fit_temperature_log


class TestCleanStateDict:
    def test_strips_module_prefix(self) -> None:
        sd = {"module.spatial_fc.weight": torch.tensor([1.0])}
        cleaned = clean_state_dict(sd)
        assert "spatial_fc.weight" in cleaned
        assert "module.spatial_fc.weight" not in cleaned

    def test_strips_orig_mod_prefix(self) -> None:
        sd = {"_orig_mod.classifier.0.weight": torch.tensor([2.0])}
        cleaned = clean_state_dict(sd)
        assert "classifier.0.weight" in cleaned

    def test_ignores_lora_keys(self) -> None:
        sd = {"lora_adapter.weight": torch.tensor([3.0]), "classifier.0.weight": torch.tensor([1.0])}
        cleaned = clean_state_dict(sd)
        assert "lora_adapter.weight" not in cleaned
        assert "classifier.0.weight" in cleaned


class TestNormalizeConfidence:
    def test_prob_above_threshold(self) -> None:
        conf = normalize_confidence(0.75, 0.5)
        assert abs(conf - 75.0) < 1e-4

    def test_prob_below_threshold(self) -> None:
        conf = normalize_confidence(0.25, 0.5)
        assert abs(conf - 75.0) < 1e-4

    def test_prob_equals_threshold(self) -> None:
        conf = normalize_confidence(0.5, 0.5)
        assert abs(conf - 50.0) < 1e-4

    def test_boundary_extreme(self) -> None:
        assert abs(normalize_confidence(1.0, 0.01) - 100.0) < 1e-2
        assert abs(normalize_confidence(0.0, 0.01) - 100.0) < 1e-2


class TestPreprocessTensorsBatch:
    def test_tensor_shapes_and_types(self) -> None:
        fake_faces = [np.ones((256, 256, 3), dtype=np.uint8) * 128 for _ in range(3)]
        norm_np, norm_torch = preprocess_tensors_batch(fake_faces, device=torch.device("cpu"))

        assert norm_np.shape == (3, 3, 256, 256)
        assert norm_torch.shape == (3, 3, 256, 256)
        assert norm_torch.dtype == torch.float32


class TestCheckpointRoundtrip:
    def test_checkpoint_roundtrip_cleaning_and_loading(self) -> None:
        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        raw_sd = model.state_dict()

        wrapped_sd = {f"module._orig_mod.{k}": v for k, v in raw_sd.items()}
        cleaned_sd = clean_state_dict(wrapped_sd)

        new_model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        incompatible_keys = new_model.load_state_dict(cleaned_sd, strict=False)

        assert len(incompatible_keys.missing_keys) == 0
        assert len(incompatible_keys.unexpected_keys) == 0


class TestProcessVideoFramesEmpty:
    def test_nonexistent_video_path_returns_none(self) -> None:
        res = process_video_frames("non_existent_file_path_12345.mp4")
        assert res is None


class TestTemperatureCalibration:
    def test_temperature_calibration_and_ece(self) -> None:
        logits = np.array([5.0, 4.0, 3.0, -5.0, -4.0, -3.0], dtype=np.float32)
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)

        temp = fit_temperature_log(logits, labels)
        assert temp > 0.1, "Temperature T must be strictly positive"

        calibrated_probs = 1.0 / (1.0 + np.exp(-logits / temp))
        raw_probs = 1.0 / (1.0 + np.exp(-logits))

        eps = 1e-7
        nll_raw = -np.mean(labels * np.log(np.clip(raw_probs, eps, 1 - eps)) + (1 - labels) * np.log(np.clip(1 - raw_probs, eps, 1 - eps)))
        nll_calibrated = -np.mean(labels * np.log(np.clip(calibrated_probs, eps, 1 - eps)) + (1 - labels) * np.log(np.clip(1 - calibrated_probs, eps, 1 - eps)))

        assert nll_calibrated <= nll_raw + 1e-4, "Temperature scaling must minimize or maintain NLL loss"

        ece_val = compute_ece(calibrated_probs, labels)
        assert 0.0 <= ece_val <= 1.0, "ECE must be a valid probability error in [0, 1]"


class TestGradCAMDiagnostics:
    def test_generate_face_diagnostics_keys_and_hook_cleanup(self) -> None:
        from src.utils.interpretability import generate_face_diagnostics

        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        model.eval()
        face_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 128
        device = torch.device("cpu")

        diag = generate_face_diagnostics(model, face_rgb, device=device)

        assert "original" in diag
        assert "srm_residual" in diag
        assert "fft_spectrum" in diag
        assert "gradcam_overlay" in diag
        assert diag["gradcam_overlay"].shape == (256, 256, 3)

        # Check hook cleanup: spatial_backbone[-1] should have no registered forward/backward hooks
        target_layer = model.spatial_backbone[-1]
        assert len(target_layer._forward_hooks) == 0, "Forward hook was not cleaned up after Grad-CAM"
        assert len(target_layer._backward_hooks) == 0, "Backward hook was not cleaned up after Grad-CAM"

    def test_gradcam_batched_input_execution(self) -> None:
        from src.utils.interpretability import ConvNeXtGradCAM

        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        model.eval()
        grad_cam = ConvNeXtGradCAM(model)
        try:
            batch_tensor = torch.rand(2, 3, 256, 256)
            heatmap = grad_cam.generate_heatmap(batch_tensor, img_size=256)
            assert heatmap.shape == (256, 256)
            assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
        finally:
            grad_cam.remove_hooks()


class TestSyntheticVideoInference:
    def test_process_video_frames_with_synthetic_mp4(self, tmp_path) -> None:
        import cv2

        video_path = str(tmp_path / "test_synthetic.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (128, 128))

        # Write 5 synthetic frames (colored boxes simulating faces)
        for i in range(5):
            frame = np.ones((128, 128, 3), dtype=np.uint8) * (50 + i * 30)
            cv2.rectangle(frame, (30, 30), (90, 90), (200, 200, 200), -1)
            out.write(frame)
        out.release()

        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=True)
        model.eval()

        res = process_video_frames(
            video_path=video_path,
            pytorch_model=model,
            num_frames=3,
        )

        # Video engine processes frames safely (returns dict or None if no face detected by YuNet)
        if res is not None:
            assert "raw_video_prob" in res
            assert 0.0 <= res["raw_video_prob"] <= 1.0

