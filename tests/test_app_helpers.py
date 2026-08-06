"""
Unit tests for app.py helper and inference functions.
"""

import pytest
import numpy as np
import torch

from app import (
    clean_state_dict,
    normalize_confidence,
    preprocess_tensors_batch,
    process_video_frames,
)


class TestCleanStateDict:
    def test_strips_module_prefix(self):
        sd = {"module.spatial_fc.weight": torch.tensor([1.0])}
        cleaned = clean_state_dict(sd)
        assert "spatial_fc.weight" in cleaned
        assert "module.spatial_fc.weight" not in cleaned

    def test_strips_orig_mod_prefix(self):
        sd = {"_orig_mod.classifier.0.weight": torch.tensor([2.0])}
        cleaned = clean_state_dict(sd)
        assert "classifier.0.weight" in cleaned

    def test_ignores_lora_keys(self):
        sd = {"lora_adapter.weight": torch.tensor([3.0]), "classifier.0.weight": torch.tensor([1.0])}
        cleaned = clean_state_dict(sd)
        assert "lora_adapter.weight" not in cleaned
        assert "classifier.0.weight" in cleaned


class TestNormalizeConfidence:
    def test_prob_above_threshold(self):
        # prob=0.75 with threshold=0.5 -> 50 + 50 * (0.25 / 0.5) = 75.0%
        conf = normalize_confidence(0.75, 0.5)
        assert abs(conf - 75.0) < 1e-4

    def test_prob_below_threshold(self):
        # prob=0.25 with threshold=0.5 -> 50 + 50 * (0.25 / 0.5) = 75.0%
        conf = normalize_confidence(0.25, 0.5)
        assert abs(conf - 75.0) < 1e-4

    def test_prob_equals_threshold(self):
        conf = normalize_confidence(0.5, 0.5)
        assert abs(conf - 50.0) < 1e-4

    def test_boundary_extreme(self):
        assert abs(normalize_confidence(1.0, 0.01) - 100.0) < 1e-2
        assert abs(normalize_confidence(0.0, 0.01) - 100.0) < 1e-2


class TestPreprocessTensorsBatch:
    def test_tensor_shapes_and_types(self):
        fake_faces = [np.ones((256, 256, 3), dtype=np.uint8) * 128 for _ in range(3)]
        norm_np, norm_torch = preprocess_tensors_batch(fake_faces, device=torch.device("cpu"))

        assert norm_np.shape == (3, 3, 256, 256)
        assert norm_torch.shape == (3, 3, 256, 256)
        assert norm_torch.dtype == torch.float32


class TestProcessVideoFramesEmpty:
    def test_nonexistent_video_path_returns_none(self):
        res = process_video_frames("non_existent_video_file_xyz.mp4")
        assert res is None
