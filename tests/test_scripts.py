"""
Unit tests for executable script helpers in scripts/*.py
(train_loto_experiment.py, evaluate_robustness.py, export_onnx.py).
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import cv2
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_loto_experiment import matches_holdout_domain, filter_loto_split_strict
from scripts.evaluate_robustness import jpeg_fn, blur_fn, noise_fn, downscale_fn


class TestLOTOScriptHelpers:
    def test_matches_holdout_domain_celeb(self):
        assert matches_holdout_domain("fake/celeb_df_v2/id0_id16/001.png", "celeb") is True
        assert matches_holdout_domain("fake/ff_c23/000_003/001.png", "celeb") is False

    def test_matches_holdout_domain_generators(self):
        assert matches_holdout_domain("fake/050_053/001.png", "deepfakes") is True
        assert matches_holdout_domain("fake/250_253/001.png", "face2face") is True
        assert matches_holdout_domain("fake/450_453/001.png", "faceswap") is True
        assert matches_holdout_domain("fake/650_653/001.png", "neuraltextures") is True

    def test_filter_loto_split_strict(self):
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
    def sample_rgb_image(self):
        # Create 256x256 test RGB image with gradient pattern
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 256, dtype=np.uint8)
        img[:, :, 1] = 128
        img[:, :, 2] = 200
        return img

    def test_jpeg_fn(self, sample_rgb_image):
        fn = jpeg_fn(quality=50)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8

    def test_blur_fn(self, sample_rgb_image):
        fn = blur_fn(sigma=2.0)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8

    def test_noise_fn(self, sample_rgb_image):
        fn = noise_fn(sigma=15.0)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8
        assert np.min(degraded) >= 0
        assert np.max(degraded) <= 255

    def test_downscale_fn(self, sample_rgb_image):
        fn = downscale_fn(scale=0.5)
        degraded = fn(sample_rgb_image)
        assert degraded.shape == sample_rgb_image.shape
        assert degraded.dtype == np.uint8
