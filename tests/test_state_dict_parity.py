"""Unit test asserting numerical logit parity between modular model and trained checkpoint."""

import os
import pytest
import torch
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict


class TestStateDictParity:
    """Verifies that refactored modular HybridDeepfakeDetector loads weights and replicates logits exactly."""

    def test_checkpoint_numerical_parity(self) -> None:
        ckpt_path = "dual_stream_calibrated.pth"
        if not os.path.exists(ckpt_path):
            pytest.skip(f"Checkpoint not found at {ckpt_path}")

        torch.manual_seed(42)
        model = HybridDeepfakeDetector(pretrained=False)
        model.eval()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)

        logits = out.squeeze().tolist()
        expected = [-16.286630630493164, -16.344982147216797]

        assert len(logits) == len(expected)
        for act, exp in zip(logits, expected):
            assert abs(act - exp) < 1e-5, f"Logit drift: actual={act}, expected={exp}"
