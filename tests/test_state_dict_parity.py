"""Unit test asserting numerical logit parity between modular model and trained checkpoint."""

import pytest
import torch

from src.dataset.resolver import find_weights_path
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict


class TestStateDictParity:
    """Verifies that refactored modular HybridDeepfakeDetector loads weights and replicates logits exactly."""

    def test_checkpoint_numerical_parity(self) -> None:
        try:
            ckpt_path = find_weights_path("models/dual_stream_calibrated.pth")
        except FileNotFoundError:
            pytest.skip("Calibrated model weights not found.")

        model = HybridDeepfakeDetector(pretrained=False)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(clean_state_dict(state_dict), strict=False)
        model.eval()

        g = torch.Generator().manual_seed(42)
        x = torch.randn(2, 3, 256, 256, generator=g)
        with torch.no_grad():
            out = model(x)

        logits = out.squeeze().tolist()
        expected = [0.347933292388916, 0.26476922631263733]

        assert len(logits) == len(expected)
        for act, exp in zip(logits, expected):
            assert abs(act - exp) < 1e-4, f"Logit drift: actual={act}, expected={exp}"
