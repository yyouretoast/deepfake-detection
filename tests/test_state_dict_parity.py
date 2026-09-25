"""Unit test asserting numerical logit parity between modular model and trained checkpoint."""

import pytest
import torch

from src.dataset.resolver import find_weights_path


class TestStateDictParity:
    """Verifies that refactored modular HybridDeepfakeDetector loads weights and replicates logits exactly."""

    def test_checkpoint_numerical_parity(self) -> None:
        try:
            ckpt_path = find_weights_path("models/dual_stream_calibrated.pth")
        except FileNotFoundError:
            pytest.skip("Calibrated model weights not found.")

        from src.utils.checkpoint import load_detector_checkpoint

        model, _, _ = load_detector_checkpoint(ckpt_path, device="cpu")
        model.eval()

        g = torch.Generator().manual_seed(42)
        x = torch.randn(2, 3, 256, 256, generator=g)
        with torch.no_grad():
            out = model(x)

        logits = out.squeeze().tolist()
        expected = [8.537198066711426, 8.25665283203125]

        assert len(logits) == len(expected)
        for act, exp in zip(logits, expected):
            assert abs(act - exp) < 1e-4, f"Logit drift: actual={act}, expected={exp}"
