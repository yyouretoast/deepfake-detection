"""Unit tests for dual-threshold Bayesian decision boundaries and three-zone classification."""

import numpy as np
from src.utils.checkpoint import classify_three_zone, compute_dual_thresholds


class TestDualThresholds:
    """Verifies dual threshold calculations and three-zone forensic classification."""

    def test_compute_dual_thresholds_ordering(self) -> None:
        # Bimodal synthetic distribution: reals clustered near 0.1, fakes near 0.9
        np.random.seed(42)
        reals = np.random.beta(1, 8, 500)
        fakes = np.random.beta(8, 1, 500)
        probs = np.concatenate([reals, fakes])
        targets = np.array([0] * 500 + [1] * 500)

        tau_real, tau_fake = compute_dual_thresholds(probs, targets, min_precision=0.95)
        assert tau_real < tau_fake
        assert 0.0 < tau_real < 0.6
        assert 0.4 < tau_fake < 1.0

    def test_classify_three_zone_verdicts(self) -> None:
        tau_real, tau_fake = 0.30, 0.70

        res_real = classify_three_zone(0.15, tau_real=tau_real, tau_fake=tau_fake)
        assert res_real["zone"] == "high_confidence_real"
        assert res_real["verdict"] == "Confirmed Authentic"
        assert not res_real["is_inconclusive"]

        res_fake = classify_three_zone(0.85, tau_real=tau_real, tau_fake=tau_fake)
        assert res_fake["zone"] == "high_confidence_fake"
        assert res_fake["verdict"] == "Confirmed Synthetic"
        assert not res_fake["is_inconclusive"]

        res_ambig = classify_three_zone(0.50, tau_real=tau_real, tau_fake=tau_fake)
        assert res_ambig["zone"] == "ambiguity_zone"
        assert res_ambig["is_inconclusive"]

    def test_compute_dual_thresholds_min_samples_floor(self) -> None:
        probs = np.array([0.1] * 50 + [0.99])
        targets = np.array([0] * 50 + [1])
        tau_real, tau_fake = compute_dual_thresholds(probs, targets, min_precision=0.98, min_samples=5)
        assert tau_fake in (0.5, 0.60)

    def test_prediction_engine_backward_compatibility(self) -> None:
        from src.services.video_engine import PredictionEngine

        engine = PredictionEngine("model", "cropper", True, 0.45, 1.2, tau_real=0.35, tau_fake=0.75)
        m, c, w, t, temp = engine
        assert m == "model"
        assert c == "cropper"
        assert w is True
        assert t == 0.45
        assert temp == 1.2
        assert len(engine) == 5

        assert engine.tau_real == 0.35
        assert engine.tau_fake == 0.75
        assert engine.threshold == 0.45
        assert engine.temperature == 1.2
