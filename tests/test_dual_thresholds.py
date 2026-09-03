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
