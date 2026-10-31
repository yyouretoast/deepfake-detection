"""Unit tests for canonical evaluation metrics, optimal threshold finding, and checkpoint loading."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import find_optimal_threshold
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import load_detector_checkpoint


class TestOptimalThresholdFinder:
    """Tests find_optimal_threshold across balanced_accuracy, macro_f1, f1, and youden_roc."""

    def test_perfect_separation_balanced_accuracy(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.9, 0.9])
        tau, score = find_optimal_threshold(y_true, y_prob, criterion="balanced_accuracy")
        assert 0.3 <= tau <= 0.7
        assert score == 1.0

    def test_perfect_separation_macro_f1(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        tau, score = find_optimal_threshold(y_true, y_prob, criterion="macro_f1")
        assert 0.3 <= tau <= 0.7
        assert score == 1.0

    def test_perfect_separation_youden_roc(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        tau, j_stat = find_optimal_threshold(y_true, y_prob, criterion="youden_roc")
        assert 0.3 <= tau <= 0.7
        assert j_stat == 1.0

    def test_single_class_edge_case(self) -> None:
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        tau, _score = find_optimal_threshold(y_true, y_prob, criterion="balanced_accuracy")
        assert tau == 0.5

    def test_invalid_criterion_raises(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.2, 0.8])
        with pytest.raises(ValueError, match="Unknown threshold optimization criterion"):
            find_optimal_threshold(y_true, y_prob, criterion="invalid_criterion")


class TestLoadDetectorCheckpoint:
    """Tests load_detector_checkpoint auto-resolution, frequency backbone detection, and mode switching."""

    def test_loads_saved_checkpoint_state(self, tmp_path) -> None:
        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        ckpt_path = str(tmp_path / "model.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimal_threshold": 0.42,
                "temperature": 1.25,
            },
            ckpt_path,
        )

        loaded_model, temp, thresh = load_detector_checkpoint(
            weights_path=ckpt_path, device=torch.device("cpu")
        )
        assert abs(thresh - 0.42) < 1e-4
        assert abs(temp - 1.25) < 1e-4
        assert not loaded_model.training
        assert loaded_model.frequency_backbone == "resse"

    def test_loads_saved_checkpoint_state_nested_calibration(self, tmp_path) -> None:
        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        ckpt_path = str(tmp_path / "model_nested.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "calibration": {
                    "effective_temperature": 3.5931,
                    "optimal_threshold": 0.2600,
                    "platt_scale_a": 0.2783,
                    "platt_bias_b": 0.4089,
                },
            },
            ckpt_path,
        )

        loaded_model, temp, thresh = load_detector_checkpoint(
            weights_path=ckpt_path, device=torch.device("cpu")
        )
        assert abs(thresh - 0.2600) < 1e-4
        assert abs(temp - 3.5931) < 1e-4
        assert not loaded_model.training


class TestCalibrationAndCertification:
    """Tests calibration fitting, ECE, EER, and dual threshold certification."""

    def test_fit_temperature_log(self) -> None:
        from src.evaluation.metrics import fit_temperature_log

        logits = np.array([-10.0, -8.0, -5.0, 5.0, 8.0, 10.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        t = fit_temperature_log(logits, labels)
        assert t > 0.1

    def test_fit_platt_scaling(self) -> None:
        from src.evaluation.metrics import fit_platt_scaling

        logits = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        labels = np.array([0, 0, 0, 1, 1])
        a, b, t_eff = fit_platt_scaling(logits, labels)
        assert a > 0.0
        assert np.isfinite(b)
        assert t_eff > 0.0

    def test_compute_ece(self) -> None:
        from src.evaluation.metrics import compute_ece

        probs = np.array([0.9, 0.9, 0.1, 0.1])
        targets = np.array([1, 1, 0, 0])
        ece = compute_ece(probs, targets, n_bins=5)
        assert 0.0 <= ece < 0.2

    def test_compute_eer(self) -> None:
        from src.evaluation.metrics import compute_eer

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        eer, thresh = compute_eer(y_true, y_score)
        assert eer == 0.0
        assert 0.4 <= thresh <= 0.6

    def test_compute_dual_thresholds_certified(self) -> None:
        from src.utils.checkpoint import compute_dual_thresholds_certified

        probs = np.concatenate([np.linspace(0.01, 0.30, 20), np.linspace(0.70, 0.99, 20)])
        targets = np.concatenate([np.zeros(20), np.ones(20)])
        t_real, t_fake, p_real, p_fake = compute_dual_thresholds_certified(
            probs, targets, min_precision=0.95
        )
        assert t_real < t_fake
        assert p_real >= 0.95
        assert p_fake >= 0.95


class TestCalibrateProbabilitiesBalanced:
    """Tests calibrate_probabilities_balanced for temperature scaling and prior-shift correction."""

    def test_pure_temperature_scaling(self) -> None:
        from src.evaluation.metrics import calibrate_probabilities_balanced

        logits = np.array([-2.0, 0.0, 2.0])
        probs = calibrate_probabilities_balanced(logits, temp=2.0)
        assert abs(probs[1] - 0.5) < 1e-5
        assert probs[0] < 0.5
        assert probs[2] > 0.5

    def test_prior_shift_correction_eliminates_bias(self) -> None:
        from src.evaluation.metrics import calibrate_probabilities_balanced

        logits = np.array([0.0])
        # Suppose a heavily skewed calibration split (90% positive) introduced +2.197 intercept
        cal_prev = 0.90
        b_shifted = np.log(cal_prev / (1.0 - cal_prev))
        # Without correction, probability would be 0.90
        # With Saerens correction, balanced probability at logit=0.0 must be exactly 0.50
        probs_balanced = calibrate_probabilities_balanced(
            logits, platt_a=1.0, platt_b=b_shifted, cal_prevalence=cal_prev, target_prevalence=0.50
        )
        assert abs(probs_balanced[0] - 0.50) < 1e-4

