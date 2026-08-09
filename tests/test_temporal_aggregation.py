"""Unit tests for src/utils/temporal_aggregation.py."""

from src.utils.temporal_aggregation import (
    aggregate_video_predictions,
    ema_aggregation,
    mean_aggregation,
    soft_max_weighted_aggregation,
    top_k_aggregation,
)


class TestMeanAggregation:
    def test_basic(self) -> None:
        result = mean_aggregation([0.2, 0.4, 0.6])
        assert abs(result - 0.4) < 1e-5

    def test_single_value(self) -> None:
        assert abs(mean_aggregation([0.7]) - 0.7) < 1e-5


class TestNestedListInput:
    def test_nested_list_flattened(self) -> None:
        flat = mean_aggregation([0.2, 0.4, 0.6])
        nested = mean_aggregation([[0.2], [0.4], [0.6]])
        assert abs(flat - nested) < 1e-5

    def test_nested_list_top_k(self) -> None:
        flat = top_k_aggregation([0.9, 0.8, 0.1], k=2)
        nested = top_k_aggregation([[0.9], [0.8], [0.1]], k=2)
        assert abs(flat - nested) < 1e-5


class TestTopKAggregation:
    def test_basic(self) -> None:
        result = top_k_aggregation([0.9, 0.8, 0.1], k=2)
        assert abs(result - 0.85) < 1e-5

    def test_k_clamped_when_exceeds_length(self) -> None:
        result = top_k_aggregation([0.9, 0.8], k=10)
        assert abs(result - 0.85) < 1e-5

    def test_k_one(self) -> None:
        result = top_k_aggregation([0.9, 0.5, 0.1], k=1)
        assert abs(result - 0.9) < 1e-5


class TestSoftMaxWeightedAggregation:
    def test_emphasises_high_scores(self) -> None:
        scores = [0.01] * 195 + [0.99] * 5
        soft = soft_max_weighted_aggregation(scores, tau=0.1)
        mean = mean_aggregation(scores)
        assert soft > mean + 0.10

    def test_tau_one_approaches_weighted_mean(self) -> None:
        scores = [0.5] * 10
        result = soft_max_weighted_aggregation(scores, tau=1.0)
        assert abs(result - 0.5) < 1e-5

    def test_single_value(self) -> None:
        result = soft_max_weighted_aggregation([0.7], tau=1.0)
        assert abs(result - 0.7) < 1e-5


class TestEMAAggregation:
    def test_out_of_order_frame_indices(self) -> None:
        result = ema_aggregation([0.9, 0.1, 0.5], frame_indices=[2, 0, 1], alpha=0.3)
        assert abs(result - 0.424) < 1e-4

    def test_no_frame_indices_uses_input_order(self) -> None:
        result = ema_aggregation([0.1, 0.5, 0.9], alpha=0.3)
        assert abs(result - 0.424) < 1e-4

    def test_single_frame(self) -> None:
        assert abs(ema_aggregation([0.6]) - 0.6) < 1e-5


class TestNaNInfSafety:
    def test_filters_nan_and_inf(self) -> None:
        result = mean_aggregation([0.5, float("nan"), float("inf"), 0.9])
        assert abs(result - 0.7) < 1e-5

    def test_all_nan_returns_default(self) -> None:
        result = mean_aggregation([float("nan"), float("nan")])
        assert result == 0.5

    def test_top_k_with_nan(self) -> None:
        result = top_k_aggregation([float("nan"), 0.9, 0.8], k=2)
        assert abs(result - 0.85) < 1e-5


class TestEmptyInputSafeDefaults:
    def test_mean_empty_flat(self) -> None:
        assert mean_aggregation([]) == 0.5

    def test_mean_empty_nested(self) -> None:
        assert mean_aggregation([[]]) == 0.5

    def test_top_k_empty(self) -> None:
        assert top_k_aggregation([]) == 0.5

    def test_soft_max_empty(self) -> None:
        assert soft_max_weighted_aggregation([]) == 0.5

    def test_ema_empty(self) -> None:
        assert ema_aggregation([]) == 0.5


class TestAggregateVideoPredicitions:
    def test_dispatcher_soft_max(self) -> None:
        result = aggregate_video_predictions([0.8, 0.9, 0.85], method="soft_max")
        assert "video_score" in result
        assert "is_fake" in result
        assert "valid_frames_count" in result
        assert result["valid_frames_count"] == 3

    def test_dispatcher_unknown_method_defaults_to_soft_max(self) -> None:
        r1 = aggregate_video_predictions([0.5, 0.6], method="soft_max")
        r2 = aggregate_video_predictions([0.5, 0.6], method="invalid_method")
        assert abs(r1["video_score"] - r2["video_score"]) < 1e-5

    def test_is_fake_flag(self) -> None:
        result = aggregate_video_predictions([0.9, 0.95], method="mean", threshold=0.5)
        assert result["is_fake"] is True
        result = aggregate_video_predictions([0.1, 0.05], method="mean", threshold=0.5)
        assert result["is_fake"] is False
