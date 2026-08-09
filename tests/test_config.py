"""Unit tests for configuration loading and dynamic overrides."""

from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector


def test_load_config_values_and_model_override() -> None:
    cfg = load_config("config/default.yaml")
    assert isinstance(cfg, dict)
    assert cfg["preprocessing"]["img_size"] == 512
    assert cfg["preprocessing"]["scale_factor"] == 1.50
    assert cfg["model"]["backbone"] == "convnext_small"

    cfg["model"]["use_fft_branch"] = False
    model = HybridDeepfakeDetector(pretrained=False, config=cfg)
    assert model.use_fft_branch is False
