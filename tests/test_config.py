import os
import pytest
from src.config import load_config, DEFAULT_CONFIG
from src.models.hybrid_detector import HybridDeepfakeDetector

def test_load_config_default():
    cfg = load_config("config/default.yaml")
    assert isinstance(cfg, dict)
    assert "paths" in cfg
    assert "preprocessing" in cfg
    assert "model" in cfg
    assert "training" in cfg

def test_config_key_values():
    cfg = load_config("config/default.yaml")
    assert cfg["preprocessing"]["img_size"] == 224
    assert cfg["preprocessing"]["padding_scale"] == 1.30
    assert cfg["model"]["backbone"] == "convnext_small"
    assert cfg["model"]["use_fft_branch"] is True

def test_model_instantiation_from_config():
    cfg = load_config("config/default.yaml")
    model = HybridDeepfakeDetector(pretrained=False, config=cfg)
    assert model is not None
    assert model.use_fft_branch is True
