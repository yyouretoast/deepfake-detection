"""Configuration loader and default settings for Deepfake Detector."""

from typing import Dict, Any, Optional
import copy
import logging
import os
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "raw_dir": "data/raw",
        "cropped_dir": "data/cropped",
        "output_dir": "models",
    },
    "preprocessing": {
        "img_size": 512,
        "scale_factor": 1.50,
        "frames_per_video": 30,
    },
    "model": {
        "backbone": "convnext_small",
        "pretrained": True,
        "use_fft_branch": True,
        "freq_embed_dim": 512,
        "dropout": 0.3,
    },
    "training": {
        "seed": 42,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "epochs_phase1": 3,
        "epochs_phase2": 15,
        "lr_phase1": 1e-3,
        "lr_backbone": 1e-5,
        "lr_head": 1e-4,
        "weight_decay": 1e-2,
        "pos_weight": 1.0,
        "patience": 4,
        "use_amp": True,
        "seq_len": 16,
        "num_workers": 4,
    },
    "explainability": {
        "gradcam_layer": "spatial_backbone.stages.3",
        "target_class": 1,
    },
}


def _deep_merge_dict(base: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested dictionaries, overriding base with custom values."""
    merged = copy.deepcopy(base)
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file and merge missing key defaults from DEFAULT_CONFIG."""
    if config_path is None or not os.path.exists(config_path):
        config_path = "config/default.yaml"
        if not os.path.exists(config_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_path = os.path.join(repo_root, "config", "default.yaml")
            if os.path.exists(alt_path):
                config_path = alt_path

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
            if isinstance(user_cfg, dict):
                return _deep_merge_dict(DEFAULT_CONFIG, user_cfg)
        except Exception as e:
            logging.warning("Failed to parse config file '%s': %s. Using DEFAULT_CONFIG.", config_path, e)

    return copy.deepcopy(DEFAULT_CONFIG)

