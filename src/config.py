from typing import Dict, Any, Optional
import os
import yaml
import copy
import logging

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "raw_dir": "data/raw",
        "cropped_dir": "data/cropped",
        "output_dir": "models",
    },
    "preprocessing": {
        "img_size": 256,
        "face_scale_factor": 1.30,
        "frames_per_video": 30,
    },
    "model": {
        "backbone": "convnext_base",
        "pretrained": True,
        "use_fft_branch": True,
        "use_lora": False,
        "lora_rank": 8,
        "freq_embed_dim": 128,
        "dropout": 0.3,
    },
    "training": {
        "seed": 42,
        "batch_size": 32,
        "epochs_phase1": 3,
        "epochs_phase2": 15,
        "lr_phase1": 1e-3,
        "lr_backbone": 1e-5,
        "lr_head": 1e-4,
        "weight_decay": 1e-2,
        "patience": 4,
        "use_amp": True,
        "seq_len": 8,
        "num_workers": 4,
    },
    "explainability": {
        "gradcam_layer": "spatial_backbone.stages.3",
        "target_class": 1,
    }
}

def _deep_merge_dict(base: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges custom dictionary into base dictionary."""
    merged = copy.deepcopy(base)
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration from YAML file and merges missing keys from DEFAULT_CONFIG."""
    if config_path is None or not os.path.exists(config_path):
        config_path = "config/default.yaml"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
            if isinstance(user_cfg, dict):
                return _deep_merge_dict(DEFAULT_CONFIG, user_cfg)
        except Exception as e:
            logging.warning("Failed to parse config file '%s': %s. Using DEFAULT_CONFIG.", config_path, e)

    return copy.deepcopy(DEFAULT_CONFIG)
