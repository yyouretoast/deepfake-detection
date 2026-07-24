from typing import Dict, Any, Optional
import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "kaggle_input": "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23",
        "output_dir": "/kaggle/working/frames",
        "processed_dir": "/kaggle/working/processed",
        "model_save_path": "/kaggle/working/deepfake_convnext_v2.pth"
    },
    "preprocessing": {
        "img_size": 224,
        "padding_scale": 1.30,
        "max_videos_per_type": 200,
        "frames_per_video": 15,
        "min_face_size": 20,
        "blur_threshold_real": 50,
        "blur_threshold_fake": 20
    },
    "model": {
        "backbone": "convnext_base",
        "pretrained": True,
        "use_fft_branch": True,
        "spatial_embed_dim": 1024,
        "freq_embed_dim": 128,
        "freq_scale": 0.1,
        "dropout": 0.3
    },
    "training": {
        "batch_size": 64,
        "num_workers": 4,
        "epochs_phase1": 3,
        "epochs_phase2": 5,
        "lr_phase1": 1.0e-3,
        "lr_backbone": 1.0e-5,
        "lr_head": 1.0e-4,
        "weight_decay": 1.0e-4,
        "use_amp": True,
        "seed": 42
    },
    "labels": {
        "real": 0,
        "fake": 1
    },
    "manipulation_types": {
        "all": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"],
        "held_out_loto": "FaceShifter"
    }
}

def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """
    Loads centralized YAML configuration dictionary.
    Falls back to DEFAULT_CONFIG if configuration file is missing or invalid.
    """
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            return cfg
    except Exception as e:
        logger.warning("Failed to parse config file %s: %s. Using DEFAULT_CONFIG.", config_path, e)

    return DEFAULT_CONFIG
