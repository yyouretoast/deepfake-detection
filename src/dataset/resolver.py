"""Canonical dataset root, split manifest, and model weights resolution."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class DatasetResolver:
    """Resolves dataset root directories, split manifests, and checkpoint paths."""

    DEFAULT_DATASET_CANDIDATES = (
        "./data/cropped",
        "./data",
        "/kaggle/working/local_crops",
        "/kaggle/input/deepfake-face-crops-256/deepfake_crops_512",
        "/kaggle/input/datasets/yassinyasserr/deepfake-face-crops-256/deepfake_crops_512",
        "/kaggle/input/deepfake-face-crops-256",
        "/kaggle/input/datasets/yassinyasserr/deepfake-dataset/deepfake_crops_512",
        "/kaggle/input/deepfake-dataset/deepfake_crops_512",
        "/kaggle/input/datasets/yassinyasserr/deepfake-crops-512/deepfake_crops_512",
        "/kaggle/input/deepfake-crops-512/deepfake_crops_512",
        "/kaggle/input/deepfake_crops_512",
    )

    @classmethod
    def is_valid_dataset_root(cls, path: str) -> bool:
        """A valid dataset root must contain splits.json and at least one image folder ('fake' or 'real')."""
        if not path or not os.path.isdir(path):
            return False
        has_splits = os.path.exists(os.path.join(path, "splits.json"))
        has_images = os.path.exists(os.path.join(path, "fake")) or os.path.exists(os.path.join(path, "real"))
        return has_splits and has_images

    @classmethod
    def find_dataset_root(cls, custom_dir: Optional[str] = None) -> str:
        """
        Locates the dataset directory containing splits.json and image crops.
        Priority: custom argument -> DATASET_ROOT env var -> candidate paths -> /kaggle/input walk.
        """
        candidates: list[str] = []
        if custom_dir:
            candidates.append(custom_dir)
        env_dir = os.getenv("DATASET_ROOT")
        if env_dir:
            candidates.append(env_dir)
        candidates.extend(cls.DEFAULT_DATASET_CANDIDATES)

        for p in candidates:
            if cls.is_valid_dataset_root(p):
                return os.path.abspath(p)

        for p in candidates:
            if p and os.path.exists(os.path.join(p, "splits.json")):
                return os.path.abspath(p)

        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "splits.json" in files and ("fake" in dirs or "real" in dirs):
                    return os.path.abspath(root)
            for root, dirs, files in os.walk("/kaggle/input"):
                if "splits.json" in files:
                    return os.path.abspath(root)

        raise FileNotFoundError(
            "Could not locate dataset containing splits.json. "
            "Specify via --data_dir or DATASET_ROOT environment variable."
        )

    @classmethod
    def resolve_splits_path(
        cls, data_root: Optional[str] = None, custom_splits: Optional[str] = None
    ) -> str:
        """
        Resolves splits.json path.
        Priority: custom_splits -> /kaggle/working/splits.json -> ./splits.json -> data_root/splits.json.
        """
        if custom_splits and os.path.exists(custom_splits):
            return os.path.abspath(custom_splits)

        if os.path.exists("/kaggle/working/splits.json"):
            return "/kaggle/working/splits.json"

        if os.path.exists("./splits.json"):
            return os.path.abspath("./splits.json")

        if data_root is None:
            data_root = cls.find_dataset_root()

        candidate = os.path.join(data_root, "splits.json")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

        raise FileNotFoundError(f"splits.json could not be resolved in {data_root}")

    @classmethod
    def find_weights_path(
        cls, custom_path: Optional[str] = None, data_root: Optional[str] = None
    ) -> str:
        """Resolves model weights checkpoint path across local and Kaggle environments."""
        candidates = []
        if custom_path:
            candidates.append(custom_path)
        env_path = os.getenv("BEST_MODEL_WEIGHTS_PATH")
        if env_path:
            candidates.append(env_path)
        candidates.extend([
            "./models/dual_stream_best.pth",
            "models/dual_stream_best.pth",
            "./dual_stream_calibrated.pth",
            "dual_stream_calibrated.pth",
            "./models/dual_stream_calibrated.pth",
            "/kaggle/working/models/dual_stream_best.pth",
            "/kaggle/working/dual_stream_best.pth",
            "/kaggle/working/repo/models/dual_stream_best.pth",
        ])
        if data_root:
            candidates.append(os.path.join(data_root, "dual_stream_best.pth"))
            candidates.append(os.path.join(data_root, "dual_stream_calibrated.pth"))

        for p in candidates:
            if p and os.path.exists(p):
                return os.path.abspath(p)

        if os.path.exists("/kaggle/working"):
            for root, _, files in os.walk("/kaggle/working"):
                for name in ("dual_stream_best.pth", "dual_stream_calibrated.pth"):
                    if name in files:
                        return os.path.abspath(os.path.join(root, name))

        raise FileNotFoundError(f"Could not locate model weights. Checked: {candidates}")


find_dataset_root = DatasetResolver.find_dataset_root
resolve_splits_path = DatasetResolver.resolve_splits_path
find_weights_path = DatasetResolver.find_weights_path
