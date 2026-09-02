"""Unified PyTorch Dataset implementations for face crops and degradation testing."""

from collections.abc import Callable
import logging
import os
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2  # noqa: F401

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None


class FaceCropDataset(Dataset):
    """
    Dataset for single-frame face crops.
    Supports online degradation functions, Albumentations / Torchvision transforms,
    and corrupt sample valid_flag tracking for loss masking.
    """

    def __init__(
        self,
        samples: list[Union[tuple[str, Union[int, float]], list[Any]]],
        root_dir: Optional[str] = None,
        is_train: bool = True,
        transform: Optional[Any] = None,
        degradation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        img_size: int = 256,
        return_valid_flag: bool = True,
    ) -> None:
        self.samples = samples
        self.root_dir = root_dir or ""
        self.is_train = is_train
        self.transform = transform
        self.degradation_fn = degradation_fn
        self.img_size = img_size
        self.return_valid_flag = return_valid_flag

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_or_abs_path: str) -> tuple[np.ndarray, float]:
        full_path = (
            os.path.join(self.root_dir, rel_or_abs_path)
            if self.root_dir and not os.path.isabs(rel_or_abs_path)
            else rel_or_abs_path
        )
        valid_flag = 1.0
        try:
            bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cv2.imread failed for {full_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != self.img_size or rgb.shape[1] != self.img_size:
                rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        except (OSError, ValueError, cv2.error):
            valid_flag = 0.0
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        return rgb, valid_flag

    def __getitem__(self, idx: int) -> Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, int]]:
        entry = self.samples[idx]
        path_rel = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
        label_val = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else 0

        rgb, valid_flag = self._load_image(path_rel)

        if self.degradation_fn is not None and valid_flag > 0.0:
            try:
                rgb = self.degradation_fn(rgb)
            except Exception as e:
                logger.debug("Degradation hook failed: %s", e)
                valid_flag = 0.0

        if self.transform is not None and valid_flag > 0.0:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                augmented = self.transform(image=rgb)
                tensor_img = augmented["image"].float()
                if augmented["image"].dtype == torch.uint8:
                    tensor_img = tensor_img / 255.0
            else:
                img_pil = Image.fromarray(rgb)
                tensor_img = self.transform(img_pil)
        else:
            tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        valid_tensor = torch.tensor(valid_flag, dtype=torch.float32)

        if self.return_valid_flag:
            return tensor_img, label_tensor, valid_tensor
        return tensor_img, int(label_val)


KaggleFastDataset = FaceCropDataset
RobustnessDataset = FaceCropDataset
TestDataset = FaceCropDataset


class DeepfakeDataset(FaceCropDataset):
    """Single-frame dataset returning (tensor, int_label) for backward compatibility."""

    def __init__(self, samples: list[Any], transform: Optional[Any] = None) -> None:
        super().__init__(samples, root_dir="", transform=transform, return_valid_flag=False)
