from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import torch
from PIL import Image

from src.config import load_config

try:
    from facenet_pytorch import MTCNN
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False

class DynamicFaceCropper:
    """
    Face extraction engine using facenet-pytorch (MTCNN) with relative dynamic padding.
    Supports single-image cropping and GPU batch cropping.
    Ingests global configuration from config/default.yaml via src/config.py.
    """
    def __init__(
        self,
        scale_factor: Optional[float] = None,
        target_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        if config is None:
            config = load_config()

        prep_cfg = config.get("preprocessing", {})
        if scale_factor is None:
            scale_factor = prep_cfg.get("padding_scale", 1.30)
        if target_size is None:
            target_size = prep_cfg.get("img_size", 224)

        self.scale_factor = scale_factor
        self.target_size = target_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if HAS_FACENET:
            self.mtcnn = MTCNN(
                keep_all=True,
                post_process=False,
                device=self.device,
                select_largest=True
            )
        else:
            self.mtcnn = None

    def crop_face(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detects largest face in single RGB image, applies relative padding, crops and resizes."""
        if image_rgb is None or image_rgb.size == 0:
            return None

        h, w, _ = image_rgb.shape
        if h == 0 or w == 0:
            return None

        boxes = None
        if self.mtcnn is not None:
            try:
                pil_img = Image.fromarray(image_rgb)
                boxes, _ = self.mtcnn.detect(pil_img)
            except Exception:
                boxes = None

        if boxes is None or len(boxes) == 0:
            return self._center_crop(image_rgb)

        return self._crop_from_box(image_rgb, boxes)

    def crop_faces_batched(self, images_rgb_list: List[np.ndarray]) -> List[np.ndarray]:
        """GPU batched face detection on a list of RGB image arrays."""
        if not images_rgb_list:
            return []

        cropped_faces: List[np.ndarray] = []

        if self.mtcnn is not None:
            try:
                pil_images = [Image.fromarray(img) for img in images_rgb_list]
                boxes_list, _ = self.mtcnn.detect(pil_images)
                if boxes_list is None:
                    boxes_list = [None] * len(images_rgb_list)
            except Exception:
                boxes_list = [None] * len(images_rgb_list)

            for img_rgb, boxes in zip(images_rgb_list, boxes_list):
                if boxes is None or len(boxes) == 0:
                    cropped_faces.append(self._center_crop(img_rgb))
                else:
                    crop = self._crop_from_box(img_rgb, boxes)
                    cropped_faces.append(crop if crop is not None else self._center_crop(img_rgb))
        else:
            for img_rgb in images_rgb_list:
                cropped_faces.append(self._center_crop(img_rgb))

        return cropped_faces

    def _get_resize_interpolation(self, src_h: int, target_h: int) -> int:
        """Selects optimal interpolation method based on upscaling vs downsampling."""
        return cv2.INTER_AREA if src_h >= target_h else cv2.INTER_CUBIC

    def _crop_from_box(self, image_rgb: np.ndarray, boxes) -> np.ndarray:
        h, w, _ = image_rgb.shape
        best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = best_box[:4]

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 5 or bh <= 5:
            return self._center_crop(image_rgb)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        new_bw = bw * self.scale_factor
        new_bh = bh * self.scale_factor

        raw_x1 = cx - new_bw / 2.0
        raw_y1 = cy - new_bh / 2.0
        raw_x2 = cx + new_bw / 2.0
        raw_y2 = cy + new_bh / 2.0

        pad_left = max(0, int(-raw_x1))
        pad_top = max(0, int(-raw_y1))
        pad_right = max(0, int(raw_x2 - w))
        pad_bottom = max(0, int(raw_y2 - h))

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_img = cv2.copyMakeBorder(
                image_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
            )
            nx1 = int(raw_x1 + pad_left)
            ny1 = int(raw_y1 + pad_top)
            nx2 = int(raw_x2 + pad_left)
            ny2 = int(raw_y2 + pad_top)
            face = padded_img[ny1:ny2, nx1:nx2]
        else:
            nx1 = max(0, int(raw_x1))
            ny1 = max(0, int(raw_y1))
            nx2 = min(w, int(raw_x2))
            ny2 = min(h, int(raw_y2))
            face = image_rgb[ny1:ny2, nx1:nx2]

        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            return self._center_crop(image_rgb)

        interp = self._get_resize_interpolation(face.shape[0], self.target_size)
        return cv2.resize(face, (self.target_size, self.target_size), interpolation=interp)

    def _center_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = image_rgb.shape
        if h <= 0 or w <= 0:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        half = max(1, min_dim // 2)
        crop = image_rgb[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]

        if crop.size == 0:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        interp = self._get_resize_interpolation(crop.shape[0], self.target_size)
        return cv2.resize(crop, (self.target_size, self.target_size), interpolation=interp)

def is_blurry(image_rgb: np.ndarray, threshold: Optional[float] = None, config: Optional[Dict[str, Any]] = None) -> bool:
    """Checks image blurriness using Laplacian variance, ingesting config defaults."""
    if threshold is None:
        if config is None:
            config = load_config()
        threshold = config.get("preprocessing", {}).get("blur_threshold_real", 30.0)

    if image_rgb is None or image_rgb.size == 0:
        return True
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var) < threshold
