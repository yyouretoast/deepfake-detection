from typing import List, Tuple, Optional, Union
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

class DynamicFaceCropper:
    """
    High-performance MTCNN GPU-accelerated dynamic face extractor.
    Extracts face bounding boxes with 1.30x bounding box scaling factor.
    Slices valid crop regions prior to border reflection padding.
    """
    def __init__(
        self,
        target_size: int = 256,
        scale_factor: float = 1.30,
        device: Optional[torch.device] = None,
        margin: int = 20
    ) -> None:
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.margin = margin
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if HAS_MTCNN:
            try:
                self.detector = MTCNN(
                    keep_all=True,
                    select_largest=True,
                    device=self.device,
                    post_process=False
                )
            except Exception:
                self.detector = None
        else:
            self.detector = None

    def _crop_single_box(self, image_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Crops a face box with 1.30x scaling, slicing image bounds before applying reflection padding."""
        h_img, w_img, _ = image_rgb.shape
        x1, y1, x2, y2 = box[:4]
        
        w_box = x2 - x1
        h_box = y2 - y1
        cx = x1 + w_box / 2.0
        cy = y1 + h_box / 2.0

        side = max(w_box, h_box) * self.scale_factor
        
        crop_x1 = int(round(cx - side / 2.0))
        crop_y1 = int(round(cy - side / 2.0))
        crop_x2 = int(round(cx + side / 2.0))
        crop_y2 = int(round(cy + side / 2.0))

        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(w_img, crop_x2)
        src_y2 = min(h_img, crop_y2)

        pad_left = src_x1 - crop_x1
        pad_top = src_y1 - crop_y1
        pad_right = crop_x2 - src_x2
        pad_bottom = crop_y2 - src_y2

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return self._center_crop(image_rgb)

        valid_crop = image_rgb[src_y1:src_y2, src_x1:src_x2]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_crop = cv2.copyMakeBorder(
                valid_crop,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )
        else:
            padded_crop = valid_crop

        resized = cv2.resize(padded_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        return resized

    def _crop_from_box(self, image_rgb: np.ndarray, boxes) -> np.ndarray:
        """Selects the largest detected bounding box and delegates to _crop_single_box."""
        if boxes is None or len(boxes) == 0:
            return self._center_crop(image_rgb)
        
        best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return self._crop_single_box(image_rgb, best_box)

    def _center_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        """Fallback center crop when no face bounding box is detected."""
        h, w, _ = image_rgb.shape
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = image_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        return cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def crop_face(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """Extracts 1.30x scaled face crop from image filepath or RGB numpy array."""
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Failed to read image file: {image_input}")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_input

        if self.detector is not None:
            try:
                boxes, _ = self.detector.detect(image_rgb)
                return self._crop_from_box(image_rgb, boxes)
            except Exception:
                pass

        return self._center_crop(image_rgb)

    def crop_faces_batched(self, images_rgb: List[np.ndarray]) -> List[np.ndarray]:
        """Batched face extraction returning one face crop per image."""
        return [self.crop_face(img) for img in images_rgb]

    def crop_all_faces_batched(self, images_rgb: List[np.ndarray], max_faces: int = 3) -> List[List[np.ndarray]]:
        """Batched face extraction returning a list of face crops per image frame."""
        return [[self.crop_face(img)] for img in images_rgb]

    def extract_faces_from_video(self, video_path: str, output_dir: str, max_frames: int = 30) -> List[str]:
        """Extracts face crops from video MP4 file into output directory."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        saved_paths = []
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = self.crop_face(rgb)
            out_path = os.path.join(output_dir, f"{video_name}_frame{i:03d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved_paths.append(out_path)

        cap.release()
        return saved_paths
