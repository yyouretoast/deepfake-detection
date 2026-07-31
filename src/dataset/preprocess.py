import logging
from typing import List, Tuple, Optional, Union
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

class DynamicFaceCropper:
    """
    High-performance MTCNN GPU-accelerated dynamic face extractor.
    Extracts face bounding boxes with 1.30x bounding box scaling factor.
    Supports native 512x512 full-resolution face crop extraction.
    """
    def __init__(
        self,
        target_size: int = 512,
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

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.haar_cascade = None

    def _detect_cpu_cascade(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Ultra-fast OpenCV Haar Cascade CPU face detector fallback (100+ FPS)."""
        if self.haar_cascade is None:
            return None
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return None
            boxes = []
            for (x, y, w, h) in faces:
                boxes.append([x, y, x + w, y + h])
            return np.array(boxes)
        except Exception:
            return None

    def _crop_single_box(self, image_rgb: np.ndarray, box: np.ndarray, landmarks: Optional[np.ndarray] = None, target_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Crops a face box with 1.30x scaling, slicing image bounds before applying reflection padding."""
        out_size = target_size if target_size is not None else self.target_size
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

        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - w_img)
        pad_bottom = max(0, crop_y2 - h_img)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padded_img = cv2.copyMakeBorder(
                image_rgb,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT
            )
        else:
            padded_img = image_rgb

        src_y1 = crop_y1 + pad_top
        src_x1 = crop_x1 + pad_left
        src_y2 = crop_y2 + pad_top
        src_x2 = crop_x2 + pad_left

        if src_y2 <= src_y1 or src_x2 <= src_x1:
            logging.warning("Fallback center crop used due to invalid crop dimensions.")
            fallback = self._center_crop(image_rgb, target_size=out_size)
            return fallback, fallback

        raw_crop = padded_img[src_y1:src_y2, src_x1:src_x2]
        if raw_crop.size == 0:
            logging.warning("Fallback center crop used due to empty crop.")
            fallback = self._center_crop(image_rgb, target_size=out_size)
            return fallback, fallback

        raw_unwarped_crop = cv2.resize(raw_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

        aligned_warped_crop = raw_unwarped_crop

        if landmarks is not None:
            canonical_landmarks = np.array([
                [0.30, 0.35],
                [0.70, 0.35],
                [0.50, 0.50],
                [0.35, 0.70],
                [0.65, 0.70]
            ], dtype=np.float32) * out_size

            try:
                M, inliers = cv2.estimateAffinePartial2D(np.array(landmarks), canonical_landmarks, method=cv2.LMEDS)
                if M is not None:
                    det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
                    if 0.2 < det < 5.0:
                        aligned_warped_crop = cv2.warpAffine(image_rgb, M, (out_size, out_size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
            except Exception:
                pass

        return aligned_warped_crop, raw_unwarped_crop

    def _crop_from_box(self, image_rgb: np.ndarray, boxes, landmarks=None, target_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Selects the largest detected bounding box and delegates to _crop_single_box."""
        out_size = target_size if target_size is not None else self.target_size
        if boxes is None or len(boxes) == 0:
            c = self._center_crop(image_rgb, target_size=out_size)
            return c, c
        
        if landmarks is not None and len(landmarks) == len(boxes):
            best_idx = max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
            best_box = boxes[best_idx]
            best_landmarks = landmarks[best_idx]
        else:
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            best_landmarks = None

        return self._crop_single_box(image_rgb, best_box, best_landmarks, target_size=out_size)

    def _center_crop(self, image_rgb: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
        """Fallback center crop when no face bounding box is detected."""
        out_size = target_size if target_size is not None else self.target_size
        h, w, _ = image_rgb.shape
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = image_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    def crop_face_dual(self, image_input: Union[str, np.ndarray], target_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts 1.30x scaled face crop from image returning both aligned warped and raw unwarped crops."""
        out_size = target_size if target_size is not None else self.target_size
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Failed to read image file: {image_input}")
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image_rgb = np.array(image_input)
        else:
            image_rgb = image_input

        if self.detector is not None:
            try:
                with torch.no_grad():
                    try:
                        res = self.detector.detect(image_rgb, landmarks=True)
                    except TypeError:
                        res = self.detector.detect(image_rgb)
                
                boxes = res[0]
                landmarks = res[2] if len(res) >= 3 else None
                if boxes is not None and len(boxes) > 0:
                    return self._crop_from_box(image_rgb, boxes, landmarks, target_size=out_size)
            except Exception:
                pass

        cascade_boxes = self._detect_cpu_cascade(image_rgb)
        if cascade_boxes is not None and len(cascade_boxes) > 0:
            return self._crop_from_box(image_rgb, cascade_boxes, None, target_size=out_size)

        c = self._center_crop(image_rgb, target_size=out_size)
        return c, c

    def crop_face(self, image_input: Union[str, np.ndarray], target_size: Optional[int] = None) -> np.ndarray:
        """Extracts 1.30x scaled face crop from image filepath or RGB numpy array."""
        aligned, _ = self.crop_face_dual(image_input, target_size=target_size)
        return aligned

    def crop_faces_batched(self, images_rgb: List[np.ndarray], target_size: Optional[int] = None) -> List[np.ndarray]:
        """Batched face extraction returning one face crop per image."""
        out_size = target_size if target_size is not None else self.target_size
        try:
            if self.detector is not None:
                pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images_rgb]
                with torch.no_grad():
                    try:
                        res = self.detector.detect(pil_images, landmarks=True)
                    except TypeError:
                        res = self.detector.detect(pil_images)
                
                boxes_list = res[0]
                landmarks_list = res[2] if len(res) >= 3 else [None] * len(boxes_list)
                
                results = []
                for img, boxes, lms in zip(images_rgb, boxes_list, landmarks_list):
                    img_np = np.array(img) if not isinstance(img, np.ndarray) else img
                    aligned, _ = self._crop_from_box(img_np, boxes, lms, target_size=out_size)
                    results.append(aligned)
                
                if len(results) == len(images_rgb):
                    return results

        except Exception:
            pass
            
        return [self.crop_face(img, target_size=out_size) for img in images_rgb]

    def extract_faces_from_video(self, video_path: str, output_dir: str, max_frames: int = 30, target_size: Optional[int] = None) -> List[str]:
        """Extracts face crops from video MP4 file into output directory."""
        out_size = target_size if target_size is not None else self.target_size
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

        try:
            for i, idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = self.crop_face(rgb, target_size=out_size)
                out_path = os.path.join(output_dir, f"{video_name}_frame{i:03d}.webp")
                Image.fromarray(crop).save(out_path, format="WEBP", lossless=True)
                saved_paths.append(out_path)
        finally:
            cap.release()
        return saved_paths
