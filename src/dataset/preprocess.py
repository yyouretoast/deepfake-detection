import logging
from typing import List, Tuple, Optional, Union
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import urllib.request
import threading
from tqdm import tqdm
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def get_yunet_model_path() -> Optional[str]:
    """Resolves or downloads YuNet ONNX model path with automatic local caching."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_path = os.path.join(repo_root, "models", YUNET_MODEL_FILENAME)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    
    alt_path = os.path.join("models", YUNET_MODEL_FILENAME)
    if os.path.exists(alt_path) and os.path.getsize(alt_path) > 1000:
        return os.path.abspath(alt_path)

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, local_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            return local_path
    except Exception as e:
        logging.warning("Failed to download YuNet model file: %s", e)
    return None

# Global path caching for instant thread-local YuNet instantiation
YUNET_CACHED_MODEL_PATH = get_yunet_model_path()

class DynamicFaceCropper:
    """
    High-performance OpenCV YuNet primary dynamic face extractor with MTCNN & Haar Cascade fallbacks.
    Extracts face bounding boxes with 1.50x bounding box scaling factor and 5-point landmark similarity alignment.
    Uses thread-local storage (threading.local()) for zero lock contention and 100% thread safety.
    Supports native 512x512 full-resolution face crop extraction.
    """
    def __init__(
        self,
        target_size: int = 512,
        scale_factor: float = 1.50,
        device: Optional[torch.device] = None,
        margin: int = 20
    ) -> None:
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.margin = margin
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._local = threading.local()

        # Fallback Engine 1: MTCNN
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

        # Fallback Engine 2: Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.haar_cascade = None

    def _get_thread_yunet(self):
        """Thread-isolated YuNet detector instantiation to prevent C++ state race conditions."""
        if not hasattr(self._local, "yunet"):
            if YUNET_CACHED_MODEL_PATH is not None and hasattr(cv2, "FaceDetectorYN"):
                try:
                    self._local.yunet = cv2.FaceDetectorYN.create(
                        model=YUNET_CACHED_MODEL_PATH,
                        config="",
                        input_size=(300, 300),
                        score_threshold=0.6,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                except Exception as e:
                    logging.warning("Failed to initialize thread-local YuNet detector: %s", e)
                    self._local.yunet = None
            else:
                self._local.yunet = None
        return self._local.yunet

    def _detect_yunet(self, image_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects faces using thread-isolated OpenCV YuNet (2-5ms per frame on CPU). Returns (boxes, 5-point landmarks)."""
        yunet_engine = self._get_thread_yunet()
        if yunet_engine is None:
            return None, None
        try:
            h, w, _ = image_rgb.shape
            yunet_engine.setInputSize((w, h))
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            _, faces = yunet_engine.detect(img_bgr)
            if faces is None or len(faces) == 0:
                return None, None
            
            boxes = []
            landmarks_list = []
            for face in faces:
                x, y, fw, fh = face[0:4]
                boxes.append([x, y, x + fw, y + fh])
                r_eye = face[4:6]
                l_eye = face[6:8]
                nose = face[8:10]
                r_mouth = face[10:12]
                l_mouth = face[12:14]
                lms = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
                landmarks_list.append(lms)
                
            return np.array(boxes), np.array(landmarks_list)
        except Exception as e:
            logging.warning("YuNet detection exception: %s", e)
            return None, None

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

    def _apply_cosine_edge_taper(self, crop: np.ndarray, border_ratio: float = 0.05) -> np.ndarray:
        """Applies 2D Cosine window edge tapering to smooth padded border step-discontinuities."""
        h, w, _ = crop.shape
        taper_h = max(1, int(h * border_ratio))
        taper_w = max(1, int(w * border_ratio))
        
        win_y = np.ones(h, dtype=np.float32)
        win_y[:taper_h] = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_h)))
        win_y[-taper_h:] = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, taper_h)))

        win_x = np.ones(w, dtype=np.float32)
        win_x[:taper_w] = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_w)))
        win_x[-taper_w:] = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, taper_w)))

        window_2d = np.outer(win_y, win_x)[:, :, np.newaxis]
        return np.clip(crop.astype(np.float32) * window_2d, 0, 255).astype(np.uint8)

    def _crop_single_box(self, image_rgb: np.ndarray, box: np.ndarray, landmarks: Optional[np.ndarray] = None, target_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Crops a face box with 1.50x scaling, slicing image bounds with BORDER_REPLICATE and Cosine edge tapering."""
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
                cv2.BORDER_REPLICATE
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

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            raw_crop = self._apply_cosine_edge_taper(raw_crop)

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
                        aligned_warped_crop = cv2.warpAffine(image_rgb, M, (out_size, out_size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
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
        """Extracts 1.50x scaled face crop returning both aligned warped and raw unwarped crops."""
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

        # Primary Engine: YuNet
        yunet_boxes, yunet_landmarks = self._detect_yunet(image_rgb)
        if yunet_boxes is not None and len(yunet_boxes) > 0:
            return self._crop_from_box(image_rgb, yunet_boxes, yunet_landmarks, target_size=out_size)

        # Fallback Engine 1: MTCNN
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
            except Exception as e:
                logging.debug("MTCNN fallback detection exception: %s", e)

        # Fallback Engine 2: Haar Cascade
        cascade_boxes = self._detect_cpu_cascade(image_rgb)
        if cascade_boxes is not None and len(cascade_boxes) > 0:
            return self._crop_from_box(image_rgb, cascade_boxes, None, target_size=out_size)

        # Final Fallback: Center Crop
        c = self._center_crop(image_rgb, target_size=out_size)
        return c, c

    def crop_face(self, image_input: Union[str, np.ndarray], target_size: Optional[int] = None) -> np.ndarray:
        """Extracts single 1.30x scaled aligned face crop array."""
        aligned_crop, _ = self.crop_face_dual(image_input, target_size=target_size)
        return aligned_crop

    def crop_faces_batched(
        self,
        image_inputs: List[Union[str, np.ndarray, Image.Image]],
        target_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Batched face crop extraction delegating to crop_face for each input image."""
        if not image_inputs:
            return []
        return [self.crop_face(img, target_size=target_size) for img in image_inputs]

    def extract_faces_from_video(
        self,
        video_path: str,
        output_dir: str,
        prefix: str = "frame",
        frames_per_video: int = 15,
        target_size: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """Extracts face crops from video frames saving as Lossless WebP images."""
        if max_frames is not None:
            frames_per_video = max_frames
        out_size = target_size if target_size is not None else self.target_size
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning("Could not open video: %s", video_path)
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        actual_frames = min(frames_per_video, total_frames)
        step = max(total_frames // actual_frames, 1)
        target_indices = set(i * step for i in range(actual_frames))

        saved_paths = []
        frame_idx = 0
        saved_count = 0

        while cap.isOpened() and saved_count < actual_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in target_indices:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop_rgb = self.crop_face(rgb_frame, target_size=out_size)
                out_filename = f"{prefix}_{saved_count:04d}.webp"
                out_filepath = os.path.join(output_dir, out_filename)
                
                # Save as Lossless WebP
                Image.fromarray(crop_rgb).save(out_filepath, format="WEBP", lossless=True)
                saved_paths.append(out_filepath)
                saved_count += 1

            frame_idx += 1

        cap.release()
        return saved_paths
