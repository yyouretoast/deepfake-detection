import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from facenet_pytorch import MTCNN

    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False

YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def get_yunet_model_path() -> Optional[str]:
    """Resolve or download local YuNet ONNX model path."""
    curr = os.path.abspath(__file__)
    repo_root = None
    for _ in range(5):
        curr = os.path.dirname(curr)
        if os.path.exists(os.path.join(curr, "models")) or os.path.exists(os.path.join(curr, "config")):
            repo_root = curr
            break
    if repo_root is None:
        repo_root = os.getcwd()

    local_path = os.path.join(repo_root, "models", YUNET_MODEL_FILENAME)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    alt_path = os.path.join("models", YUNET_MODEL_FILENAME)
    if os.path.exists(alt_path) and os.path.getsize(alt_path) > 1000:
        return os.path.abspath(alt_path)

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        req = urllib.request.Request(
            YUNET_URL, headers={"User-Agent": "DeepfakeDetector/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            return local_path
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("Failed to download YuNet model file: %s", e)
    return None


_YUNET_CACHED_MODEL_PATH: Optional[str] = None


def get_cached_yunet_path() -> Optional[str]:
    """Lazily resolve and cache YuNet model path."""
    global _YUNET_CACHED_MODEL_PATH
    if _YUNET_CACHED_MODEL_PATH is None:
        _YUNET_CACHED_MODEL_PATH = get_yunet_model_path()
    return _YUNET_CACHED_MODEL_PATH


class DynamicFaceCropper:
    """Multi-engine face extractor with 5-point landmark similarity transform alignment and fallback detectors."""

    def __init__(
        self,
        target_size: int = 512,
        scale_factor: float = 1.50,
        device: Optional[torch.device] = None,
        margin: int = 20,
    ) -> None:
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.margin = margin
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._local = threading.local()

        if HAS_MTCNN:
            try:
                self.detector = MTCNN(
                    keep_all=True,
                    select_largest=True,
                    device=self.device,
                    post_process=False,
                )
            except (ImportError, OSError, ValueError, RuntimeError) as e:
                logger.debug("MTCNN initialization exception: %s", e)
                self.detector = None
        else:
            self.detector = None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.haar_cascade = None

    def _get_thread_yunet(self) -> Optional[Any]:
        """Fetch or instantiate thread-isolated YuNet detector."""
        if not hasattr(self._local, "yunet"):
            cached_path = get_cached_yunet_path()
            if cached_path is not None and hasattr(cv2, "FaceDetectorYN"):
                try:
                    self._local.yunet = cv2.FaceDetectorYN.create(
                        model=cached_path,
                        config="",
                        input_size=(300, 300),
                        score_threshold=0.6,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                except (cv2.error, OSError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to initialize thread-local YuNet detector: %s", e)
                    self._local.yunet = None
            else:
                self._local.yunet = None
        return self._local.yunet

    def _detect_yunet(
        self, image_rgb: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect faces using OpenCV YuNet on RGB array [H, W, 3]. Returns (bounding_boxes, 5_point_landmarks)."""
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
        except (cv2.error, AttributeError, ValueError, RuntimeError) as e:
            logger.warning("YuNet detection exception: %s", e)
            return None, None

    def _detect_cpu_cascade(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces using CPU Haar Cascade on RGB array [H, W, 3]."""
        if self.haar_cascade is None:
            return None
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return None
            boxes = [[x, y, x + w, y + h] for (x, y, w, h) in faces]
            return np.array(boxes)
        except (cv2.error, AttributeError, ValueError, RuntimeError) as e:
            logger.debug("Cascade detection exception: %s", e)
            return None

    def _apply_cosine_edge_taper(
        self, crop: np.ndarray, border_ratio: float = 0.05
    ) -> np.ndarray:
        """Apply 2D Cosine window edge tapering to smooth padded border step-discontinuities on crop [H, W, 3]."""
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

    def _crop_single_box(
        self,
        image_rgb: np.ndarray,
        box: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop face region from RGB image [H, W, 3] with scale factor and 5-point landmark similarity transform alignment."""
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
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
        else:
            padded_img = image_rgb

        src_y1 = crop_y1 + pad_top
        src_x1 = crop_x1 + pad_left
        src_y2 = crop_y2 + pad_top
        src_x2 = crop_x2 + pad_left

        if src_y2 <= src_y1 or src_x2 <= src_x1:
            logger.warning("Fallback center crop used due to invalid crop dimensions.")
            fallback = self._center_crop(image_rgb, target_size=out_size)
            return fallback, fallback

        raw_crop = padded_img[src_y1:src_y2, src_x1:src_x2]
        if raw_crop.size == 0:
            logger.warning("Fallback center crop used due to empty crop.")
            fallback = self._center_crop(image_rgb, target_size=out_size)
            return fallback, fallback

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            raw_crop = self._apply_cosine_edge_taper(raw_crop)

        raw_unwarped_crop = cv2.resize(raw_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        aligned_warped_crop = raw_unwarped_crop

        if landmarks is not None:
            # The canonical points define where each facial landmark should land in the
            # output crop. They must incorporate scale_factor so the landmark-aligned
            # warp produces the same effective crop zoom as the bounding-box path.
            # Without this, landmark crops are tightly framed (~1.0x) while bbox crops
            # use the full self.scale_factor expansion, creating a bimodal distribution.
            #
            # We model scale_factor as a zoom-out: push landmarks toward the center by
            # (1 - 1/scale_factor)/2 on each side.
            sf = max(self.scale_factor, 1.0)
            margin_frac = (1.0 - 1.0 / sf) / 2.0  # fraction of frame that is padding
            canonical_landmarks = (
                np.array(
                    [
                        [0.30, 0.35],
                        [0.70, 0.35],
                        [0.50, 0.50],
                        [0.35, 0.70],
                        [0.65, 0.70],
                    ],
                    dtype=np.float32,
                )
            )
            # Shift canonical points inward by margin_frac to match the scaled crop region
            canonical_landmarks = (canonical_landmarks * (1.0 - 2.0 * margin_frac) + margin_frac) * out_size

            try:
                M, inliers = cv2.estimateAffinePartial2D(
                    np.array(landmarks), canonical_landmarks, method=cv2.LMEDS
                )
                if M is not None:
                    det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
                    if 0.2 < det < 5.0:
                        aligned_warped_crop = cv2.warpAffine(
                            image_rgb,
                            M,
                            (out_size, out_size),
                            flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
            except (cv2.error, ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                logger.debug("Affine warp alignment exception: %s", e)

        return aligned_warped_crop, raw_unwarped_crop


    def _crop_from_box(
        self,
        image_rgb: np.ndarray,
        boxes: Any,
        landmarks: Any = None,
        target_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select largest detected bounding box and perform crop extraction."""
        out_size = target_size if target_size is not None else self.target_size
        if boxes is None or len(boxes) == 0:
            c = self._center_crop(image_rgb, target_size=out_size)
            return c, c

        if landmarks is not None and len(landmarks) == len(boxes):
            best_idx = max(
                range(len(boxes)),
                key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
            )
            best_box = boxes[best_idx]
            best_landmarks = landmarks[best_idx]
        else:
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            best_landmarks = None

        return self._crop_single_box(image_rgb, best_box, best_landmarks, target_size=out_size)

    def _center_crop(
        self, image_rgb: np.ndarray, target_size: Optional[int] = None
    ) -> np.ndarray:
        """Fallback center square crop when no face bounding box is detected."""
        out_size = target_size if target_size is not None else self.target_size
        h, w, _ = image_rgb.shape
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = image_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    def crop_face_dual(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        target_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract face crop from image returning (aligned_warped_crop, raw_unwarped_crop)."""
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

        yunet_boxes, yunet_landmarks = self._detect_yunet(image_rgb)
        if yunet_boxes is not None and len(yunet_boxes) > 0:
            return self._crop_from_box(image_rgb, yunet_boxes, yunet_landmarks, target_size=out_size)

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
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.debug("MTCNN fallback detection exception: %s", e)

        cascade_boxes = self._detect_cpu_cascade(image_rgb)
        if cascade_boxes is not None and len(cascade_boxes) > 0:
            return self._crop_from_box(image_rgb, cascade_boxes, None, target_size=out_size)

        c = self._center_crop(image_rgb, target_size=out_size)
        return c, c

    def crop_face(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extract aligned face crop RGB numpy array of shape [target_size, target_size, 3]."""
        aligned_crop, _ = self.crop_face_dual(image_input, target_size=target_size)
        return aligned_crop

    def crop_faces_batched(
        self,
        image_inputs: list[Union[str, np.ndarray, Image.Image]],
        target_size: Optional[int] = None,
    ) -> list[np.ndarray]:
        """Batched face crop extraction returning list of RGB face arrays [H, W, 3]."""
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
        max_frames: Optional[int] = None,
    ) -> list[str]:
        """Extract face crops from video frames saving lossless WebP images to output directory."""
        if max_frames is not None:
            frames_per_video = max_frames
        out_size = target_size if target_size is not None else self.target_size
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
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

                Image.fromarray(crop_rgb).save(out_filepath, format="WEBP", lossless=True)
                saved_paths.append(out_filepath)
                saved_count += 1

            frame_idx += 1

        cap.release()
        return saved_paths


def preprocess_tensors_batch(
    faces_rgb_list: list[np.ndarray], device: torch.device = torch.device("cpu")
) -> tuple[np.ndarray, torch.Tensor]:
    """Convert list of uint8 RGB face crop arrays [H, W, 3] to [B, 3, H, W] in [0, 1] range. Return (numpy_batch, torch_tensor_batch)."""
    batch_arr = np.stack(faces_rgb_list)
    batch_nchw = batch_arr.transpose(0, 3, 1, 2)

    tensor = torch.from_numpy(batch_nchw).float().to(device) / 255.0
    norm_nchw = tensor.cpu().numpy()
    return norm_nchw, tensor
