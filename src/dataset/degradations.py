"""Image degradation and corruption functions for robustness testing and perturbation benchmarks."""

from collections.abc import Callable

import cv2
import numpy as np


def jpeg_fn(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    """JPEG compression degradation at specified quality level (0-100)."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return fn


def blur_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Gaussian blur degradation at specified kernel sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        ksize = int(6 * sigma + 1) | 1
        return cv2.GaussianBlur(rgb, (ksize, ksize), sigma)

    return fn


def noise_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Additive Gaussian noise degradation at specified pixel-space sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*rgb.shape).astype(np.float32) * sigma
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return fn


def downscale_fn(scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """Downscale and re-upsample image back to original resolution."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = cv2.resize(
            rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return fn


__all__ = [
    "blur_fn",
    "downscale_fn",
    "jpeg_fn",
    "noise_fn",
]
