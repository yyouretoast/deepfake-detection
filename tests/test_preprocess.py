"""Unit tests for DynamicFaceCropper face detection and alignment math."""

from unittest.mock import MagicMock

import numpy as np

from src.dataset.preprocess import DynamicFaceCropper


def test_dynamic_face_cropper_synthetic_fallback() -> None:
    cropper = DynamicFaceCropper(target_size=512, scale_factor=1.50)

    synthetic_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    cropped_face = cropper.crop_face(synthetic_image)
    assert cropped_face is not None
    assert cropped_face.shape == (512, 512, 3), f"Expected shape (512, 512, 3), got {cropped_face.shape}"

    zero_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cropped_zero = cropper.crop_face(zero_image)
    assert cropped_zero is not None
    assert cropped_zero.shape == (512, 512, 3)


def test_dynamic_face_cropper_bounding_box_expansion_math() -> None:
    cropper = DynamicFaceCropper(target_size=512, scale_factor=1.50)

    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 100, 100]])
    cropper.haar_cascade = mock_cascade

    synthetic_image = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    cropped = cropper.crop_face(synthetic_image)

    assert cropped is not None
    assert cropped.shape == (512, 512, 3), f"Expected cropped shape (512, 512, 3), got {cropped.shape}"


def test_dynamic_face_cropper_similarity_transform_math() -> None:
    cropper = DynamicFaceCropper(target_size=256, scale_factor=1.50)
    synthetic_image = np.ones((400, 400, 3), dtype=np.uint8) * 128

    landmarks = np.array([
        [150.0, 150.0],
        [250.0, 150.0],
        [200.0, 200.0],
        [160.0, 260.0],
        [240.0, 260.0],
    ], dtype=np.float32)

    box = np.array([100, 100, 300, 300])
    aligned_face, raw_crop = cropper._crop_single_box(synthetic_image, box, landmarks=landmarks, target_size=256)
    assert aligned_face is not None
    assert aligned_face.shape == (256, 256, 3)
    assert raw_crop.shape == (256, 256, 3)
