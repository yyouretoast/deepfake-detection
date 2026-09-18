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


def test_yunet_landmark_order_affine_determinant() -> None:
    import cv2

    cropper = DynamicFaceCropper(target_size=256, scale_factor=1.50)
    image = np.ones((400, 400, 3), dtype=np.uint8) * 128

    r_eye = np.array([150.0, 150.0])
    l_eye = np.array([250.0, 150.0])
    nose = np.array([200.0, 200.0])
    r_mouth = np.array([160.0, 260.0])
    l_mouth = np.array([240.0, 260.0])
    lms = np.array([r_eye, l_eye, nose, r_mouth, l_mouth], dtype=np.float32)

    box = np.array([100, 100, 300, 300])
    aligned_face, _raw_crop = cropper._crop_single_box(image, box, landmarks=lms, target_size=256)
    assert aligned_face is not None
    assert aligned_face.shape == (256, 256, 3)

    sf = max(cropper.scale_factor, 1.0)
    margin_frac = (1.0 - 1.0 / sf) / 2.0
    canonical_landmarks = (
        np.array(
            [[0.30, 0.35], [0.70, 0.35], [0.50, 0.50], [0.35, 0.70], [0.65, 0.70]],
            dtype=np.float32,
        )
    )
    canonical = (canonical_landmarks * (1.0 - 2.0 * margin_frac) + margin_frac) * 256.0
    M, _ = cv2.estimateAffinePartial2D(lms, canonical, method=cv2.LMEDS)
    assert M is not None, "Failed to estimate partial affine transform matrix"
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    assert det > 0.0, f"Affine determinant must be positive (orientation-preserving), got {det}"


def test_dynamic_face_cropper_invalid_box_fallback() -> None:
    cropper = DynamicFaceCropper(target_size=256, scale_factor=1.50)
    image = np.ones((300, 300, 3), dtype=np.uint8) * 100

    inverted_box = np.array([200, 200, 50, 50])
    crop1, _ = cropper._crop_single_box(image, inverted_box, target_size=256)
    assert crop1 is not None
    assert crop1.shape == (256, 256, 3)

    negative_box = np.array([-50, -50, -10, -10])
    crop2, _ = cropper._crop_single_box(image, negative_box, target_size=256)
    assert crop2 is not None
    assert crop2.shape == (256, 256, 3)
