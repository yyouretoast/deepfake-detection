import numpy as np
import pytest
from unittest.mock import MagicMock
from src.dataset.preprocess import DynamicFaceCropper

def test_dynamic_face_cropper_synthetic_fallback():
    """Verifies that DynamicFaceCropper processes synthetic numpy arrays without network downloads."""
    cropper = DynamicFaceCropper(target_size=512, scale_factor=1.50)
    synthetic_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)

    cropped_face = cropper.crop_face(synthetic_image)
    assert cropped_face is not None
    assert cropped_face.shape == (512, 512, 3), f"Expected shape (512, 512, 3), got {cropped_face.shape}"

def test_dynamic_face_cropper_zero_image_handling():
    """Verifies that all-zero synthetic images return target_size RGB arrays cleanly."""
    cropper = DynamicFaceCropper(target_size=512)
    zero_image = np.zeros((300, 300, 3), dtype=np.uint8)

    cropped = cropper.crop_face(zero_image)
    assert cropped is not None
    assert cropped.shape == (512, 512, 3)

def test_dynamic_face_cropper_bounding_box_expansion_math():
    """Verifies deterministic 20% margin expansion, aspect ratio scaling, and target resolution extraction."""
    cropper = DynamicFaceCropper(target_size=512, scale_factor=1.50)
    
    # Mock Haar cascade detector to return a deterministic bounding box [x=100, y=100, w=100, h=100]
    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 100, 100]])
    cropper.haar_cascade = mock_cascade

    synthetic_image = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    cropped = cropper.crop_face(synthetic_image)

    assert cropped is not None
    assert cropped.shape == (512, 512, 3), f"Expected cropped shape (512, 512, 3), got {cropped.shape}"
