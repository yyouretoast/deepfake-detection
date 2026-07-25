import numpy as np
import pytest
from src.dataset.preprocess import DynamicFaceCropper

def test_dynamic_face_cropper_bounds():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)

    # Synthetic 500x500 RGB image
    synthetic_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    cropped_face = cropper.crop_face(synthetic_img)

    assert cropped_face is not None
    assert cropped_face.shape == (256, 256, 3), f"Expected shape (256, 256, 3), got {cropped_face.shape}"

def test_dynamic_face_cropper_batched():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)

    img_list = [np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8) for _ in range(3)]
    cropped_list = cropper.crop_faces_batched(img_list)

    assert len(cropped_list) == 3
    for crop in cropped_list:
        assert crop.shape == (256, 256, 3)
