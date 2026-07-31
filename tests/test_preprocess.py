import numpy as np
import pytest
import os
from src.dataset.preprocess import DynamicFaceCropper

def test_dynamic_face_cropper_bounds():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)

    # Synthetic 500x500 RGB image
    synthetic_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    cropped_face = cropper.crop_face(synthetic_img)

    assert cropped_face is not None
    assert isinstance(cropped_face, np.ndarray), "crop_face must return a single ndarray"
    assert cropped_face.shape == (256, 256, 3), f"Expected shape (256, 256, 3), got {cropped_face.shape}"

def test_dynamic_face_cropper_dual():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)

    synthetic_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    res = cropper.crop_face_dual(synthetic_img)

    assert isinstance(res, tuple) and len(res) == 2, "crop_face_dual must return a 2-tuple"
    aligned, unaligned = res
    assert isinstance(aligned, np.ndarray)
    assert isinstance(unaligned, np.ndarray)
    assert aligned.shape == (256, 256, 3)
    assert unaligned.shape == (256, 256, 3)

def test_dynamic_face_cropper_batched():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)

    img_list = [np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8) for _ in range(3)]
    cropped_list = cropper.crop_faces_batched(img_list)

    assert len(cropped_list) == 3
    for crop in cropped_list:
        assert crop.shape == (256, 256, 3)

def test_dynamic_face_cropper_batched_edge_cases():
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)
    
    # Test empty list
    assert cropper.crop_faces_batched([]) == []
    
    # Test with PIL images
    from PIL import Image
    pil_list = [Image.fromarray(np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)) for _ in range(2)]
    cropped_pil = cropper.crop_faces_batched(pil_list)
    assert len(cropped_pil) == 2
    for crop in cropped_pil:
        assert crop.shape == (256, 256, 3)
    
    # Test with missing faces (where MTCNN returns None arrays for boxes)
    no_face_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cropped_no_face = cropper.crop_faces_batched([no_face_img])
    assert len(cropped_no_face) == 1
    assert cropped_no_face[0].shape == (256, 256, 3)

def test_extract_faces_from_video(tmp_path):
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=256)
    
    # Create a dummy video file
    video_path = str(tmp_path / "dummy.mp4")
    out_dir = str(tmp_path / "out")
    
    # Since we can't easily mock cv2.VideoCapture easily without mocking the module, 
    # and testing an actual video generation might be complex, we just check extension logic if we mocked it.
    # We will mock the method internally or just rely on the implementation if it's safe.
    # To be safe, we could write a simple test for the extension logic if we create a valid video, 
    # but the requirement is to check .webp extension.
    import cv2
    
    # Let's create a minimal valid video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 1.0, (100, 100))
    for _ in range(3):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()
    
    saved_paths = cropper.extract_faces_from_video(video_path, out_dir, max_frames=2)
    assert len(saved_paths) > 0
    for path in saved_paths:
        assert path.endswith(".webp"), f"Expected .webp extension, got {path}"
        assert os.path.exists(path)
