import pytest
import numpy as np
import torch
import cv2
from unittest.mock import patch, MagicMock

from app import (
    clean_state_dict,
    preprocess_tensors_batch,
    normalize_confidence,
    process_video_frames
)

@pytest.mark.fast
def test_clean_state_dict():
    input_dict = {
        "lora_layer.weight": torch.tensor([1.0]),
        "module.conv1.weight": torch.tensor([2.0]),
        "_orig_mod.fc.bias": torch.tensor([3.0]),
        "normal_layer.weight": torch.tensor([4.0])
    }
    
    cleaned = clean_state_dict(input_dict)
    
    assert "lora_layer.weight" not in cleaned
    assert "conv1.weight" in cleaned
    assert "fc.bias" in cleaned
    assert "normal_layer.weight" in cleaned
    
    assert cleaned["conv1.weight"].item() == 2.0
    assert cleaned["fc.bias"].item() == 3.0
    assert cleaned["normal_layer.weight"].item() == 4.0

@pytest.mark.fast
def test_preprocess_tensors_batch():
    faces = [np.ones((256, 256, 3), dtype=np.uint8) * 128, np.ones((256, 256, 3), dtype=np.uint8) * 200]
    norm_numpy, norm_tensor = preprocess_tensors_batch(faces, device=torch.device('cpu'))
    
    assert norm_numpy.shape == (2, 3, 256, 256)
    assert norm_tensor.shape == (2, 3, 256, 256)
    
    val = 128 / 255.0
    expected_c0 = (val - 0.485) / 0.229
    
    assert np.allclose(norm_numpy[0, 0, 0, 0], expected_c0, atol=1e-4)

@pytest.mark.fast
def test_normalize_confidence():
    threshold = 0.5
    assert normalize_confidence(0.75, threshold) == 75.0
    assert normalize_confidence(1.0, threshold) == 100.0
    assert normalize_confidence(0.25, threshold) == 75.0
    assert normalize_confidence(0.0, threshold) == 100.0

@pytest.mark.fast
def test_process_video_frames_temperature():
    with patch('app.cv2.VideoCapture') as mock_vid, \
         patch('app.DynamicFaceCropper') as mock_cropper, \
         patch('app.PyTorchGradCAM') as mock_gradcam:
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 5 
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        
        mock_vid.return_value = mock_cap
        
        mock_cropper_inst = mock_cropper.return_value
        mock_cropper_inst.crop_faces_batched.return_value = [np.zeros((256, 256, 3), dtype=np.uint8)] * 5
        
        mock_model = MagicMock()
        # Mock forward_sequence
        mock_model.forward_sequence.return_value = torch.tensor([[0.5]])
        # Mock __call__
        mock_model.return_value = torch.tensor([0.4])
        mock_model.module = mock_model
        
        res = process_video_frames(
            video_path="dummy.mp4",
            enable_gradcam=False,
            pytorch_model=mock_model,
            onnx_predictor=None,
            cropper=mock_cropper_inst,
            classification_threshold=0.5,
            temperature=2.0,
            has_pytorch_weights=True
        )
        
        assert res is not None
        assert "final_label" in res
        assert "final_conf" in res
        assert "real_frames" in res
        assert "fake_frames" in res
