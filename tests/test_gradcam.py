import numpy as np
import torch
import torch.nn as nn
import pytest

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.explainability.gradcam import PyTorchGradCAM, overlay_cam

def test_gradcam_import_and_init():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    gradcam = PyTorchGradCAM(model)
    assert gradcam is not None
    assert gradcam.target_layer is not None

def test_gradcam_target_layer_spatial_redirection():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    gradcam = PyTorchGradCAM(model, target_stream="spatial")
    
    spatial_layers = set(dict(model.spatial_backbone.named_modules()).values())
    freq_layers = set(dict(model.freq_extractor.named_modules()).values())
    
    assert gradcam.target_layer in spatial_layers, "Grad-CAM target layer must belong to spatial_backbone!"
    assert gradcam.target_layer not in freq_layers, "Grad-CAM target layer incorrectly resolved to freq_extractor!"

def test_gradcam_target_layer_frequency_redirection():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    gradcam = PyTorchGradCAM(model, target_stream="frequency")
    
    spatial_layers = set(dict(model.spatial_backbone.named_modules()).values())
    freq_layers = set(dict(model.freq_extractor.named_modules()).values())
    
    assert gradcam.target_layer in freq_layers, "Grad-CAM target layer must belong to freq_extractor!"
    assert gradcam.target_layer not in spatial_layers, "Grad-CAM target layer incorrectly resolved to spatial_backbone!"

def test_gradcam_unregistration_and_memory():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    gradcam = PyTorchGradCAM(model)
    
    # Hooks should be registered
    assert gradcam.fh is not None
    assert gradcam.bh is not None
    
    gradcam.remove()
    
    assert gradcam.fh is None
    assert gradcam.bh is None

def test_gradcam_dataparallel():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    dp_model = nn.DataParallel(model)
    
    gradcam = PyTorchGradCAM(dp_model)
    assert gradcam.target_layer is not None
    assert gradcam.model == dp_model

def test_gradcam_heatmap_generation():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with PyTorchGradCAM(model) as gradcam:
        heatmap = gradcam.generate_heatmap(dummy_input, target_class=1)
        
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0 + 1e-6
    assert gradcam.feature_maps is None
    assert gradcam.gradients is None

def test_gradcam_target_class_signed_scores():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with PyTorchGradCAM(model) as gradcam:
        heatmap_real = gradcam.generate_heatmap(dummy_input, target_class=0)
        heatmap_fake = gradcam.generate_heatmap(dummy_input, target_class=1)
        
    assert isinstance(heatmap_real, np.ndarray)
    assert isinstance(heatmap_fake, np.ndarray)

def test_gradcam_batched_heatmap_generation():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    dummy_batch = torch.randn(4, 3, 224, 224)
    
    with PyTorchGradCAM(model) as gradcam:
        heatmaps = gradcam.generate_heatmaps_batch(dummy_batch, target_classes=[1, 0, 1, 0])
        
    assert isinstance(heatmaps, list)
    assert len(heatmaps) == 4
    for hm in heatmaps:
        assert isinstance(hm, np.ndarray)
        assert hm.shape == (224, 224)
        assert np.min(hm) >= 0.0
        assert np.max(hm) <= 1.0 + 1e-6

def test_gradcam_video_heatmap_generation():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    # Mocking video support for tests: [B, T, C, H, W]
    dummy_video = torch.randn(2, 5, 3, 224, 224)
    
    # We will pass the first frame just to bypass model shape errors if model doesn't support 5D natively
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 10, 3, padding=1)
            self.fc = nn.Linear(10*224*224, 1)
        def forward(self, x):
            if x.ndim == 5: x = x[:, 0]
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
            
    video_model = MockModel()
    with PyTorchGradCAM(video_model) as gradcam:
        heatmaps = gradcam.generate_heatmaps_batch(dummy_video, target_classes=[1, 0])
        
    assert isinstance(heatmaps, list)
    assert len(heatmaps) == 2
    for hm in heatmaps:
        assert hm.shape == (5, 224, 224)

def test_gradcam_overlay():
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_heatmap = np.ones((224, 224), dtype=np.float32)
    
    blended = overlay_cam(dummy_image, dummy_heatmap, alpha=0.4)
    assert isinstance(blended, np.ndarray)
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8

def test_gradcam_overlay_float_image():
    dummy_image_float = np.zeros((224, 224, 3), dtype=np.float32)
    dummy_heatmap = np.ones((224, 224), dtype=np.float32)
    
    blended = overlay_cam(dummy_image_float, dummy_heatmap, alpha=0.4)
    assert isinstance(blended, np.ndarray)
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8
