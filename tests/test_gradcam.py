import numpy as np
import torch
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
    gradcam = PyTorchGradCAM(model)
    
    spatial_layers = set(dict(model.spatial_backbone.named_modules()).values())
    freq_layers = set(dict(model.freq_extractor.named_modules()).values())
    
    assert gradcam.target_layer in spatial_layers, "Grad-CAM target layer must belong to spatial_backbone!"
    assert gradcam.target_layer not in freq_layers, "Grad-CAM target layer incorrectly resolved to freq_extractor!"

def test_gradcam_heatmap_generation():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with PyTorchGradCAM(model) as gradcam:
        heatmap = gradcam.generate_heatmap(dummy_input, target_class=1)
        
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0 + 1e-6

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

def test_gradcam_overlay():
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_heatmap = np.ones((224, 224), dtype=np.float32)
    
    blended = overlay_cam(dummy_image, dummy_heatmap, alpha=0.4)
    assert isinstance(blended, np.ndarray)
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8
