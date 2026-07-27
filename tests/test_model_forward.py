import torch
import pytest
from src.models.hybrid_detector import HybridDeepfakeDetector

def test_hybrid_detector_dual_stream_forward():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2]), f"Expected logits shape torch.Size([2]), got {logits.shape}"
    assert probs.shape == torch.Size([2]), f"Expected probs shape torch.Size([2]), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"

def test_hybrid_detector_full_res_512_forward():
    """Verifies Pre-Downsample 512x512 FFT Frequency Extraction with GPU downscaling."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True)
    model.eval()

    dummy_512 = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(dummy_512)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2]), f"Expected logits shape torch.Size([2]), got {logits.shape}"
    assert probs.shape == torch.Size([2]), f"Expected probs shape torch.Size([2]), got {probs.shape}"

def test_hybrid_detector_spatial_only_forward():
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=False)
    model.eval()

    dummy_input = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([4]), f"Expected logits shape torch.Size([4]), got {logits.shape}"
    assert probs.shape == torch.Size([4]), f"Expected probs shape torch.Size([4]), got {probs.shape}"

def test_use_fft_false_actually_disables_fft():
    """Regression test for BUG-1: config must not override explicit use_fft_branch=False."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=False)
    assert model.use_fft_branch is False, "use_fft_branch=False was overridden by config"
    assert model.freq_extractor is None, "freq_extractor should be None when use_fft_branch=False"

    first_linear = model.classifier[0]
    assert first_linear.in_features == model.spatial_backbone.num_features, (
        f"Classifier input dim {first_linear.in_features} != spatial backbone dim {model.spatial_backbone.num_features}"
    )
