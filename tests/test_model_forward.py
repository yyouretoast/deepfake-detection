import torch
import pytest
from src.models.hybrid_detector import HybridDeepfakeDetector

def test_hybrid_detector_dual_stream_forward():
    model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=False, use_fft_branch=True)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2]), f"Expected logits shape torch.Size([2]), got {logits.shape}"
    assert probs.shape == torch.Size([2]), f"Expected probs shape torch.Size([2]), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"

def test_hybrid_detector_spatial_only_forward():
    model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=False, use_fft_branch=False)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([4]), f"Expected logits shape torch.Size([4]), got {logits.shape}"
    assert probs.shape == torch.Size([4]), f"Expected probs shape torch.Size([4]), got {probs.shape}"
