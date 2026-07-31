import torch
import pytest
from src.models.hybrid_detector import HybridDeepfakeDetector

def test_hybrid_detector_dual_stream_forward(eval_model_factory, dummy_4d_batch):
    model = eval_model_factory(use_fft=True)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size]), f"Expected logits shape torch.Size([{b_size}]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size]), f"Expected probs shape torch.Size([{b_size}]), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"

def test_hybrid_detector_full_res_512_forward(eval_model_factory):
    """Verifies Pre-Downsample 512x512 FFT Frequency Extraction with GPU downscaling."""
    model = eval_model_factory(use_fft=True)

    dummy_512 = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(dummy_512)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2]), f"Expected logits shape torch.Size([2]), got {logits.shape}"
    assert probs.shape == torch.Size([2]), f"Expected probs shape torch.Size([2]), got {probs.shape}"

def test_hybrid_detector_spatial_only_forward(eval_model_factory, dummy_4d_batch):
    model = eval_model_factory(use_fft=False)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size]), f"Expected logits shape torch.Size([{b_size}]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size]), f"Expected probs shape torch.Size([{b_size}]), got {probs.shape}"

def test_use_fft_false_actually_disables_fft(eval_model_factory):
    """Regression test for BUG-1: config must not override explicit use_fft_branch=False."""
    model = eval_model_factory(use_fft=False)
    assert model.use_fft_branch is False, "use_fft_branch=False was overridden by config"
    assert model.freq_extractor is None, "freq_extractor should be None when use_fft_branch=False"

    first_linear = model.classifier[0]
    assert first_linear.in_features == model.spatial_backbone.num_features, (
        f"Classifier input dim {first_linear.in_features} != spatial backbone dim {model.spatial_backbone.num_features}"
    )

def test_legacy_1channel_weight_adapter(eval_model_factory):
    """Verifies that legacy 1-channel checkpoint weights are adapted to 2-channel FFT model."""
    model = eval_model_factory(use_fft=True)
    state = model.state_dict()
    state["freq_extractor.conv_net.0.weight"] = torch.randn(32, 1, 3, 3)
    
    model.load_state_dict(state)
    assert model.freq_extractor.conv_net[0].weight.shape == torch.Size([32, 2, 3, 3])
