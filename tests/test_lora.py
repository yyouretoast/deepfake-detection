import torch
import pytest
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.lora import LoRAConv2d, apply_lora_to_model, merge_all_lora_weights, get_lora_state_dict

def test_lora_injection_and_parameter_reduction():
    """Verifies LoRA parameter reduction from 88M to <25M trainable parameters with spatial+channel adaptation."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=True, lora_rank=8)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params < 25_000_000, f"Trainable params {trainable_params} expected < 25M with LoRA"
    assert trainable_params < total_params * 0.25, f"Trainable params ratio should be < 25% of total ({total_params})"

def test_lora_depthwise_only_parameter_reduction():
    """Verifies LoRA depthwise-only parameter reduction in backbone to <1M parameters."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=False)
    apply_lora_to_model(model.spatial_backbone, rank=8, target_keywords=("dwconv",))

    backbone_trainable_params = sum(p.numel() for p in model.spatial_backbone.parameters() if p.requires_grad)
    assert backbone_trainable_params < 1_000_000, f"Depthwise-only backbone LoRA params {backbone_trainable_params} expected < 1M"

def test_lora_zero_initialization_parity():
    """Verifies LoRA starts with exact zero-op identity output (Epoch 0 parity)."""
    torch.manual_seed(42)
    model_base = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=False)
    
    torch.manual_seed(42)
    model_lora = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=True, lora_rank=8)

    model_base.eval()
    model_lora.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_base = model_base(dummy_input)
        out_lora = model_lora(dummy_input)

    assert torch.allclose(out_base, out_lora, atol=1e-4), "LoRA zero-initialization mismatch at Epoch 0"

def test_lora_weight_folding():
    """Verifies weight folding parity (merge_weights yields identical output with 0ms overhead)."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=True, lora_rank=8)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_unmerged = model(dummy_input)

    model.merge_lora_weights()

    with torch.no_grad():
        out_merged = model(dummy_input)

    assert torch.allclose(out_unmerged, out_merged, atol=1e-4), "Weight folding (merge_weights) parity check failed"

def test_get_lora_state_dict():
    """Verifies micro-checkpoint state dict extraction size."""
    model = HybridDeepfakeDetector(backbone_name="convnext_base", pretrained=False, use_fft_branch=True, use_lora=True, lora_rank=8)
    lora_sd = get_lora_state_dict(model)
    
    assert len(lora_sd) > 0, "Micro-checkpoint state dict should not be empty"
    for k, v in lora_sd.items():
        assert "lora_" in k or "classifier" in k or "cross_attn" in k or "spatial_proj" in k or "freq_proj" in k or v.requires_grad
