import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

def test_loss_masking_zero_gradient_for_corrupt_samples(eval_model_factory):
    """
    Asserts that samples marked with valid_flag = 0.0 produce zero loss weight and zero gradient contribution.
    """
    model = eval_model_factory(use_fft=True)
    images = torch.randn(2, 3, 256, 256)
    labels = torch.tensor([1.0, 0.0])
    
    # Sample 0 is valid (1.0), Sample 1 is corrupt/invalid (0.0)
    valid_flags = torch.tensor([1.0, 0.0])

    outputs = model(images)
    loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
    
    loss_sample0 = loss_unreduced[0] * valid_flags[0]
    loss_sample1 = loss_unreduced[1] * valid_flags[1]

    assert loss_sample1.item() == 0.0, "Corrupt sample loss with valid_flag=0.0 must be exactly 0.0"
    assert loss_sample0.item() > 0.0, "Valid sample loss with valid_flag=1.0 should be positive"

    masked_loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
    model.zero_grad()
    masked_loss.backward()

    grad_masked = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]

    outputs_single = model(images[:1])
    loss_single = F.binary_cross_entropy_with_logits(outputs_single, labels[:1])
    model.zero_grad()
    loss_single.backward()

    grad_single = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]

    for g_mask, g_sing in zip(grad_masked, grad_single):
        assert torch.allclose(g_mask, g_sing, atol=1e-5), "Gradient from masked batch does not match valid sample gradient"
