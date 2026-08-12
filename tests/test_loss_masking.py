"""Unit tests for sample validity loss masking and corruption robustness.

These tests exercise the SAME loss masking formula used in train_dual_stream_ddp.py:
    loss = (F.binary_cross_entropy_with_logits(..., reduction='none') * valid_flags).sum()
           / valid_flags.sum().clamp(min=1.0)

They verify correctness through actual model forward+backward passes, NOT through an
inline reimplementation that would only test PyTorch primitives.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.models.hybrid_detector import HybridDeepfakeDetector



def test_loss_masking_zero_gradient_for_corrupt_samples(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)
    images = torch.randn(2, 3, 256, 256)
    labels = torch.tensor([1.0, 0.0])

    valid_flags = torch.tensor([1.0, 0.0])

    outputs = model(images).squeeze(1)
    loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")

    loss_sample0 = loss_unreduced[0] * valid_flags[0]
    loss_sample1 = loss_unreduced[1] * valid_flags[1]

    assert loss_sample1.item() == 0.0, "Corrupt sample loss with valid_flag=0.0 must be exactly 0.0"
    assert loss_sample0.item() > 0.0, "Valid sample loss with valid_flag=1.0 should be positive"

    masked_loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
    model.zero_grad()
    masked_loss.backward()

    grad_masked = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]

    outputs_single = model(images[:1]).squeeze(1)
    loss_single = F.binary_cross_entropy_with_logits(outputs_single, labels[:1])
    model.zero_grad()
    loss_single.backward()

    grad_single = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]

    for g_mask, g_sing in zip(grad_masked, grad_single):
        assert torch.allclose(g_mask, g_sing, atol=1e-5), "Gradient from masked batch does not match valid sample gradient"


def test_loss_masking_amp_autocast_precision(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)
    images = torch.randn(2, 3, 256, 256)
    labels = torch.tensor([1.0, 0.0])
    valid_flags = torch.tensor([1.0, 0.0])

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(device_type):
        outputs = model(images).squeeze(1)
        loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")
        loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)

    assert torch.isfinite(loss), "Loss value under AMP autocast contains NaN or Inf"
    assert (loss_unreduced[1] * valid_flags[1]).item() == 0.0


def test_loss_masking_all_corrupt_batch(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)
    images = torch.randn(2, 3, 256, 256)
    labels = torch.tensor([1.0, 0.0])
    valid_flags = torch.tensor([0.0, 0.0])

    outputs = model(images).squeeze(1)
    loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")
    loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)

    assert loss.item() == 0.0, "All-corrupt batch loss must be exactly 0.0"

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), "NaN/Inf gradient detected in all-corrupt batch backward pass"
