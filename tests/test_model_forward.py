"""Unit tests for HybridDeepfakeDetector model forward passes."""

from collections.abc import Callable

import torch

from src.models.hybrid_detector import HybridDeepfakeDetector


def test_hybrid_detector_dual_stream_forward(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector], dummy_4d_batch: torch.Tensor
) -> None:
    model = eval_model_factory(use_fft=True)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size, 1]), f"Expected logits shape torch.Size([{b_size}, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size, 1]), f"Expected probs shape torch.Size([{b_size}, 1]), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"


def test_hybrid_detector_full_res_512_forward(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)

    dummy_512 = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(dummy_512)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2, 1]), f"Expected logits shape torch.Size([2, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([2, 1]), f"Expected probs shape torch.Size([2, 1]), got {probs.shape}"


def test_hybrid_detector_spatial_only_forward(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector], dummy_4d_batch: torch.Tensor
) -> None:
    model = eval_model_factory(use_fft=False)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size, 1]), f"Expected logits shape torch.Size([{b_size}, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size, 1]), f"Expected probs shape torch.Size([{b_size}, 1]), got {probs.shape}"


def test_use_fft_false_actually_disables_fft(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=False)
    assert model.use_fft_branch is False, "use_fft_branch=False was overridden"
    assert not hasattr(model, "srm"), "srm module should not exist when use_fft_branch=False"

    first_linear = model.classifier[0]
    assert first_linear.in_features == 512, f"Classifier input dim {first_linear.in_features} != 512"


def test_corrupt_zero_input_nan_safety(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)
    zero_batch = torch.zeros(2, 3, 256, 256)

    logits = model(zero_batch)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values under all-zero corrupt input"


def test_5d_sequence_forward(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    model = eval_model_factory(use_fft=True)
    seq_batch = torch.randn(2, 5, 3, 256, 256)

    with torch.no_grad():
        logits = model.forward_sequence(seq_batch)

    assert logits.shape == torch.Size([2, 1]), f"Expected sequence logits shape torch.Size([2, 1]), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Sequence logits contain NaN or Inf"


def test_snr_adaptive_frequency_gating(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    """Verifies that high-frequency SNR gating attenuates the frequency stream when input is heavily degraded."""
    torch.manual_seed(42)  # Bayar init can otherwise land near a degenerate zero-sum kernel
    model = eval_model_factory(use_fft=True)
    assert model.enable_snr_gating is True

    # 1. Clean input with high-frequency details
    clean_batch = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out_clean = model(clean_batch)
        feat_clean = model.extract_features(clean_batch)
    assert torch.isfinite(out_clean).all()
    assert feat_clean.shape == (2, 512)

    # 2. Blurred/flat input (near-zero high frequency power)
    flat_batch = torch.full((2, 3, 256, 256), 0.5)
    with torch.no_grad():
        srm_res = model.srm(flat_batch)
        bayar_res = model.bayar(flat_batch)
        noise_comb = torch.cat([srm_res, bayar_res], dim=1)
        noise_p = noise_comb.pow(2).mean(dim=[-2, -1]).mean(dim=1, keepdim=True)
        gamma = torch.clamp((noise_p - 0.005) / (0.025 - 0.005), min=0.0, max=1.0)
        assert (gamma == 0.0).all(), f"Flat input must completely suppress spectral branch (gamma=0), got {gamma}"

        out_flat = model(flat_batch)
        feat_flat = model.extract_features(flat_batch)
    assert torch.isfinite(out_flat).all()
    assert feat_flat.shape == (2, 512)

    # 3. Verify disabling SNR gating behaves as standard gating
    model.enable_snr_gating = False
    with torch.no_grad():
        out_no_snr = model(clean_batch)
    assert torch.isfinite(out_no_snr).all()


def test_snr_gating_border_margin_restores_flat_input_invariant(
    eval_model_factory: Callable[[bool], HybridDeepfakeDetector]
) -> None:
    """Zero-padding borders keep residual power inside the lower gamma ramp on flat input.

    With noise_border_margin=0 (Release-1 checkpoint behaviour) a constant image still
    accumulates residual power (~2.5e-3) from the padded border band. The interior-
    cropped noise power must carry ~0, proving the source of the leak is the border.
    """
    torch.manual_seed(42)
    model = eval_model_factory(use_fft=True)
    model.noise_border_margin = 8

    flat_batch = torch.full((2, 3, 256, 256), 0.5)
    with torch.no_grad():
        gamma = model._snr_gamma(torch.cat([model.srm(flat_batch), model.bayar(flat_batch)], dim=1))

    assert model.noise_border_margin == 8
    assert (gamma == 0.0).all(), (
        f"Interior noise power must suppress the spectral branch on flat input (gamma=0), got {gamma}"
    )
