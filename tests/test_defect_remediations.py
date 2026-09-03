"""Regression test suite for verified defect remediations across model, training, and preprocessing."""

import os
import tempfile
import threading
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

from scripts.train_dual_stream_ddp import (
    ExponentialMovingAverage,
    FocalLossWithLogits,
    find_dataset_root,
)
from scripts.train_loto_experiment import matches_holdout_domain
from src.dataset.preprocess import DynamicFaceCropper
from src.models.hybrid_detector import RealFFT2DModule
from src.utils.interpretability import MODEL_INFERENCE_LOCK


def test_gradient_accumulation_order() -> None:
    """Verifies that micro-batch 0 gradients are NOT wiped before micro-batch 1 under accelerator.accumulate."""
    acc = Accelerator(gradient_accumulation_steps=2)
    model = nn.Linear(4, 1, bias=False)
    model.weight.data.fill_(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    model, opt = acc.prepare(model, opt)

    x0 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    x1 = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

    for i, x in enumerate([x0, x1]):
        with acc.accumulate(model):
            loss = model(x).sum()
            acc.backward(loss)
            if acc.sync_gradients:
                opt.step()
                opt.zero_grad(set_to_none=True)

    # Expected: x0 contributes 0.5, x1 contributes 1.0 -> total grad 1.5 -> w = 1.0 - 0.1 * 1.5 = 0.85
    expected = torch.tensor([[0.85, 0.85, 0.85, 0.85]])
    assert torch.allclose(model.weight.data, expected, atol=1e-5), f"Expected {expected}, got {model.weight.data}"


def test_real_fft_phase_gradient_stability() -> None:
    """Verifies that RealFFT2DModule phase autograd does NOT diverge or produce NaN/inf at near-zero bins."""
    fft_mod = RealFFT2DModule()

    # 1. Zero input
    x_zero = torch.zeros(1, 10, 32, 32, requires_grad=True)
    out_zero = fft_mod(x_zero)
    out_zero.sum().backward()
    assert not torch.isnan(x_zero.grad).any(), "Gradient contains NaN on zero input"
    assert not torch.isinf(x_zero.grad).any(), "Gradient contains Inf on zero input"

    # 2. Sub-epsilon input
    x_sub = (torch.ones(1, 10, 32, 32) * 1e-12).requires_grad_(True)
    out_sub = fft_mod(x_sub)
    out_sub.sum().backward()
    assert not torch.isnan(x_sub.grad).any(), "Gradient contains NaN on sub-epsilon input"
    assert not torch.isinf(x_sub.grad).any(), "Gradient contains Inf on sub-epsilon input"
    assert x_sub.grad.abs().max() < 10.0, f"Gradient exploded: max={x_sub.grad.abs().max()}"

    # 3. Normal signal input
    x_norm = torch.randn(1, 10, 32, 32, requires_grad=True)
    out_norm = fft_mod(x_norm)
    out_norm.sum().backward()
    assert not torch.isnan(x_norm.grad).any()
    assert not torch.isinf(x_norm.grad).any()


def test_crop_face_fallback_modes() -> None:
    """Verifies fallback_on_empty parameter preserves backwards compatibility while supporting None returns."""
    cropper = DynamicFaceCropper(target_size=256)
    noise_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # fallback_on_empty=True (default) MUST return (256, 256, 3)
    crop_fallback = cropper.crop_face(noise_img, fallback_on_empty=True)
    assert crop_fallback is not None
    assert crop_fallback.shape == (256, 256, 3)

    # fallback_on_empty=False MUST return None when no face is found
    crop_none = cropper.crop_face(noise_img, fallback_on_empty=False)
    assert crop_none is None

    # Dual crop mode
    aligned, raw = cropper.crop_face_dual(noise_img, fallback_on_empty=False)
    assert aligned is None and raw is None


def test_inference_lock_concurrency() -> None:
    """Verifies MODEL_INFERENCE_LOCK functions as a reentrant lock and guards model execution."""
    model = nn.Linear(4, 1)

    def thread_worker():
        with MODEL_INFERENCE_LOCK:
            x = torch.randn(2, 4)
            y = model(x).sum()
            y.backward()
            model.zero_grad()

    t1 = threading.Thread(target=thread_worker)
    t2 = threading.Thread(target=thread_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def test_focal_loss_and_ema_integration() -> None:
    """Verifies FocalLossWithLogits unreduced masking and ExponentialMovingAverage lifecycle."""
    criterion = FocalLossWithLogits(gamma=2.0)
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = criterion(logits, targets)
    assert loss.shape == (2, 1)
    assert not torch.isnan(loss).any()

    # EMA lifecycle
    m = nn.Linear(2, 2)
    m.weight.data.fill_(1.0)
    ema = ExponentialMovingAverage(m, decay=0.9)

    m.weight.data.fill_(2.0)
    ema.update(m)
    assert torch.allclose(ema.shadow["weight"], torch.tensor([[1.1, 1.1], [1.1, 1.1]]))

    backup = ema.apply_shadow(m)
    assert torch.allclose(m.weight.data, torch.tensor([[1.1, 1.1], [1.1, 1.1]]))

    ema.restore(m, backup)
    assert torch.allclose(m.weight.data, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))


def test_loto_path_invariant_matching() -> None:
    """Verifies matches_holdout_domain handles flat, nested, and Kaggle path prefixes correctly."""
    assert matches_holdout_domain("fake/400_403/frame_001.webp", "faceswap") is True
    assert matches_holdout_domain("fake/ff_c23/400_403/frame_001.webp", "faceswap") is True
    assert matches_holdout_domain("/kaggle/input/deepfake-face-crops-256/deepfake_crops_512/fake/400_403/frame_001.webp", "faceswap") is True
    assert matches_holdout_domain("fake/250_253/frame_001.webp", "face2face") is True
    assert matches_holdout_domain("fake/250_253/frame_001.webp", "faceswap") is False
    assert matches_holdout_domain("fake/450_453/frame_001.webp", "faceswap") is True

    assert matches_holdout_domain("fake/600_603/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/ff_c23/600_603/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("/kaggle/input/deepfake-face-crops-256/deepfake_crops_512/fake/600_603/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/350_355/frame_001.webp", "face2face") is True
    assert matches_holdout_domain("fake/350_355/frame_001.webp", "neuraltextures") is False
    assert matches_holdout_domain("fake/650_653/frame_001.webp", "neuraltextures") is True

    assert matches_holdout_domain("fake/400_403/frame_001.webp", "deepfakes") is False
    assert matches_holdout_domain("fake/400_403/frame_001.webp", "face2face") is False


def test_find_dataset_root_filters_empty_code_repo() -> None:
    """Verifies find_dataset_root prioritizes directories containing actual fake/ image folders over code repos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        crops_dir = os.path.join(tmpdir, "deepfake_crops_512")
        os.makedirs(os.path.join(crops_dir, "fake"))
        with open(os.path.join(crops_dir, "splits.json"), "w") as f:
            f.write("{}")

        root = find_dataset_root(custom_dir=crops_dir)
        assert root == crops_dir
