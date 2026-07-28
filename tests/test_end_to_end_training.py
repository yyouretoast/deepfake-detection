import os
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.hybrid_detector import build_model
from src.training.trainer import TwoPhaseTrainer


def test_end_to_end_multi_epoch_training(tmp_path):
    """
    End-to-end multi-epoch integration test:
    1. Executes 2 full epochs of Phase 1 and 2 full epochs of Phase 2 on a dummy dataset.
    2. Verifies all epoch losses are non-NaN and show an overall decrease.
    3. Saves checkpoint and verifies state_dict reload parity & forward pass output parity.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Distinguishable dummy data to ensure loss reduction
    real_x = torch.randn(8, 3, 256, 256) * 0.1
    fake_x = torch.randn(8, 3, 256, 256) * 0.1 + 2.0
    dummy_x = torch.cat([real_x, fake_x], dim=0)
    dummy_y = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)

    dataset = TensorDataset(dummy_x, dummy_y)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = build_model(use_fft=True, pretrained=False)

    config = {
        "training": {
            "epochs_phase1": 2,
            "epochs_phase2": 2,
            "lr_phase1": 1e-3,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "use_amp": False,
            "patience": 10,
        }
    }

    trainer = TwoPhaseTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    checkpoint_data, opt_thresh, metrics = trainer.train()

    # 1. Verify structure and non-NaN losses
    assert "state_dict" in checkpoint_data
    assert "optimal_threshold" in checkpoint_data
    assert "epoch_losses" in metrics, "epoch_losses should be reported in evaluation metrics"

    epoch_losses = metrics["epoch_losses"]
    assert len(epoch_losses) == 4, f"Expected 4 epoch losses (2 for Phase 1, 2 for Phase 2), got {len(epoch_losses)}"

    for loss_val in epoch_losses:
        assert not math.isnan(loss_val), f"Encountered NaN loss value: {loss_val}"
        assert not math.isinf(loss_val), f"Encountered Inf loss value: {loss_val}"

    # Verify non-NaN loss decrease (final epoch loss < initial epoch loss)
    assert epoch_losses[-1] < epoch_losses[0], (
        f"Expected loss to decrease from initial ({epoch_losses[0]:.4f}) to final ({epoch_losses[-1]:.4f})"
    )

    # 2. Save checkpoint to disk
    save_path = os.path.join(tmp_path, "end_to_end_checkpoint.pth")
    torch.save(checkpoint_data, save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

    # 3. Reload checkpoint into a fresh model & verify state_dict reload parity
    loaded_ckpt = torch.load(save_path, weights_only=False)
    eval_model = build_model(use_fft=True, pretrained=False)
    eval_model.load_state_dict(loaded_ckpt["state_dict"])
    eval_model.eval()

    # Verify state_dict keys parity
    orig_keys = set(checkpoint_data["state_dict"].keys())
    reloaded_keys = set(eval_model.state_dict().keys())
    assert orig_keys == reloaded_keys, "State dict keys mismatch after reload"

    # Verify parameter parity
    for key, val in checkpoint_data["state_dict"].items():
        reloaded_val = eval_model.state_dict()[key]
        assert torch.equal(val, reloaded_val), f"Parameter mismatch for key: {key}"

    # Verify forward pass output parity between original model and reloaded model
    unwrapped_orig = model.module if hasattr(model, "module") else model
    unwrapped_orig.eval()
    with torch.no_grad():
        orig_preds = torch.sigmoid(unwrapped_orig(dummy_x[:4]))
        reloaded_preds = torch.sigmoid(eval_model(dummy_x[:4]))
        assert torch.allclose(orig_preds, reloaded_preds, atol=1e-6), (
            f"Forward pass output mismatch between original and reloaded model:\nOriginal: {orig_preds}\nReloaded: {reloaded_preds}"
        )
