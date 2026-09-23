"""Test checkpoint resumption for Release 1 pipeline."""

import os
import shutil
import sys
import tempfile
import torch
from torch import nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import create_scheduler, get_differential_param_groups


def test_epoch_checkpoint_save_and_restore():
    with tempfile.TemporaryDirectory() as tmp_dir:
        device = torch.device("cpu")
        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        optimizer = torch.optim.AdamW(
            get_differential_param_groups(model, lr_backbone=1e-5, lr_head=1e-4),
            weight_decay=1e-4,
        )
        scheduler = create_scheduler(optimizer, warmup_epochs=1, total_epochs=5)
        ema = ExponentialMovingAverage(model, decay=0.999)

        # Modify parameter to verify restore
        first_param = next(p for p in model.parameters() if p.requires_grad)
        first_param.data.fill_(1.234)
        ema.update(model)

        resume_path = os.path.join(tmp_dir, "dual_stream_epoch_resume.pth")
        resume_dict = {
            "epoch": 2,
            "best_macro_auc": 0.8523,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        }
        torch.save(resume_dict, resume_path)
        assert os.path.exists(resume_path)

        # Now simulate resuming on a fresh model
        new_model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        new_optimizer = torch.optim.AdamW(
            get_differential_param_groups(new_model, lr_backbone=1e-5, lr_head=1e-4),
            weight_decay=1e-4,
        )
        new_scheduler = create_scheduler(new_optimizer, warmup_epochs=1, total_epochs=5)
        new_ema = ExponentialMovingAverage(new_model, decay=0.999)

        ckpt = torch.load(resume_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_macro_auc = ckpt.get("best_macro_auc", 0.0)
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        new_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        new_ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

        assert start_epoch == 3
        assert abs(best_macro_auc - 0.8523) < 1e-6
        new_first_param = next(p for p in new_model.parameters() if p.requires_grad)
        assert torch.allclose(new_first_param, first_param)
        print("PASS: Checkpoint save and restore verified with exact parameter fidelity!")


if __name__ == "__main__":
    test_epoch_checkpoint_save_and_restore()
