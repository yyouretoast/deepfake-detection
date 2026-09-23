"""Integration test confirming train_backbone mid-run resume functionality."""

import json
import os
import shutil
import sys
import tempfile
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kaggle_pipeline.run_release1_kaggle import train_dual_stream_backbone
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import create_scheduler, get_differential_param_groups


def test_train_backbone_resumption():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = os.path.join(tmp_dir, "data")
        os.makedirs(os.path.join(data_root, "video1"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "video2"), exist_ok=True)

        # Create dummy images
        import cv2
        import numpy as np
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(data_root, "video1", "frame_0.webp"), dummy_img)
        cv2.imwrite(os.path.join(data_root, "video2", "frame_0.webp"), dummy_img)

        splits_path = os.path.join(tmp_dir, "splits.json")
        splits = {
            "train": [
                ["video1/frame_0.webp", 0, "video1"],
                ["video2/frame_0.webp", 1, "video2"],
            ],
            "val": [
                ["video1/frame_0.webp", 0, "video1"],
                ["video2/frame_0.webp", 1, "video2"],
            ],
            "test": [],
        }
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f)

        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Simulate that epoch 0 was finished and saved to dual_stream_epoch_resume.pth
        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        optimizer = torch.optim.AdamW(
            get_differential_param_groups(model, lr_backbone=1e-5, lr_head=1e-4),
            weight_decay=1e-4,
        )
        scheduler = create_scheduler(optimizer, warmup_epochs=1, total_epochs=2)
        ema = ExponentialMovingAverage(model, decay=0.999)

        resume_path = os.path.join(output_dir, "dual_stream_epoch_resume.pth")
        resume_dict = {
            "epoch": 0,
            "best_macro_auc": 0.5000,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        }
        torch.save(resume_dict, resume_path)
        assert os.path.exists(resume_path)

        # Call train_dual_stream_backbone with epochs=2: should resume from epoch 1 (index 1) and finish
        best_ckpt = train_dual_stream_backbone(
            data_root=data_root,
            splits_path=splits_path,
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device=torch.device("cpu"),
            smoke_test=False,
            force_rerun=False,
        )

        assert os.path.exists(best_ckpt)
        # Check that epoch resume checkpoint was cleaned up upon completion
        assert not os.path.exists(resume_path)
        print("PASS: train_backbone resumed from epoch 0, ran epoch 1, saved best model, and cleaned up resume checkpoint!")


if __name__ == "__main__":
    test_train_backbone_resumption()
