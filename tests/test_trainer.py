import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.hybrid_detector import build_model
from src.training.trainer import TwoPhaseTrainer, train_two_phase

def test_trainer_llrd_param_groups():
    model = build_model(use_fft=True, pretrained=False)
    dummy_x = torch.randn(4, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=2)

    trainer = TwoPhaseTrainer(model=model, train_loader=loader, val_loader=loader)
    param_groups = trainer._get_llrd_param_groups(lr_backbone=1e-5, lr_head=1e-4)

    assert len(param_groups) == 5, f"Expected 5 LLRD tiers, got {len(param_groups)}"
    assert param_groups[0]['lr'] == 1e-5 * 0.2   # stem + stages 0-1
    assert param_groups[1]['lr'] == 1e-5 * 0.5   # stage 2
    assert param_groups[2]['lr'] == 1e-5 * 1.0   # stage 3
    assert param_groups[3]['lr'] == 1e-5 * 1.0   # fusion/freq layers (same tier as stage 3)
    assert param_groups[4]['lr'] == 1e-4          # classifier head

def test_trainer_single_step_execution(tmp_path):
    model = build_model(use_fft=True, pretrained=False)
    dummy_x = torch.randn(4, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=2)

    config = {
        "training": {
            "epochs_phase1": 1,
            "epochs_phase2": 1,
            "lr_phase1": 1e-3,
            "lr_backbone": 1e-5,
            "lr_head": 1e-4
        }
    }

    trainer = TwoPhaseTrainer(model=model, train_loader=loader, val_loader=loader, config=config)
    checkpoint_data, opt_thresh, metrics = trainer.train()

    assert "state_dict" in checkpoint_data
    assert "optimal_threshold" in checkpoint_data
    assert 0.0 <= opt_thresh <= 1.0
    assert "val_auc" in metrics
