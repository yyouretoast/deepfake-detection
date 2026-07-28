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

def test_trainer_multi_epoch_integration(tmp_path):
    """
    Full end-to-end multi-epoch integration test verifying train-eval-save-load cycle.
    """
    model = build_model(use_fft=True, pretrained=False)
    dummy_x = torch.randn(8, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4)

    config = {
        "training": {
            "epochs_phase1": 2,
            "epochs_phase2": 2,
            "lr_phase1": 1e-3,
            "lr_backbone": 1e-5,
            "lr_head": 1e-4,
            "use_amp": False,
            "patience": 5
        }
    }

    trainer = TwoPhaseTrainer(model=model, train_loader=loader, val_loader=loader, config=config)
    
    # 1. Train multi-epoch
    checkpoint_data, opt_thresh, metrics = trainer.train()

    # 2. Verify metrics structure
    assert "state_dict" in checkpoint_data
    assert "optimal_threshold" in checkpoint_data
    assert "val_auc" in metrics
    assert "macro_f1" in metrics
    assert "eer" in metrics
    assert "val_acc" in metrics
    assert 0.0 <= opt_thresh <= 1.0

    # 3. Save checkpoint to disk
    save_path = os.path.join(tmp_path, "deepfake_convnext_v2.pth")
    torch.save(checkpoint_data, save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0

    # 4. Load checkpoint & verify inference on reloaded model
    loaded_ckpt = torch.load(save_path, weights_only=False)
    eval_model = build_model(use_fft=True, pretrained=False)
    eval_model.load_state_dict(loaded_ckpt["state_dict"])
    eval_model.eval()

    with torch.no_grad():
        out = eval_model(dummy_x[:2])
        assert out.shape == (2,)
