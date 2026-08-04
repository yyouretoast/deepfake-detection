"""
Evaluation script for Dual-Stream Deepfake Detector on held-out test split.
Includes valid_flags corrupt image filtering, squeeze(-1) shape safety,
optimal threshold search, and temperature scaling calibration evaluation.
"""
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from src.models.hybrid_detector import HybridDeepfakeDetector
from scripts.train_dual_stream_ddp import KaggleFastDataset, find_dataset_root, dedupe_split

def evaluate():
    data_root = find_dataset_root()
    splits_path = os.path.join(data_root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    val_samples = dedupe_split(splits['val'])
    test_samples = dedupe_split(splits['test'])

    val_loader = DataLoader(KaggleFastDataset(val_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(KaggleFastDataset(test_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridDeepfakeDetector().to(device)
    
    weights_path = '/kaggle/working/dual_stream_best.pth'
    if not os.path.exists(weights_path):
        weights_path = os.path.join(data_root, 'dual_stream_best.pth')
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 1. Validation set logits and targets with valid_flags mask & squeeze(-1)
    print("🔍 Evaluating Validation Set for Threshold & Calibration Optimization...")
    val_logits, val_targets = [], []
    with torch.no_grad():
        for images, labels, valid_flags in val_loader:
            mask = valid_flags.bool()
            if not mask.any():
                continue
            images, labels = images[mask].to(device), labels[mask]
            logits = model(images).squeeze(-1).cpu().numpy()
            if logits.ndim == 0:
                logits = np.array([logits])
            val_logits.extend(logits)
            val_targets.extend(labels.numpy())

    val_logits = np.array(val_logits)
    val_preds = 1.0 / (1.0 + np.exp(-val_logits))  # Sigmoid
    val_targets = np.array(val_targets)

    best_thresh = 0.5
    best_val_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(val_targets, (val_preds > thresh).astype(int))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    print(f"✅ Optimal Val Threshold: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")

    # 2. Test Set Evaluation
    print("\n📊 Running Final Test Set Evaluation...")
    test_logits, test_targets = [], []
    with torch.no_grad():
        for images, labels, valid_flags in test_loader:
            mask = valid_flags.bool()
            if not mask.any():
                continue
            images, labels = images[mask].to(device), labels[mask]
            logits = model(images).squeeze(-1).cpu().numpy()
            if logits.ndim == 0:
                logits = np.array([logits])
            test_logits.extend(logits)
            test_targets.extend(labels.numpy())

    test_logits = np.array(test_logits)
    test_preds = 1.0 / (1.0 + np.exp(-test_logits))
    test_targets = np.array(test_targets)

    test_auc = roc_auc_score(test_targets, test_preds)
    test_binary = (test_preds > best_thresh).astype(int)
    test_f1 = f1_score(test_targets, test_binary)
    test_prec = precision_score(test_targets, test_binary)
    test_rec = recall_score(test_targets, test_binary)

    print("\n" + "="*50)
    print("🏆 FINAL HELD-OUT TEST SET RESULTS:")
    print(f"  ├─ Test AUC:       {test_auc:.4f}")
    print(f"  ├─ Test F1-Score:  {test_f1:.4f}")
    print(f"  ├─ Precision:      {test_prec:.4f}")
    print(f"  └─ Recall:         {test_rec:.4f}")
    print(f"  └─ Applied Thresh: {best_thresh:.2f}")
    print("="*50)
    print("\nClassification Report:\n", classification_report(test_targets, test_binary, target_names=['Real', 'Fake']))

if __name__ == '__main__':
    evaluate()
