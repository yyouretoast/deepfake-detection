"""
Evaluation script for Dual-Stream Deepfake Detector on held-out test split.
Includes valid_flags corrupt image filtering, squeeze(-1) shape safety,
optimal threshold search, Log-Temperature Calibration (L-BFGS), ECE,
and saves a calibrated checkpoint contract for app.py.
"""
import os
import sys

# Ensure repository root is on sys.path for standalone subprocess execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from src.models.hybrid_detector import HybridDeepfakeDetector
from scripts.train_dual_stream_ddp import KaggleFastDataset, find_dataset_root, dedupe_split

def compute_ece(probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy = np.mean(targets[in_bin])
            avg_confidence = np.mean(probs[in_bin])
            ece += np.abs(accuracy - avg_confidence) * prop_in_bin
    return ece

def fit_temperature_log(val_logits, val_targets):
    """
    Fits log_temperature to guarantee strictly positive T = exp(log_T) > 0
    at every L-BFGS optimization step without transient sign flips.
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    targets_t = torch.tensor(val_targets, dtype=torch.float32)
    
    log_temperature = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1.0 initial
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        T = torch.exp(log_temperature)
        loss = criterion(logits_t / T, targets_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    optimal_T = torch.exp(log_temperature).item()
    return max(0.1, optimal_T)

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
    val_targets = np.array(val_targets)
    val_preds_uncalibrated = 1.0 / (1.0 + np.exp(-val_logits))

    best_thresh = 0.5
    best_val_f1 = 0.0
    for thresh in np.arange(0.01, 0.95, 0.01):
        f1 = f1_score(val_targets, (val_preds_uncalibrated > thresh).astype(int))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    optimal_temp = fit_temperature_log(val_logits, val_targets)
    val_preds_calibrated = 1.0 / (1.0 + np.exp(-(val_logits / optimal_temp)))
    
    val_ece_before = compute_ece(val_preds_uncalibrated, val_targets)
    val_ece_after = compute_ece(val_preds_calibrated, val_targets)

    print(f"✅ Optimal Val Decision Threshold: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")
    print(f"🌡️  Optimal Temperature (T*): {optimal_temp:.4f}")
    print(f"📊 Validation ECE: {val_ece_before:.4f} (Raw) → {val_ece_after:.4f} (Calibrated)")

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
    test_targets = np.array(test_targets)

    test_preds_raw = 1.0 / (1.0 + np.exp(-test_logits))
    test_preds_cal = 1.0 / (1.0 + np.exp(-(test_logits / optimal_temp)))

    test_auc = roc_auc_score(test_targets, test_preds_raw)
    test_binary = (test_preds_raw > best_thresh).astype(int)
    test_f1 = f1_score(test_targets, test_binary)
    test_prec = precision_score(test_targets, test_binary)
    test_rec = recall_score(test_targets, test_binary)

    test_ece_before = compute_ece(test_preds_raw, test_targets)
    test_ece_after = compute_ece(test_preds_cal, test_targets)

    print("\n" + "="*50)
    print("🏆 FINAL HELD-OUT TEST SET RESULTS:")
    print(f"  ├─ Test AUC:             {test_auc:.4f}")
    print(f"  ├─ Test F1-Score:        {test_f1:.4f}")
    print(f"  ├─ Precision:            {test_prec:.4f}")
    print(f"  ├─ Recall:               {test_rec:.4f}")
    print(f"  ├─ Optimal Threshold:    {best_thresh:.2f}")
    print(f"  ├─ Temperature (T*):     {optimal_temp:.4f}")
    print(f"  └─ Test ECE:             {test_ece_before:.4f} (Raw) → {test_ece_after:.4f} (Calibrated)")
    print("="*50)
    print("\nClassification Report:\n", classification_report(test_targets, test_binary, target_names=['Real', 'Fake']))

    calibrated_ckpt_path = '/kaggle/working/dual_stream_calibrated.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimal_threshold': float(best_thresh),
        'temperature': float(optimal_temp),
    }, calibrated_ckpt_path)
    print(f"\n💾 Saved Calibrated Model Checkpoint contract to {calibrated_ckpt_path}")

if __name__ == '__main__':
    evaluate()
