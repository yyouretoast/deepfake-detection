"""Evaluation script for Dual-Stream Deepfake Detector on held-out test split."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from scripts.train_dual_stream_ddp import KaggleFastDataset, dedupe_split, find_dataset_root
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import compute_ece, fit_temperature_log


def evaluate() -> None:
    data_root = find_dataset_root()
    splits_path = os.path.join(data_root, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    val_samples = dedupe_split(splits["val"])
    test_samples = dedupe_split(splits["test"])

    val_loader = DataLoader(KaggleFastDataset(val_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(KaggleFastDataset(test_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDeepfakeDetector().to(device)

    weights_path = "/kaggle/working/dual_stream_best.pth"
    if not os.path.exists(weights_path):
        weights_path = os.path.join(data_root, "dual_stream_best.pth")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
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

    val_logits_arr = np.array(val_logits)
    val_targets_arr = np.array(val_targets)
    val_preds_uncalibrated = 1.0 / (1.0 + np.exp(-val_logits_arr))

    best_thresh = 0.5
    best_val_f1 = 0.0
    for thresh in np.arange(0.01, 0.95, 0.01):
        f1 = f1_score(val_targets_arr, (val_preds_uncalibrated > thresh).astype(int))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    optimal_temp = fit_temperature_log(val_logits_arr, val_targets_arr)
    val_preds_calibrated = 1.0 / (1.0 + np.exp(-(val_logits_arr / optimal_temp)))

    val_ece_before = compute_ece(val_preds_uncalibrated, val_targets_arr)
    val_ece_after = compute_ece(val_preds_calibrated, val_targets_arr)

    print(f"Optimal Validation Decision Threshold: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")
    print(f"Optimal Temperature (T*): {optimal_temp:.4f}")
    print(f"Validation ECE: {val_ece_before:.4f} (Raw) -> {val_ece_after:.4f} (Calibrated)")

    print("\nRunning Final Test Set Evaluation...")
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

    test_logits_arr = np.array(test_logits)
    test_targets_arr = np.array(test_targets)

    test_preds_raw = 1.0 / (1.0 + np.exp(-test_logits_arr))
    test_preds_cal = 1.0 / (1.0 + np.exp(-(test_logits_arr / optimal_temp)))

    test_auc = roc_auc_score(test_targets_arr, test_preds_raw)
    test_binary = (test_preds_raw > best_thresh).astype(int)
    test_f1 = f1_score(test_targets_arr, test_binary)
    test_prec = precision_score(test_targets_arr, test_binary)
    test_rec = recall_score(test_targets_arr, test_binary)

    test_ece_before = compute_ece(test_preds_raw, test_targets_arr)
    test_ece_after = compute_ece(test_preds_cal, test_targets_arr)

    print("\nFINAL HELD-OUT TEST SET RESULTS:")
    print(f"  Test AUC:          {test_auc:.4f}")
    print(f"  Test F1-Score:     {test_f1:.4f}")
    print(f"  Precision:         {test_prec:.4f}")
    print(f"  Recall:            {test_rec:.4f}")
    print(f"  Optimal Threshold: {best_thresh:.2f}")
    print(f"  Temperature (T*):  {optimal_temp:.4f}")
    print(f"  Test ECE:          {test_ece_before:.4f} (Raw) -> {test_ece_after:.4f} (Calibrated)")
    print("\nClassification Report:\n", classification_report(test_targets_arr, test_binary, target_names=["Real", "Fake"]))

    calibrated_ckpt_path = "/kaggle/working/dual_stream_calibrated.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimal_threshold": float(best_thresh),
            "temperature": float(optimal_temp),
        },
        calibrated_ckpt_path,
    )
    print(f"\nSaved Calibrated Model Checkpoint contract to {calibrated_ckpt_path}")


if __name__ == "__main__":
    evaluate()
