"""Evaluation script for Dual-Stream Deepfake Detector on held-out test split."""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import dedupe_split
from src.dataset.resolver import find_dataset_root, find_weights_path, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import compute_ece, fit_temperature_log
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict, compute_dual_thresholds


def evaluate(
    data_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
    save_calibrated: Optional[str] = None,
) -> None:
    data_root = find_dataset_root(data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    print(f"Loading evaluation splits from: {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    val_samples = dedupe_split(splits["val"])
    test_samples = dedupe_split(splits["test"])

    val_loader = DataLoader(
        FaceCropDataset(val_samples, data_root, is_train=False),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        FaceCropDataset(test_samples, data_root, is_train=False),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDeepfakeDetector().to(device)

    resolved_weights = find_weights_path(weights_path, data_root)
    print(f"Loading weights from: {resolved_weights}")

    checkpoint = torch.load(resolved_weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.eval()

    evaluator = ModelEvaluator(model, device=device)

    print("\n--- Running Validation Split Inference ---")
    val_logits, val_targets, val_valid = evaluator.predict_loader(val_loader)
    mask = val_valid > 0.0
    val_logits = val_logits[mask]
    val_targets = val_targets[mask]

    print("\n--- Fitting Temperature Scaling on Validation Split ---")
    optimal_temp = fit_temperature_log(val_logits, val_targets)
    print(f"Optimal Calibration Temperature T* = {optimal_temp:.4f}")

    val_probs_cal = 1.0 / (1.0 + np.exp(-(val_logits / optimal_temp)))
    val_ece_uncal = compute_ece(1.0 / (1.0 + np.exp(-val_logits)), val_targets)
    val_ece_cal = compute_ece(val_probs_cal, val_targets)
    print(f"Validation ECE: {val_ece_uncal:.4f} (Uncalibrated) -> {val_ece_cal:.4f} (Calibrated)")

    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (val_probs_cal >= t).astype(int)
        score = f1_score(val_targets, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    print(f"Optimal Decision Threshold tau* = {best_thresh:.4f} (Val Macro F1 = {best_f1:.4f})")

    print("\n--- Running Held-out Test Split Inference ---")
    test_logits, test_targets, test_valid = evaluator.predict_loader(test_loader)
    mask_test = test_valid > 0.0
    test_logits = test_logits[mask_test]
    test_targets = test_targets[mask_test]

    test_probs_uncal = 1.0 / (1.0 + np.exp(-test_logits))
    test_probs_cal = 1.0 / (1.0 + np.exp(-(test_logits / optimal_temp)))

    test_auc = roc_auc_score(test_targets, test_probs_cal)
    test_ece_uncal = compute_ece(test_probs_uncal, test_targets)
    test_ece_cal = compute_ece(test_probs_cal, test_targets)

    test_preds_default = (test_probs_cal >= 0.5).astype(int)
    test_preds_opt = (test_probs_cal >= best_thresh).astype(int)

    print("\n=======================================================")
    print("           HELD-OUT TEST SET BENCHMARK RESULTS         ")
    print("=======================================================")
    print(f"  Test Samples Evaluated: {len(test_targets):,}")
    print(f"  Test ROC AUC:           {test_auc:.4f}")
    print(f"  Test ECE (Uncalibrated):{test_ece_uncal:.4f}")
    print(f"  Test ECE (Calibrated):  {test_ece_cal:.4f}")
    print("-------------------------------------------------------")
    print("  Classification Metrics at Default Threshold (0.50):")
    print(f"    Macro F1:   {f1_score(test_targets, test_preds_default, average='macro', zero_division=0):.4f}")
    print(f"    Fake F1:    {f1_score(test_targets, test_preds_default, zero_division=0):.4f}")
    print(f"    Precision:  {precision_score(test_targets, test_preds_default, zero_division=0):.4f}")
    print(f"    Recall:     {recall_score(test_targets, test_preds_default, zero_division=0):.4f}")
    print(f"  Classification Metrics at Optimal Threshold ({best_thresh:.2f}):")
    print(f"    Macro F1:   {f1_score(test_targets, test_preds_opt, average='macro', zero_division=0):.4f}")
    print(f"    Fake F1:    {f1_score(test_targets, test_preds_opt, zero_division=0):.4f}")
    print(f"    Precision:  {precision_score(test_targets, test_preds_opt, zero_division=0):.4f}")
    print(f"    Recall:     {recall_score(test_targets, test_preds_opt, zero_division=0):.4f}")
    print("=======================================================\n")
    print("Detailed Classification Report (Full Test Set, Optimal Threshold):")
    print(classification_report(test_targets, test_preds_opt, target_names=["Real", "Fake"], digits=4))

    real_indices = np.where(test_targets == 0)[0]
    fake_indices = np.where(test_targets == 1)[0]
    n_balanced = min(len(real_indices), len(fake_indices))
    if n_balanced > 0 and len(real_indices) != len(fake_indices):
        rng = np.random.default_rng(42)
        sampled_fake_idx = rng.choice(fake_indices, size=n_balanced, replace=False)
        balanced_idx = np.concatenate([real_indices[:n_balanced], sampled_fake_idx])
        b_targets = test_targets[balanced_idx]
        b_preds = test_preds_opt[balanced_idx]
        b_probs = test_probs_cal[balanced_idx]
        b_auc = roc_auc_score(b_targets, b_probs)
        print(f"Balanced 1:1 Benchmark Subset ({n_balanced:,} Real vs {n_balanced:,} Fake):")
        print(f"    Balanced ROC AUC: {b_auc:.4f}")
        print(classification_report(b_targets, b_preds, target_names=["Real", "Fake"], digits=4))

    tau_real, tau_fake = compute_dual_thresholds(val_probs_cal, val_targets, min_precision=0.98)
    print("\nDual Bayesian High-Precision Thresholds (>=98% Precision):")
    print(f"    tau_real (Authentic): <= {tau_real:.4f}")
    print(f"    tau_fake (Synthetic): >= {tau_fake:.4f}")
    print(f"    Inconclusive Ambiguity Band: ({tau_real:.4f}, {tau_fake:.4f})")

    calibrated_ckpt_path = save_calibrated or (
        "/kaggle/working/dual_stream_calibrated.pth"
        if os.path.exists("/kaggle/working")
        else "./dual_stream_calibrated.pth"
    )
    os.makedirs(os.path.dirname(os.path.abspath(calibrated_ckpt_path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimal_threshold": float(best_thresh),
            "temperature": float(optimal_temp),
            "tau_real": float(tau_real),
            "tau_fake": float(tau_fake),
        },
        calibrated_ckpt_path,
    )
    print(f"\nSaved Calibrated Model Checkpoint to: {calibrated_ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dual-Stream Deepfake Detector on Held-out Test Set")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json and dataset")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to dual_stream_best.pth")
    parser.add_argument("--save_calibrated", type=str, default=None, help="Path to save calibrated model weights")
    args = parser.parse_args()

    evaluate(data_dir=args.data_dir, weights_path=args.weights_path, save_calibrated=args.save_calibrated)


if __name__ == "__main__":
    main()
