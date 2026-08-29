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


import argparse
from typing import Any, Callable, Optional


def find_weights_path(custom_path: Optional[str] = None, data_root: Optional[str] = None) -> str:
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.getenv("BEST_MODEL_WEIGHTS_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "./models/dual_stream_best.pth",
        "models/dual_stream_best.pth",
        "/kaggle/working/repo/models/dual_stream_best.pth",
        "/kaggle/working/models/dual_stream_best.pth",
        "/kaggle/working/dual_stream_best.pth",
    ])
    if data_root:
        candidates.append(os.path.join(data_root, "dual_stream_best.pth"))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    if os.path.exists("/kaggle/working"):
        for root, dirs, files in os.walk("/kaggle/working"):
            if "dual_stream_best.pth" in files:
                return os.path.join(root, "dual_stream_best.pth")
    raise FileNotFoundError(f"Could not locate dual_stream_best.pth. Checked: {candidates}")


def evaluate(data_dir: Optional[str] = None, weights_path: Optional[str] = None) -> None:
    data_root = find_dataset_root(data_dir)
    splits_path = os.path.join(data_root, "splits.json")
    if os.path.exists("/kaggle/working/splits.json"):
        splits_path = "/kaggle/working/splits.json"
    elif os.path.exists("./splits.json"):
        splits_path = "./splits.json"

    print(f"Loading evaluation splits from: {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    val_samples = dedupe_split(splits["val"])
    test_samples = dedupe_split(splits["test"])

    val_loader = DataLoader(KaggleFastDataset(val_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(KaggleFastDataset(test_samples, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDeepfakeDetector().to(device)

    resolved_weights_path = find_weights_path(weights_path, data_root)
    print(f"Loading best model weights from: {resolved_weights_path}")
    model.load_state_dict(torch.load(resolved_weights_path, map_location=device, weights_only=False))
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

    # Fit temperature on the validation set first
    optimal_temp = fit_temperature_log(val_logits_arr, val_targets_arr)
    val_preds_calibrated = 1.0 / (1.0 + np.exp(-(val_logits_arr / optimal_temp)))

    val_ece_before = compute_ece(val_preds_uncalibrated, val_targets_arr)
    val_ece_after = compute_ece(val_preds_calibrated, val_targets_arr)

    # CRITICAL: tune threshold on CALIBRATED probabilities.
    # If threshold were tuned on raw (uncalibrated) preds but applied to calibrated preds,
    # the probability scale mismatch (T*=1.4788 squashes toward 0.5) would silently tank
    # Recall and F1 for all downstream evaluation scripts.
    best_thresh = 0.5
    best_val_f1 = 0.0
    for thresh in np.arange(0.01, 0.95, 0.01):
        f1 = f1_score(val_targets_arr, (val_preds_calibrated > thresh).astype(int))
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    print(f"Optimal Temperature (T*): {optimal_temp:.4f}")
    print(f"Optimal Validation Decision Threshold (tuned on calibrated probs): {best_thresh:.2f} (Val F1: {best_val_f1:.4f})")
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
    test_binary = (test_preds_cal > best_thresh).astype(int)
    test_f1 = f1_score(test_targets_arr, test_binary)
    test_prec = precision_score(test_targets_arr, test_binary)
    test_rec = recall_score(test_targets_arr, test_binary)

    def compute_bootstrap_ci(
        y_true: np.ndarray, y_score_or_pred: np.ndarray, metric_fn: Any, n_bootstraps: int = 1000, seed: int = 42
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        bootstrapped_scores = []
        n = len(y_true)
        for _ in range(n_bootstraps):
            indices = rng.choice(n, size=n, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = metric_fn(y_true[indices], y_score_or_pred[indices])
            bootstrapped_scores.append(score)
        if not bootstrapped_scores:
            return 0.0, 0.0
        sorted_scores = np.sort(bootstrapped_scores)
        lower = float(np.percentile(sorted_scores, 2.5))
        upper = float(np.percentile(sorted_scores, 97.5))
        return lower, upper

    auc_ci = compute_bootstrap_ci(test_targets_arr, test_preds_raw, roc_auc_score)
    f1_ci = compute_bootstrap_ci(test_targets_arr, test_binary, f1_score)
    prec_ci = compute_bootstrap_ci(test_targets_arr, test_binary, precision_score)
    rec_ci = compute_bootstrap_ci(test_targets_arr, test_binary, recall_score)

    test_ece_before = compute_ece(test_preds_raw, test_targets_arr)
    test_ece_after = compute_ece(test_preds_cal, test_targets_arr)

    print("\nFINAL HELD-OUT TEST SET RESULTS:")
    print(f"  Test AUC:          {test_auc:.4f} [95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}]")
    print(f"  Test F1-Score:     {test_f1:.4f} [95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
    print(f"  Precision:         {test_prec:.4f} [95% CI: {prec_ci[0]:.4f} - {prec_ci[1]:.4f}]")
    print(f"  Recall:            {test_rec:.4f} [95% CI: {rec_ci[0]:.4f} - {rec_ci[1]:.4f}]")
    print(f"  Optimal Threshold: {best_thresh:.2f}")
    print(f"  Temperature (T*):  {optimal_temp:.4f}")
    print(f"  Test ECE:          {test_ece_before:.4f} (Raw) -> {test_ece_after:.4f} (Calibrated)")
    print("\nClassification Report:\n", classification_report(test_targets_arr, test_binary, target_names=["Real", "Fake"]))

    os.makedirs("./models", exist_ok=True)
    calibrated_ckpt_path = "./models/dual_stream_calibrated.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimal_threshold": float(best_thresh),
            "temperature": float(optimal_temp),
        },
        calibrated_ckpt_path,
    )
    if os.path.exists("/kaggle/working"):
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimal_threshold": float(best_thresh),
                    "temperature": float(optimal_temp),
                },
                "/kaggle/working/dual_stream_calibrated.pth",
            )
        except OSError:
            pass
    print(f"\nSaved Calibrated Model Checkpoint contract to {calibrated_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dual-Stream Deepfake Detector")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json and dataset")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to dual_stream_best.pth")
    args = parser.parse_args()
    evaluate(data_dir=args.data_dir, weights_path=args.weights_path)
