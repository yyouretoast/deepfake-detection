"""Evaluation and recalibration script for Dual-Stream Deepfake Detector on sanitized splits."""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier
from src.dataset.loader import dedupe_split
from src.dataset.resolver import find_dataset_root, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import compute_ece, find_optimal_threshold, fit_temperature_log
from src.utils.checkpoint import compute_dual_thresholds, load_detector_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_eer(targets: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and operating threshold."""
    fpr, tpr, thresholds = roc_curve(targets, probs, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_thresh


def run_video_bootstrap(
    sample_paths: list[str],
    targets: np.ndarray,
    probs: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute 95% paired bootstrap confidence intervals resampled over video sequences."""
    # Group samples by video sequence folder
    video_map: dict[str, list[int]] = {}
    for i, p in enumerate(sample_paths):
        vid_dir = os.path.normpath(p).replace("\\", "/").split("/")[-2]
        video_map.setdefault(vid_dir, []).append(i)

    unique_vids = list(video_map.keys())
    vid_labels = np.array([targets[video_map[v][0]] for v in unique_vids])

    real_vids = [unique_vids[i] for i in range(len(unique_vids)) if vid_labels[i] == 0]
    fake_vids = [unique_vids[i] for i in range(len(unique_vids)) if vid_labels[i] == 1]

    rng = np.random.default_rng(seed)
    boot_aucs = []
    boot_eers = []

    logger.info("Computing %d video-level bootstrap iterations (%d real vids, %d fake vids)...", n_iterations, len(real_vids), len(fake_vids))

    for _ in range(n_iterations):
        sample_real = rng.choice(real_vids, size=len(real_vids), replace=True)
        sample_fake = rng.choice(fake_vids, size=len(fake_vids), replace=True)

        boot_idx = []
        for v in sample_real:
            boot_idx.extend(video_map[v])
        for v in sample_fake:
            boot_idx.extend(video_map[v])

        b_targets = targets[boot_idx]
        b_probs = probs[boot_idx]

        if len(np.unique(b_targets)) < 2:
            continue

        try:
            b_auc = roc_auc_score(b_targets, b_probs)
            b_eer, _ = compute_eer(b_targets, b_probs)
            boot_aucs.append(b_auc)
            boot_eers.append(b_eer)
        except ValueError:
            continue

    boot_aucs = np.array(boot_aucs)
    boot_eers = np.array(boot_eers)

    auc_mean = float(np.mean(boot_aucs))
    auc_ci_lower = float(np.percentile(boot_aucs, 2.5))
    auc_ci_upper = float(np.percentile(boot_aucs, 97.5))

    eer_mean = float(np.mean(boot_eers)) * 100.0
    eer_ci_lower = float(np.percentile(boot_eers, 2.5)) * 100.0
    eer_ci_upper = float(np.percentile(boot_eers, 97.5)) * 100.0

    return {
        "auc_mean": auc_mean,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "eer_mean": eer_mean,
        "eer_ci_lower": eer_ci_lower,
        "eer_ci_upper": eer_ci_upper,
    }


def evaluate(
    data_dir: str | None = None,
    weights_path: str | None = None,
    output_json: str | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    n_bootstrap: int = 1000,
) -> dict:
    data_root = find_dataset_root(data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    logger.info("Loading sanitized splits from: %s", splits_path)
    with open(splits_path, "r") as f:
        splits = json.load(f)

    val_samples = dedupe_split(splits["val"])
    test_samples = dedupe_split(splits["test"])

    logger.info("Validation samples: %d", len(val_samples))
    logger.info("Test samples: %d", len(test_samples))

    val_loader = DataLoader(
        FaceCropDataset(val_samples, data_root, is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        FaceCropDataset(test_samples, data_root, is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)
    model, _, _ = load_detector_checkpoint(weights_path=weights_path, device=device, data_root=data_root)

    evaluator = ModelEvaluator(model, device=device)

    # 1. Validation Split Inference & Calibration Refitting
    logger.info("\n--- Running Validation Inference for Calibration Refitting ---")
    val_logits, val_targets, val_valid = evaluator.predict_loader(val_loader)
    v_mask = val_valid > 0.0
    val_logits = val_logits[v_mask]
    val_targets = val_targets[v_mask].astype(int)

    optimal_temp = fit_temperature_log(val_logits, val_targets)
    val_probs_cal = 1.0 / (1.0 + np.exp(-(val_logits / optimal_temp)))
    val_ece = compute_ece(val_probs_cal, val_targets)

    best_thresh, best_bal_acc = find_optimal_threshold(
        val_targets, val_probs_cal, criterion="balanced_accuracy", n_thresholds=101
    )
    youden_j = 2.0 * best_bal_acc - 1.0
    tau_real, tau_fake = compute_dual_thresholds(val_probs_cal, val_targets, min_precision=0.98)

    logger.info("Fitted Optimal Temperature T* = %.4f", optimal_temp)
    logger.info("Optimal Decision Threshold tau* = %.4f (Val Balanced Acc = %.4f, Youden J = %.4f)", best_thresh, best_bal_acc, youden_j)
    logger.info("Bayesian Cutoffs: tau_real (Authentic) <= %.4f | tau_fake (Synthetic) >= %.4f", tau_real, tau_fake)

    # 2. Test Split Inference
    logger.info("\n--- Running Test Split Inference ---")
    test_logits, test_targets, test_valid = evaluator.predict_loader(test_loader)
    t_mask = test_valid > 0.0
    test_logits = test_logits[t_mask]
    test_targets = test_targets[t_mask].astype(int)
    test_paths = [test_samples[i][0] for i in range(len(test_samples)) if t_mask[i]]

    test_probs_uncal = 1.0 / (1.0 + np.exp(-test_logits))
    test_probs_cal = 1.0 / (1.0 + np.exp(-(test_logits / optimal_temp)))
    test_preds_opt = (test_probs_cal >= best_thresh).astype(int)

    # Overall Metrics
    overall_auc = roc_auc_score(test_targets, test_probs_cal)
    overall_bal_acc = balanced_accuracy_score(test_targets, test_preds_opt)
    overall_macro_f1 = f1_score(test_targets, test_preds_opt, average="macro", zero_division=0)
    overall_fake_f1 = f1_score(test_targets, test_preds_opt, zero_division=0)
    overall_precision = precision_score(test_targets, test_preds_opt, zero_division=0)
    overall_recall = recall_score(test_targets, test_preds_opt, zero_division=0)
    overall_eer, _ = compute_eer(test_targets, test_probs_cal)
    overall_ece_cal = compute_ece(test_probs_cal, test_targets)
    overall_ece_uncal = compute_ece(test_probs_uncal, test_targets)

    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds_opt)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("\n==================================================================")
    logger.info("         SANITIZED HELD-OUT TEST COHORT BENCHMARK RESULTS         ")
    logger.info("==================================================================")
    logger.info("  Total Test Samples:    %d", len(test_targets))
    logger.info("  Total Real Samples:    %d (True Negatives: %d, False Positives: %d)", tn + fp, tn, fp)
    logger.info("  Total Fake Samples:    %d (True Positives: %d, False Negatives: %d)", fn + tp, tp, fn)
    logger.info("  ROC AUC:               %.4f", overall_auc)
    logger.info("  Balanced Accuracy:     %.4f", overall_bal_acc)
    logger.info("  Macro F1:              %.4f", overall_macro_f1)
    logger.info("  Fake F1:               %.4f", overall_fake_f1)
    logger.info("  Precision:             %.4f", overall_precision)
    logger.info("  Recall:                %.4f", overall_recall)
    logger.info("  Specificity:           %.4f", spec)
    logger.info("  Equal Error Rate (EER):%.2f%%", overall_eer * 100.0)
    logger.info("  Calibrated ECE:        %.4f (Uncalibrated: %.4f)", overall_ece_cal, overall_ece_uncal)

    # 3. Per-Domain Subdomain Breakdown
    logger.info("\n------------------------------------------------------------------")
    logger.info("                FINE-GRAINED SUBDOMAIN PERFORMANCE                ")
    logger.info("------------------------------------------------------------------")

    # Classify each test path into its domain
    subdomain_results = {}
    ffpp_real_mask = np.array([("real" in p.lower() and "id" not in p.lower()) for p in test_paths])
    celeb_real_mask = np.array([("real" in p.lower() and "id" in p.lower()) for p in test_paths])

    subdomains = {
        "deepfakes": ("FF++ Deepfakes", lambda p: DomainClassifier.matches_holdout(p, "deepfakes")),
        "face2face": ("FF++ Face2Face", lambda p: DomainClassifier.matches_holdout(p, "face2face")),
        "faceswap": ("FF++ FaceSwap", lambda p: DomainClassifier.matches_holdout(p, "faceswap")),
        "neuraltextures": ("FF++ NeuralTextures", lambda p: DomainClassifier.matches_holdout(p, "neuraltextures")),
        "dfd": ("Google DFD Challenge", lambda p: DomainClassifier.matches_holdout(p, "dfd")),
        "celeb": ("Celeb-DF v2 (In-Domain)", lambda p: DomainClassifier.matches_holdout(p, "celeb")),
    }

    for key, (display_name, matcher) in subdomains.items():
        fake_mask = np.array([matcher(p) for p in test_paths])
        n_fakes = int(np.sum(fake_mask))
        if n_fakes == 0:
            continue

        # Matched reals
        if key == "celeb":
            eval_mask = fake_mask | celeb_real_mask
        else:
            eval_mask = fake_mask | ffpp_real_mask

        sub_targets = test_targets[eval_mask]
        sub_probs = test_probs_cal[eval_mask]
        sub_preds = test_preds_opt[eval_mask]

        sub_auc = roc_auc_score(sub_targets, sub_probs) if len(np.unique(sub_targets)) > 1 else 0.0
        sub_f1 = f1_score(sub_targets, sub_preds, zero_division=0)
        sub_prec = precision_score(sub_targets, sub_preds, zero_division=0)
        sub_rec = recall_score(sub_targets, sub_preds, zero_division=0)

        subdomain_results[key] = {
            "name": display_name,
            "fakes": n_fakes,
            "auc": float(sub_auc),
            "f1": float(sub_f1),
            "precision": float(sub_prec),
            "recall": float(sub_rec),
        }
        logger.info("  %-25s | Fakes: %5d | AUC: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f", display_name, n_fakes, sub_auc, sub_f1, sub_prec, sub_rec)

    # 4. Bootstrap Confidence Intervals
    bootstrap_ci = run_video_bootstrap(
        test_paths, test_targets, test_probs_cal, n_iterations=n_bootstrap, seed=42
    )
    logger.info("\n------------------------------------------------------------------")
    logger.info("           1,000-ITERATION VIDEO-LEVEL BOOTSTRAP 95%% CIs          ")
    logger.info("------------------------------------------------------------------")
    logger.info("  ROC AUC: %.4f (95%% CI: [%.4f, %.4f])", overall_auc, bootstrap_ci["auc_ci_lower"], bootstrap_ci["auc_ci_upper"])
    logger.info("  EER:     %.2f%% (95%% CI: [%.2f%%, %.2f%%])", overall_eer * 100.0, bootstrap_ci["eer_ci_lower"], bootstrap_ci["eer_ci_upper"])

    output_payload = {
        "validation_calibration": {
            "temperature": float(optimal_temp),
            "threshold": float(best_thresh),
            "youden_j": float(youden_j),
            "tau_real": float(tau_real),
            "tau_fake": float(tau_fake),
            "val_ece": float(val_ece),
            "val_balanced_acc": float(best_bal_acc),
        },
        "overall_test_metrics": {
            "n_samples": len(test_targets),
            "n_reals": int(tn + fp),
            "n_fakes": int(fn + tp),
            "roc_auc": float(overall_auc),
            "balanced_accuracy": float(overall_bal_acc),
            "macro_f1": float(overall_macro_f1),
            "fake_f1": float(overall_fake_f1),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "specificity": float(spec),
            "eer": float(overall_eer * 100.0),
            "ece_calibrated": float(overall_ece_cal),
            "ece_uncalibrated": float(overall_ece_uncal),
            "bootstrap_ci": bootstrap_ci,
        },
        "subdomain_breakdown": subdomain_results,
    }

    out_file = output_json or os.path.join(REPO_ROOT, "results", "sanitized_benchmark_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("\nSaved benchmark results payload to: %s", out_file)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dual-Stream Deepfake Detector on Sanitized Test Set")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--weights_path", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        weights_path=args.weights_path,
        output_json=args.output_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
