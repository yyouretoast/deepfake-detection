"""Comprehensive Statistical Significance Testing and Domain Classifier Probe.

1. Statistical Significance of Dual-Stream vs Spatial Baseline:
   - DeLong's test for correlated ROC curves (Z-score, two-tailed p-value).
   - Paired Clustered Bootstrap (1,000 iterations) 95% CI on Delta AUC.
   - McNemar's test on discordant predictions (overall and fake-recall specific).
2. Domain/Source Classifier Probe:
   - Linear probe on frozen 512-d feature embeddings trained on validation set,
     evaluated on held-out test set to classify domain (FF++, Celeb-DF, DFD).
   - Quantifies the cross-dataset confound quantitatively.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time

from src.utils.fs import patch_pathlib_mkdir

patch_pathlib_mkdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mpl_cache"))
os.makedirs(".mpl_cache", exist_ok=True)

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import get_transforms
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("significance_probe")


# -------------------------------------------------------------------------
# DeLong Test Implementation for Correlated ROC Curves
# -------------------------------------------------------------------------
def compute_midrank(x):
    """Computes midranks of array x."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong method for structural components of AUC covariance.
    predictions_sorted_transposed: shape (k, n)
    label_1_count: number of positive samples (m)
    """
    k, n = predictions_sorted_transposed.shape
    m = label_1_count
    n_minus_m = n - m
    
    # Positive samples are first m, negative are remaining n - m
    positive_submatrix = predictions_sorted_transposed[:, :m]
    negative_submatrix = predictions_sorted_transposed[:, m:]
    
    # Compute midranks
    V_10 = np.empty((k, m), dtype=float)
    V_01 = np.empty((k, n_minus_m), dtype=float)
    
    for i in range(k):
        ranks = compute_midrank(predictions_sorted_transposed[i, :])
        V_10[i, :] = (ranks[:m] - compute_midrank(positive_submatrix[i, :])) / n_minus_m
        V_01[i, :] = 1.0 - (ranks[m:] - compute_midrank(negative_submatrix[i, :])) / m
        
    auc = np.mean(V_10, axis=1)
    
    # Covariance matrices
    S_10 = np.cov(V_10)
    S_01 = np.cov(V_01)
    S = S_10 / m + S_01 / n_minus_m
    return auc, S


def delong_roc_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray):
    """
    DeLong's test comparing two correlated ROC curves.
    Returns: auc1, auc2, diff, z_score, p_value
    """
    y_true = np.asarray(y_true).astype(bool)
    order = np.argsort(~y_true)  # positives first
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted))
    
    preds = np.vstack([y_pred1[order], y_pred2[order]])
    aucs, cov = fast_delong(preds, m)
    
    auc1, auc2 = float(aucs[0]), float(aucs[1])
    diff = auc1 - auc2
    sigma_diff = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]))
    
    if sigma_diff < 1e-12:
        z_score = 0.0
        p_val = 1.0
    else:
        z_score = diff / sigma_diff
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
        
    return auc1, auc2, diff, sigma_diff, float(z_score), float(p_val)


# -------------------------------------------------------------------------
# McNemar's Test Implementation
# -------------------------------------------------------------------------
def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray):
    """
    McNemar's test with continuity correction.
    pred1: Model 1 binary predictions
    pred2: Model 2 binary predictions
    """
    c1 = (pred1 == y_true)
    c2 = (pred2 == y_true)
    
    n_00 = int(np.sum((~c1) & (~c2)))
    n_01 = int(np.sum((~c2) & c1))  # model 2 wrong, model 1 right
    n_10 = int(np.sum(c2 & (~c1)))  # model 2 right, model 1 wrong
    n_11 = int(np.sum(c1 & c2))
    
    discordant = n_01 + n_10
    if discordant == 0:
        return {"n_01": 0, "n_10": 0, "chi2": 0.0, "p_value": 1.0}
        
    chi2 = float((abs(n_01 - n_10) - 1.0) ** 2 / discordant)
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return {
        "n_both_correct": n_11,
        "n_both_wrong": n_00,
        "n_model1_right_model2_wrong": n_01,
        "n_model2_right_model1_wrong": n_10,
        "discordant_total": discordant,
        "chi2": chi2,
        "p_value": p_val,
    }


def get_domain_label(path: str) -> int:
    norm = path.replace("\\", "/").lower()
    parts = norm.split("/")
    folder = parts[1] if len(parts) > 1 else parts[0]
    if "celeb" in norm or folder.startswith("id"):
        return 1  # Celeb-DF
    elif "__" in folder or (len(folder) == 5 and folder.isdigit()) or "dfd" in norm:
        return 2  # Google DFD
    else:
        return 0  # FaceForensics++


# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_root = "deepfake_crops_512"
    splits_path = os.path.join(data_root, "splits.json")
    dual_ckpt_path = "results/release1_run/dual_stream_calibrated.pth"
    spatial_ckpt_path = "results/release1_run/spatial_convnext_best.pth"
    output_dir = "results/release1_run"
    os.makedirs(output_dir, exist_ok=True)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_samples = splits["test"]
    val_samples = splits["val"]
    logger.info("Loaded splits: Test = %d, Val = %d", len(test_samples), len(val_samples))

    # Platt calibration constants
    scale_a = 0.27831112352514586
    bias_b = 0.4089141487626174
    tau_star = 0.2600

    _, eval_transform = get_transforms(img_size=256, hardened=False)

    # 1. Load Models
    logger.info("Loading Dual-Stream model from %s...", dual_ckpt_path)
    dual_model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    dual_state = torch.load(dual_ckpt_path, map_location=device, weights_only=False)
    dual_model.load_state_dict(clean_state_dict(dual_state.get("model_state_dict", dual_state)), strict=False)
    dual_model.to(device).eval()

    logger.info("Loading Spatial-Only model from %s...", spatial_ckpt_path)
    spatial_model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=False)
    spatial_state = torch.load(spatial_ckpt_path, map_location=device, weights_only=False)
    spatial_model.load_state_dict(clean_state_dict(spatial_state.get("model_state_dict", spatial_state)), strict=False)
    spatial_model.to(device).eval()

    # 2. Extract Test Predictions and Embeddings
    test_loader = DataLoader(
        FaceCropDataset(test_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    logger.info("Running paired inference across %d test samples...", len(test_samples))
    y_test = []
    video_ids = []
    domain_labels = []
    p_dual_list = []
    p_spatial_list = []
    feat_dual_test = []

    start_time = time.time()
    with torch.no_grad():
        for images, targets, _ in tqdm(test_loader, desc="Test Inference"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                # Dual-stream forward and feature extraction
                logits_dual = dual_model(images)
                feats_dual = dual_model.extract_features(images)
                # Spatial-only forward
                logits_spatial = spatial_model(images)

            z_dual = logits_dual.view(-1).cpu().numpy()
            z_spatial = logits_spatial.view(-1).cpu().numpy()
            
            # Calibrated dual-stream probability
            p_d = 1.0 / (1.0 + np.exp(-(scale_a * z_dual + bias_b)))
            # Standard sigmoid spatial probability
            p_s = 1.0 / (1.0 + np.exp(-z_spatial))

            p_dual_list.extend(p_d)
            p_spatial_list.extend(p_s)
            y_test.extend(targets.numpy().astype(int))
            feat_dual_test.append(feats_dual.cpu().numpy())

    inference_time = time.time() - start_time
    logger.info("Test inference completed in %.2f seconds (%.1f FPS)", inference_time, len(test_samples) / inference_time)

    y_test = np.array(y_test)
    p_dual = np.array(p_dual_list)
    p_spatial = np.array(p_spatial_list)
    feat_dual_test = np.vstack(feat_dual_test)

    # Extract video IDs and domain labels
    for s in test_samples:
        p = s[0].replace("\\", "/")
        parts = p.split("/")
        vid = parts[1] if len(parts) > 1 else parts[0]
        video_ids.append(f"{parts[0]}/{vid}")
        domain_labels.append(get_domain_label(s[0]))
    video_ids = np.array(video_ids)
    domain_labels = np.array(domain_labels)

    # Binary predictions
    pred_dual = (p_dual >= tau_star).astype(int)
    pred_spatial = (p_spatial >= 0.50).astype(int)

    # ---------------------------------------------------------------------
    # Test 1: DeLong Test for Correlated ROC Curves
    # ---------------------------------------------------------------------
    auc_d, auc_s, diff_auc, sigma_diff, z_delong, p_delong = delong_roc_test(y_test, p_dual, p_spatial)
    logger.info("--- DELONG ROC AUC TEST ---")
    logger.info("Dual-Stream AUC: %.4f | Spatial-Only AUC: %.4f | Diff: %+.4f", auc_d, auc_s, diff_auc)
    logger.info("DeLong Z-score = %.4f | p-value = %.4e", z_delong, p_delong)

    # ---------------------------------------------------------------------
    # Test 2: Paired Clustered Bootstrap (1,000 Iterations)
    # ---------------------------------------------------------------------
    logger.info("Running 1,000 paired clustered bootstrap iterations...")
    unique_vids = np.unique(video_ids)
    vid_to_idx = {}
    for idx, v in enumerate(video_ids):
        vid_to_idx.setdefault(v, []).append(idx)

    rng = np.random.RandomState(42)
    boot_diffs = []
    n_vids = len(unique_vids)
    for _ in range(1000):
        sampled_vids = rng.choice(unique_vids, size=n_vids, replace=True)
        idx_pool = []
        for sv in sampled_vids:
            idx_pool.extend(vid_to_idx[sv])
        y_b = y_test[idx_pool]
        if len(np.unique(y_b)) >= 2:
            auc_d_b = roc_auc_score(y_b, p_dual[idx_pool])
            auc_s_b = roc_auc_score(y_b, p_spatial[idx_pool])
            boot_diffs.append(auc_d_b - auc_s_b)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    logger.info("Paired Clustered Bootstrap 95%% CI on Delta AUC: [%.4f, %.4f]", ci_lower, ci_upper)

    # ---------------------------------------------------------------------
    # Test 3: McNemar's Test on Overall and Fake Recall
    # ---------------------------------------------------------------------
    mcnemar_overall = mcnemar_test(y_test, pred_dual, pred_spatial)
    logger.info("--- MCNEMAR TEST (OVERALL) ---")
    logger.info("Chi2 = %.4f | p-value = %.4e | Dual+ Spat-: %d | Dual- Spat+: %d",
                mcnemar_overall["chi2"], mcnemar_overall["p_value"],
                mcnemar_overall["n_model1_right_model2_wrong"], mcnemar_overall["n_model2_right_model1_wrong"])

    # McNemar on Fakes Only (Recall Discordance)
    fake_mask = (y_test == 1)
    mcnemar_fake = mcnemar_test(y_test[fake_mask], pred_dual[fake_mask], pred_spatial[fake_mask])
    rec_dual = recall_score(y_test, pred_dual)
    rec_spat = recall_score(y_test, pred_spatial)
    logger.info("--- MCNEMAR TEST (FAKES ONLY / RECALL) ---")
    logger.info("Dual Fake Recall: %.2f%% | Spatial Fake Recall: %.2f%% (+%.2f%%)",
                rec_dual * 100, rec_spat * 100, (rec_dual - rec_spat) * 100)
    logger.info("Chi2 = %.4f | p-value = %.4e | Dual detected / Spatial missed: %d | Spatial detected / Dual missed: %d",
                mcnemar_fake["chi2"], mcnemar_fake["p_value"],
                mcnemar_fake["n_model1_right_model2_wrong"], mcnemar_fake["n_model2_right_model1_wrong"])

    # ---------------------------------------------------------------------
    # Test 4: Source/Domain Classifier Sanity Check Probe
    # ---------------------------------------------------------------------
    logger.info("--- EXTRACTING VALIDATION EMBEDDINGS FOR DOMAIN PROBE ---")
    # Subsample 6,000 balanced validation crops to train probe
    rng_val = random.Random(42)
    val_subset = rng_val.sample(val_samples, min(6000, len(val_samples)))
    val_loader = DataLoader(
        FaceCropDataset(val_subset, data_root, is_train=False, transform=eval_transform),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    feat_val = []
    domain_val = []
    with torch.no_grad():
        for images, _, _ in tqdm(val_loader, desc="Val Embeddings"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                f_d = dual_model.extract_features(images)
            feat_val.append(f_d.cpu().numpy())

    feat_val = np.vstack(feat_val)
    for s in val_subset:
        domain_val.append(get_domain_label(s[0]))
    domain_val = np.array(domain_val)

    logger.info("Training Domain Classifier Probe (Logistic Regression) on Val embeddings...")
    probe = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, random_state=42)
    probe.fit(feat_val, domain_val)

    # Evaluate probe on held-out test embeddings
    domain_preds = probe.predict(feat_dual_test)
    domain_acc = accuracy_score(domain_labels, domain_preds)
    domain_bacc = balanced_accuracy_score(domain_labels, domain_preds)
    domain_cm = confusion_matrix(domain_labels, domain_preds).tolist()
    domain_report = classification_report(domain_labels, domain_preds, target_names=["FF++", "Celeb-DF", "Google DFD"], output_dict=True)

    logger.info("--- DOMAIN PROBE EVALUATION ON TEST EMBEDDINGS ---")
    logger.info("Domain Classification Accuracy: %.2f%% (Balanced Acc: %.2f%%)", domain_acc * 100, domain_bacc * 100)
    logger.info("Per-Domain Precision/Recall: FF++: P=%.2f%%, R=%.2f%% | Celeb: P=%.2f%%, R=%.2f%% | DFD: P=%.2f%%, R=%.2f%%",
                domain_report["FF++"]["precision"] * 100, domain_report["FF++"]["recall"] * 100,
                domain_report["Celeb-DF"]["precision"] * 100, domain_report["Celeb-DF"]["recall"] * 100,
                domain_report["Google DFD"]["precision"] * 100, domain_report["Google DFD"]["recall"] * 100)

    # ---------------------------------------------------------------------
    # Save Structured Results
    # ---------------------------------------------------------------------
    out_results = {
        "statistical_significance": {
            "delong_test": {
                "auc_dual_stream": auc_d,
                "auc_spatial_only": auc_s,
                "delta_auc": diff_auc,
                "sigma_diff": sigma_diff,
                "z_score": z_delong,
                "p_value": p_delong,
                "is_significant_p05": bool(p_delong < 0.05),
                "is_significant_p001": bool(p_delong < 0.001),
            },
            "paired_clustered_bootstrap": {
                "n_bootstraps": 1000,
                "mean_delta_auc": float(np.mean(boot_diffs)),
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "excludes_zero": bool(ci_lower > 0.0),
            },
            "mcnemar_test_overall": mcnemar_overall,
            "mcnemar_test_fake_recall": {
                "fake_recall_dual_stream": float(rec_dual),
                "fake_recall_spatial_only": float(rec_spat),
                "recall_gain": float(rec_dual - rec_spat),
                "mcnemar": mcnemar_fake,
            },
        },
        "domain_classifier_probe": {
            "probe_type": "LogisticRegression(C=1.0, class_weight='balanced')",
            "train_samples_val": len(domain_val),
            "test_samples": len(domain_labels),
            "overall_accuracy": float(domain_acc),
            "balanced_accuracy": float(domain_bacc),
            "confusion_matrix": domain_cm,
            "classification_report": domain_report,
        },
    }

    out_file = os.path.join(output_dir, "tier1_rigorous_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    logger.info("Saved complete rigorous evaluation results to: %s", out_file)


if __name__ == "__main__":
    main()
