"""Stage 4 of the Release-1 pipeline: held-out test evaluation with video-level
clustered bootstrap confidence intervals.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from scripts.release1.common import subsample_stratified_test
from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier
from src.dataset.loader import get_transforms
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import compute_eer
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logger = logging.getLogger("release1_runner")

def compute_video_clustered_bootstrap_auc(
    video_ids: list[str],
    targets: np.ndarray,
    probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Evaluates ROC AUC with 95% video-level clustered bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    unique_videos = np.unique(video_ids)
    video_to_indices: dict[str, list[int]] = {}
    for idx, vid in enumerate(video_ids):
        if vid not in video_to_indices:
            video_to_indices[vid] = []
        video_to_indices[vid].append(idx)

    # Point estimate
    point_auc = float(roc_auc_score(targets, probs))

    boot_aucs: list[float] = []
    n_vids = len(unique_videos)
    for _ in range(n_bootstraps):
        sampled_vids = rng.choice(unique_videos, size=n_vids, replace=True)
        sampled_indices = []
        for sv in sampled_vids:
            sampled_indices.extend(video_to_indices[sv])

        y_sample = targets[sampled_indices]
        p_sample = probs[sampled_indices]
        if len(np.unique(y_sample)) >= 2:
            boot_aucs.append(float(roc_auc_score(y_sample, p_sample)))

    if len(boot_aucs) == 0:
        return point_auc, point_auc, point_auc

    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return point_auc, ci_lower, ci_upper


def evaluate_held_out_test(
    calibrated_path: str,
    data_root: str,
    splits_path: str,
    calib_meta: dict[str, float],
    output_dir: str,
    spatial_checkpoint_path: str | None = None,
    n_bootstraps: int = 1000,
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_summary_path = os.path.join(output_dir, "test_evaluation_summary.json")
    if not force_rerun and os.path.exists(test_summary_path):
        logger.info("Found existing test evaluation summary: %s. Skipping test evaluation.", test_summary_path)
        with open(test_summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if smoke_test:
        n_bootstraps = 10
        logger.info("--- [SMOKE TEST] Evaluating Test Split (Subsampled, 10 Bootstraps) ---")
    else:
        logger.info("--- Evaluating Full Test Set & Computing Clustered Bootstrap CIs ---")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_samples = splits["test"]
    if smoke_test:
        test_samples = subsample_stratified_test(test_samples, 34, seed=42)
        logger.info("Smoke test subsample: Test=%d", len(test_samples))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    test_loader = DataLoader(
        FaceCropDataset(test_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
    )

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)

    evaluator = ModelEvaluator(model, device=device)
    test_logits, test_targets, test_valid = evaluator.predict_loader(test_loader)
    mask = test_valid > 0.0
    test_logits = test_logits[mask]
    test_targets = test_targets[mask]

    scale_a = calib_meta.get("platt_scale_a", 1.0 / max(1e-6, calib_meta.get("optimal_temperature", 1.0)))
    bias_b = calib_meta.get("platt_bias_b", 0.0)
    tau_star = calib_meta["optimal_threshold"]
    test_probs = 1.0 / (1.0 + np.exp(-(scale_a * test_logits + bias_b)))
    test_preds = (test_probs >= tau_star).astype(int)

    # 1. Overall Dual-Stream Test Metrics
    overall_auc = float(roc_auc_score(test_targets, test_probs))
    overall_prauc = float(average_precision_score(test_targets, test_probs))
    overall_f1 = float(f1_score(test_targets, test_preds, zero_division=0))
    overall_bal_acc = float(balanced_accuracy_score(test_targets, test_preds))
    overall_prec = float(precision_score(test_targets, test_preds, zero_division=0))
    overall_rec = float(recall_score(test_targets, test_preds, zero_division=0))
    overall_eer, _ = compute_eer(test_targets, test_probs)

    logger.info(
        "Overall Test (Dual-Stream): ROC AUC = %.4f | PR AUC = %.4f | Fake F1 = %.4f | Bal Acc = %.4f | Prec = %.4f | Rec = %.4f | EER = %.2f%%",
        overall_auc,
        overall_prauc,
        overall_f1,
        overall_bal_acc,
        overall_prec,
        overall_rec,
        overall_eer * 100.0,
    )

    # 2. Trivial Baseline (Always Predict Fake)
    n_fake = int(np.sum(test_targets == 1))
    n_total = len(test_targets)
    trivial_always_fake = {
        "auc": 0.5000,
        "pr_auc": float(n_fake / n_total),
        "f1": float(2.0 * n_fake / (n_total + n_fake)),
        "balanced_accuracy": 0.5000,
        "precision": float(n_fake / n_total),
        "recall": 1.0000,
        "eer": 50.0,
    }

    # 2b. Spatial-Only ConvNeXt Baseline Evaluation
    spatial_only_metrics = {}
    if spatial_checkpoint_path and os.path.exists(spatial_checkpoint_path):
        try:
            sp_model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=False)
            sp_state = torch.load(spatial_checkpoint_path, map_location=device, weights_only=False)
            sp_model.load_state_dict(clean_state_dict(sp_state), strict=False)
            sp_model.to(device)
            sp_eval = ModelEvaluator(sp_model, device=device)
            sp_logits, sp_targets, sp_valid = sp_eval.predict_loader(test_loader)
            sp_m = sp_valid > 0.0
            sp_l = sp_logits[sp_m]
            sp_t = sp_targets[sp_m]
            sp_p = 1.0 / (1.0 + np.exp(-sp_l))
            sp_preds = (sp_p >= 0.50).astype(int)
            sp_eer, _ = compute_eer(sp_t, sp_p)
            spatial_only_metrics = {
                "auc": float(roc_auc_score(sp_t, sp_p)),
                "pr_auc": float(average_precision_score(sp_t, sp_p)),
                "f1": float(f1_score(sp_t, sp_preds, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(sp_t, sp_preds)),
                "precision": float(precision_score(sp_t, sp_preds, zero_division=0)),
                "recall": float(recall_score(sp_t, sp_preds, zero_division=0)),
                "eer": float(sp_eer * 100.0),
            }
            logger.info(
                "Spatial-Only Baseline Test: AUC = %.4f | PR AUC = %.4f | Fake F1 = %.4f | Bal Acc = %.4f | EER = %.2f%%",
                spatial_only_metrics["auc"],
                spatial_only_metrics["pr_auc"],
                spatial_only_metrics["f1"],
                spatial_only_metrics["balanced_accuracy"],
                spatial_only_metrics["eer"],
            )
        except Exception as e:
            logger.warning("Failed evaluating spatial-only baseline from %s: %s", spatial_checkpoint_path, e)

    # 3. Fine-Grained Cohort Categorization
    video_ids = []
    fine_source = []
    gen_names = []
    for s in test_samples:
        path = s[0].replace("\\", "/")
        parts = path.split("/")
        vid = parts[1] if len(parts) > 1 else parts[0]
        video_ids.append(f"{parts[0]}/{vid}")
        lbl = int(s[1])

        if lbl == 0:
            if re.match(r"^\d{3}$", vid):
                fine_source.append("ffpp_real")
            elif re.match(r"^\d{5}$", vid):
                fine_source.append("youtube_real")
            else:
                fine_source.append("celeb_real")
            gen_names.append("authentic")
        else:
            if "__" in vid:
                fine_source.append("dfd_fake")
                gen_names.append("dfd")
            elif re.match(r"^\d{3}_\d{3}$", vid):
                fine_source.append("ffpp_fake")
                dinfo = DomainClassifier.classify(path)
                gen_names.append(dinfo.domain.value)
            else:
                fine_source.append("celeb_fake")
                gen_names.append("celeb")

    video_ids = [video_ids[i] for i, m in enumerate(mask) if m]
    fine_source_arr = np.array([fine_source[i] for i, m in enumerate(mask) if m])
    gen_names_arr = np.array([gen_names[i] for i, m in enumerate(mask) if m])

    # Distinct Per-Source Evaluation Cohorts
    cohort_definitions = {
        "ffpp": (fine_source_arr == "ffpp_fake") | (fine_source_arr == "ffpp_real"),
        "celeb_id_only": (fine_source_arr == "celeb_fake") | (fine_source_arr == "celeb_real"),
        "celeb_all": (fine_source_arr == "celeb_fake") | (fine_source_arr == "celeb_real") | (fine_source_arr == "youtube_real"),
        "dfd_vs_ffpp": (fine_source_arr == "dfd_fake") | (fine_source_arr == "ffpp_real"),
        "dfd_vs_youtube": (fine_source_arr == "dfd_fake") | (fine_source_arr == "youtube_real"),
    }

    per_source_results = {}
    for c_name, c_mask in cohort_definitions.items():
        if np.sum(c_mask) == 0:
            continue
        y_c = test_targets[c_mask]
        p_c = test_probs[c_mask]
        v_c = [video_ids[i] for i, sm in enumerate(c_mask) if sm]
        preds_c = (p_c >= tau_star).astype(int)

        point_auc, ci_low, ci_high = compute_video_clustered_bootstrap_auc(
            v_c, y_c, p_c, n_bootstraps=n_bootstraps
        )
        c_prauc = float(average_precision_score(y_c, p_c))
        c_f1 = float(f1_score(y_c, preds_c, zero_division=0))
        c_bacc = float(balanced_accuracy_score(y_c, preds_c))
        c_prec = float(precision_score(y_c, preds_c, zero_division=0))
        c_rec = float(recall_score(y_c, preds_c, zero_division=0))
        c_eer, _ = compute_eer(y_c, p_c)

        per_source_results[c_name] = {
            "total_crops": int(np.sum(c_mask)),
            "fakes": int(np.sum(y_c == 1)),
            "reals": int(np.sum(y_c == 0)),
            "auc": point_auc,
            "auc_ci_lower_95": ci_low,
            "auc_ci_upper_95": ci_high,
            "pr_auc": c_prauc,
            "f1": c_f1,
            "balanced_accuracy": c_bacc,
            "precision": c_prec,
            "recall": c_rec,
            "eer": float(c_eer * 100.0),
        }
        logger.info(
            "Cohort [%s]: AUC = %.4f [95%% CI: %.4f - %.4f] | F1 = %.4f | Bal Acc = %.4f | Prec = %.4f | EER = %.2f%%",
            c_name.upper(),
            point_auc,
            ci_low,
            ci_high,
            c_f1,
            c_bacc,
            c_prec,
            c_eer * 100.0,
        )

    # Calculate DFD cross-dataset confound gap
    dfd_confound_gap = per_source_results.get("dfd_vs_youtube", {}).get("auc", 0.0) - per_source_results.get("dfd_vs_ffpp", {}).get("auc", 0.0)
    logger.info("DFD Cross-Dataset Confound Gap (YouTube-reals vs FF++ reals): Delta(AUC) = %+.4f", dfd_confound_gap)

    # Legacy compatibility aliases for manuscript generator
    per_source_results["celeb"] = per_source_results.get("celeb_all", per_source_results.get("celeb_id_only", {}))
    per_source_results["dfd"] = per_source_results.get("dfd_vs_ffpp", {})

    # 4. Fine-Grained Subdomain Breakdown (Table 2)
    subdomain_results = {}
    for gen_key in ["deepfakes", "face2face", "faceswap", "neuraltextures", "celeb", "dfd"]:
        gen_fake_mask = (gen_names_arr == gen_key) & (test_targets == 1)
        if np.sum(gen_fake_mask) == 0:
            continue

        if gen_key == "dfd":
            matched_real_mask = (fine_source_arr == "ffpp_real")
        elif gen_key == "celeb":
            matched_real_mask = (fine_source_arr == "celeb_real")
        else:
            matched_real_mask = (fine_source_arr == "ffpp_real")

        eval_mask = gen_fake_mask | matched_real_mask
        y_sub = test_targets[eval_mask]
        p_sub = test_probs[eval_mask]
        preds_sub = (p_sub >= tau_star).astype(int)

        sub_auc = float(roc_auc_score(y_sub, p_sub))
        sub_f1 = float(f1_score(y_sub, preds_sub, zero_division=0))
        sub_prec = float(precision_score(y_sub, preds_sub, zero_division=0))
        sub_rec = float(recall_score(y_sub, preds_sub, zero_division=0))

        subdomain_results[gen_key] = {
            "fake_crops": int(np.sum(gen_fake_mask)),
            "auc": sub_auc,
            "f1": sub_f1,
            "precision": sub_prec,
            "recall": sub_rec,
        }
        logger.info(
            "Subdomain [%s]: Fakes = %d | AUC = %.4f | F1 = %.4f | Prec = %.4f | Rec = %.4f",
            gen_key,
            np.sum(gen_fake_mask),
            sub_auc,
            sub_f1,
            sub_prec,
            sub_rec,
        )

    test_summary = {
        "overall": {
            "auc": overall_auc,
            "pr_auc": overall_prauc,
            "f1": overall_f1,
            "balanced_accuracy": overall_bal_acc,
            "precision": overall_prec,
            "recall": overall_rec,
            "eer": float(overall_eer * 100.0),
        },
        "trivial_baseline_always_fake": trivial_always_fake,
        "spatial_only": spatial_only_metrics,
        "per_source": per_source_results,
        "subdomain_breakdown": subdomain_results,
        "dfd_confound_gap": float(dfd_confound_gap),
    }

    with open(os.path.join(output_dir, "test_evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    return test_summary


# ---------------------------------------------------------------------------
# Stage 5: Spatiotemporal Bi-GRU Video Sequence Modeling
# ---------------------------------------------------------------------------

