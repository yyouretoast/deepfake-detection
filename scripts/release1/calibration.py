"""Stage 3 of the Release-1 pipeline: validation-split Platt temperature scaling,
Youden threshold, and certified Bayesian dual thresholds.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.release1.common import get_sample_stratum, subsample_stratified_val
from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import get_transforms
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import (
    compute_ece,
    find_optimal_threshold,
    fit_platt_scaling,
    fit_temperature_log,
)
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict, compute_dual_thresholds_certified

logger = logging.getLogger("release1_runner")

def calibrate_validation_split(
    backbone_path: str,
    data_root: str,
    splits_path: str,
    output_dir: str,
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> tuple[str, dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrated_path = os.path.join(output_dir, "dual_stream_calibrated.pth")
    if not force_rerun and os.path.exists(calibrated_path):
        try:
            ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
            if "calibration" in ckpt:
                logger.info("Found existing calibrated checkpoint: %s. Skipping calibration.", calibrated_path)
                return calibrated_path, ckpt["calibration"]
        except Exception as e:
            logger.warning("Failed loading existing calibrated checkpoint: %s. Re-calibrating.", e)

    logger.info("--- Calibrating Probabilities and Deriving Decision Thresholds on Val ---")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_samples = splits["val"]
    if smoke_test:
        val_samples = subsample_stratified_val(val_samples, 50, seed=42)
        logger.info("Smoke test subsample: Val=%d", len(val_samples))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    val_loader = DataLoader(
        FaceCropDataset(val_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
    )

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    state_dict = torch.load(backbone_path, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)

    evaluator = ModelEvaluator(model, device=device)
    val_logits, val_targets, val_valid = evaluator.predict_loader(val_loader)
    mask = val_valid > 0.0
    val_logits = val_logits[mask]
    val_targets = val_targets[mask]

    real_logits = val_logits[val_targets == 0]
    fake_logits = val_logits[val_targets == 1]
    logger.info("Validation Logit Separation Analysis:")
    logger.info("  Reals (N=%d): mean=%.4f, std=%.4f, [min=%.4f, max=%.4f]", len(real_logits), float(np.mean(real_logits)), float(np.std(real_logits)), float(np.min(real_logits)), float(np.max(real_logits)))
    logger.info("  Fakes (N=%d): mean=%.4f, std=%.4f, [min=%.4f, max=%.4f]", len(fake_logits), float(np.mean(fake_logits)), float(np.std(fake_logits)), float(np.min(fake_logits)), float(np.max(fake_logits)))
    logger.info("  Separation Delta(mean) = %.4f", float(np.mean(fake_logits) - np.mean(real_logits)))

    # 4-Way Stratified Validation Weights to prevent Celeb-DF fakes (83.8% of val) from dominating calibration
    val_strata = [get_sample_stratum(s[0], s[1]) for s in val_samples]
    val_strata_arr = np.array(val_strata)[mask]
    val_counts = Counter(val_strata_arr)
    num_v_strata = len(val_counts)
    val_weights = np.array([1.0 / (num_v_strata * max(1, val_counts[st])) for st in val_strata_arr])
    logger.info("Validation Stratified Weights for Platt Scaling: %s", dict(val_counts))

    # 2-Parameter Platt Scaling (scale a, bias b) with stratified sample weighting
    scale_a, bias_b, eff_t = fit_platt_scaling(val_logits, val_targets, sample_weights=val_weights)
    t_star = fit_temperature_log(val_logits, val_targets)
    logger.info("Fitted Platt Calibration: a = %.4f, b = %.4f (Eff T = %.4f) | Temp-Only T* = %.4f", scale_a, bias_b, eff_t, t_star)

    is_at_bound = (scale_a <= 0.101 or scale_a >= 9.99 or abs(bias_b) >= 9.99 or t_star <= 0.101 or t_star >= 9.99)
    if is_at_bound:
        err_msg = (
            f"CALIBRATION BOUND HIT: scale_a={scale_a:.4f}, bias_b={bias_b:.4f}, T*={t_star:.4f}. "
            f"Search bounds are a in [0.1, 10.0], b in [-10.0, 10.0], T in [0.1, 10.0]. "
            f"Logit separation Delta(mean) = {float(np.mean(fake_logits) - np.mean(real_logits)):.4f}."
        )
        logger.error(err_msg)
        if not smoke_test:
            raise RuntimeError(err_msg + " Stopping run as calibration optimizer landed on a boundary parameter constraint.")
        else:
            logger.warning("Proceeding under smoke test mode despite bound hit.")

    val_probs_uncal = 1.0 / (1.0 + np.exp(-val_logits))
    val_probs_cal = 1.0 / (1.0 + np.exp(-(scale_a * val_logits + bias_b)))

    ece_uncal = compute_ece(val_probs_uncal, val_targets)
    ece_cal = compute_ece(val_probs_cal, val_targets)
    ece_reduction = ((ece_uncal - ece_cal) / max(1e-6, ece_uncal)) * 100.0
    logger.info("Val ECE: %.4f (Uncal) -> %.4f (Calibrated) [-%.1f%%]", ece_uncal, ece_cal, ece_reduction)

    # Youden's J / Balanced Accuracy optimal threshold
    tau_star, best_bal_acc = find_optimal_threshold(val_targets, val_probs_cal, criterion="balanced_accuracy")
    youden_j = 2.0 * best_bal_acc - 1.0
    logger.info("Optimal Decision Threshold tau* = %.4f (Val Bal. Acc = %.4f, Youden J = %.4f)", tau_star, best_bal_acc, youden_j)

    # Operational Bayesian 3-zone dual thresholds with certified precision reporting
    tau_real, tau_fake, p_real_cert, p_fake_cert = compute_dual_thresholds_certified(
        val_probs_cal, val_targets, min_precision=0.980
    )
    if p_fake_cert < 0.980 or p_real_cert < 0.980:
        logger.warning(
            "Target precision 0.980 unattainable on validation cohort. Operating at maximal certified precision: P*(fake) = %.4f, P*(real) = %.4f",
            p_fake_cert, p_real_cert
        )
    logger.info("Bayesian Decision Thresholds: tau_real = %.4f (P* = %.4f), tau_fake = %.4f (P* = %.4f)", tau_real, p_real_cert, tau_fake, p_fake_cert)

    calib_meta = {
        "optimal_temperature": float(t_star),
        "platt_scale_a": float(scale_a),
        "platt_bias_b": float(bias_b),
        "effective_temperature": float(eff_t),
        "uncalibrated_ece": float(ece_uncal),
        "calibrated_ece": float(ece_cal),
        "ece_reduction_pct": float(ece_reduction),
        "optimal_threshold": float(tau_star),
        "val_balanced_accuracy": float(best_bal_acc),
        "youden_j": float(youden_j),
        "tau_real": float(tau_real),
        "tau_fake": float(tau_fake),
        "certified_precision_real": float(p_real_cert),
        "certified_precision_fake": float(p_fake_cert),
        "is_precision_certified": bool(p_fake_cert >= 0.980 and p_real_cert >= 0.980),
    }

    # Save calibrated checkpoint with metadata (both nested and top-level for backward-compatibility)
    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "calibration": calib_meta,
        "temperature": calib_meta["effective_temperature"],
        "effective_temperature": calib_meta["effective_temperature"],
        "optimal_threshold": calib_meta["optimal_threshold"],
        "tau_real": calib_meta["tau_real"],
        "tau_fake": calib_meta["tau_fake"],
        "platt_scale_a": calib_meta["platt_scale_a"],
        "platt_bias_b": calib_meta["platt_bias_b"],
    }
    torch.save(ckpt_payload, calibrated_path)
    logger.info("Saved Calibrated Checkpoint -> %s", calibrated_path)
    return calibrated_path, calib_meta


# ---------------------------------------------------------------------------
# Stage 4: Test Set Evaluation with Video-Level Clustered Bootstrap
# ---------------------------------------------------------------------------

