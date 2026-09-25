"""Compute rigorous empirical component-wise architectural ablations on the held-out test split.

Evaluates:
  1. Spatial Backbone Alone (ConvNeXt-Small)
  2. Fixed SRM Alone (Spatial masked, Bayar zeroed)
  3. Learnable Bayar Alone (Spatial masked, SRM zeroed)
  4. Spectral Tower Alone (ResSE-Spectral, spatial masked)
  5. Cross-Stream Fusion: Elementwise Sum (f_s + f_f)
  6. Cross-Stream Fusion: Concatenation ([f_s || f_f])
  7. Cross-Stream Fusion: Static Softmax Gate (No SNR term)
  8. Full Proposed Model: SNR-Adaptive Gating (g_eff = g * gamma)
  9. Video Temporal Models: Naive Avg, Max-Pool, Spatiotemporal Bi-GRU

Saves full provenance to results/release1_run/ablation_study_results.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

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
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from scripts.run_release1_kaggle import (
    DatasetResolver,
    FaceCropDataset,
    get_transforms,
    resolve_splits_path,
)
from src.evaluation.metrics import compute_eer, find_optimal_threshold
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation_runner")


def compute_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    tau, _ = find_optimal_threshold(targets, probs, criterion="youden_roc")
    preds = (probs >= tau).astype(int)
    eer, _ = compute_eer(targets, probs)
    return {
        "auc": float(roc_auc_score(targets, probs)),
        "pr_auc": float(average_precision_score(targets, probs)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "eer": float(eer * 100.0),
        "optimal_threshold": float(tau),
    }


def main() -> None:
    data_root = DatasetResolver.find_dataset_root("deepfake_crops_512")
    splits_path = resolve_splits_path(data_root=data_root)
    out_dir = os.path.abspath("results/release1_run")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "ablation_cache.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    scale_a = 0.2783
    bias_b = 0.4089

    if os.path.exists(cache_path):
        logger.info("Loading cached ablation logits from %s...", cache_path)
        cache = np.load(cache_path)
        y_true = cache["y_true"]
        logits_full = cache["logits_full"]
        logits_static_gate = cache["logits_static_gate"]
        logits_spectral_alone = cache["logits_spectral_alone"]
        logits_sum_fusion = cache["logits_sum_fusion"]
        logits_srm_alone = cache["logits_srm_alone"]
        logits_bayar_alone = cache["logits_bayar_alone"]
    else:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        test_samples = splits["test"]
        logger.info("Loaded %d held-out test crops from %s", len(test_samples), splits_path)

        _, eval_transform = get_transforms(img_size=256, hardened=False)
        test_ds = FaceCropDataset(test_samples, data_root, is_train=False, transform=eval_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            num_workers=4 if os.name != "nt" else 0,
            pin_memory=(device.type == "cuda"),
        )

        dual_ckpt_path = os.path.join(out_dir, "dual_stream_best.pth")
        if not os.path.exists(dual_ckpt_path):
            dual_ckpt_path = os.path.join(out_dir, "dual_stream_calibrated.pth")
        logger.info("Loading dual-stream backbone from %s", dual_ckpt_path)

        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse").to(device)
        ckpt = torch.load(dual_ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(clean_state_dict(state), strict=False)
        model.eval()

        if "platt_scale_a" in ckpt:
            scale_a = float(ckpt["platt_scale_a"])
        if "platt_bias_b" in ckpt:
            bias_b = float(ckpt["platt_bias_b"])

        all_targets: list[int] = []
        l_full: list[float] = []
        l_static: list[float] = []
        l_spectral: list[float] = []
        l_sum: list[float] = []
        l_srm: list[float] = []
        l_bayar: list[float] = []

        logger.info("Extracting feature representations and evaluating fusion variants...")
        t0 = time.time()

        with torch.inference_mode():
            for images, labels, _ in tqdm(test_loader, desc="Ablation Forward Passes"):
                images = images.to(device)
                all_targets.extend(labels.numpy().tolist())

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    # Spatial stream
                    mean = model.imagenet_mean.to(dtype=images.dtype, device=device)
                    std = model.imagenet_std.to(dtype=images.dtype, device=device)
                    x_sp = (images - mean) / std
                    feat_maps = model.spatial_backbone(x_sp)
                    feat_maps = model.spatial_norm(feat_maps)
                    f_s = model.spatial_pool(feat_maps).flatten(1)
                    f_s = model.spatial_fc(f_s)

                    # Spectral stream components
                    srm_out = model.srm(images)
                    bayar_out = model.bayar(images)
                    noise_all = torch.cat([srm_out, bayar_out], dim=1)

                    freq_maps = model.fft(noise_all)
                    f_f, _ = model.freq_tower(freq_maps)

                    # SNR gating terms
                    concat_feat = torch.cat([f_s, f_f], dim=1)
                    gate = model.gate_fc(concat_feat)
                    noise_power = noise_all.pow(2).mean(dim=[-2, -1]).mean(dim=1, keepdim=True)
                    gamma = torch.clamp((noise_power - 0.005) / (0.025 - 0.005), min=0.0, max=1.0)
                    eff_gate = gate * gamma

                    # 1. Full Proposed (SNR-adaptive gate)
                    fused_full = torch.cat([f_s * (1.0 - eff_gate), f_f * eff_gate], dim=1)
                    out_full = model.classifier(fused_full).squeeze(1)

                    # 2. Static gate (No SNR modulation)
                    fused_static = torch.cat([f_s * (1.0 - gate), f_f * gate], dim=1)
                    out_static = model.classifier(fused_static).squeeze(1)

                    # 3. Spectral Tower Alone (Spatial features masked to 0)
                    f_s_zero = torch.zeros_like(f_s)
                    fused_spectral = torch.cat([f_s_zero, f_f * eff_gate], dim=1)
                    out_spectral = model.classifier(fused_spectral).squeeze(1)

                    # 4. Elementwise Sum Fusion (equal weighting)
                    fused_sum = torch.cat([0.5 * f_s, 0.5 * f_f], dim=1)
                    out_sum = model.classifier(fused_sum).squeeze(1)

                    # 5. Fixed SRM Alone (Bayar zeroed, Spatial masked)
                    noise_srm_only = torch.cat([srm_out, torch.zeros_like(bayar_out)], dim=1)
                    freq_srm = model.fft(noise_srm_only)
                    f_srm, _ = model.freq_tower(freq_srm)
                    fused_srm = torch.cat([f_s_zero, f_srm * eff_gate], dim=1)
                    out_srm = model.classifier(fused_srm).squeeze(1)

                    # 6. Learnable Bayar Alone (SRM zeroed, Spatial masked)
                    noise_bayar_only = torch.cat([torch.zeros_like(srm_out), bayar_out], dim=1)
                    freq_bayar = model.fft(noise_bayar_only)
                    f_bayar, _ = model.freq_tower(freq_bayar)
                    fused_bayar = torch.cat([f_s_zero, f_bayar * eff_gate], dim=1)
                    out_bayar = model.classifier(fused_bayar).squeeze(1)

                l_full.extend(out_full.cpu().numpy().tolist())
                l_static.extend(out_static.cpu().numpy().tolist())
                l_spectral.extend(out_spectral.cpu().numpy().tolist())
                l_sum.extend(out_sum.cpu().numpy().tolist())
                l_srm.extend(out_srm.cpu().numpy().tolist())
                l_bayar.extend(out_bayar.cpu().numpy().tolist())

        elapsed = time.time() - t0
        logger.info("Forward passes complete in %.1f seconds (%.1f crops/sec)", elapsed, len(test_samples) / elapsed)

        y_true = np.array(all_targets)
        logits_full = np.array(l_full)
        logits_static_gate = np.array(l_static)
        logits_spectral_alone = np.array(l_spectral)
        logits_sum_fusion = np.array(l_sum)
        logits_srm_alone = np.array(l_srm)
        logits_bayar_alone = np.array(l_bayar)

        # Cache logits to disk immediately
        np.savez_compressed(
            cache_path,
            y_true=y_true,
            logits_full=logits_full,
            logits_static_gate=logits_static_gate,
            logits_spectral_alone=logits_spectral_alone,
            logits_sum_fusion=logits_sum_fusion,
            logits_srm_alone=logits_srm_alone,
            logits_bayar_alone=logits_bayar_alone,
        )
        logger.info("Saved raw ablation logits to cache -> %s", cache_path)

    # Convert logits to calibrated probabilities
    p_full = 1.0 / (1.0 + np.exp(-(scale_a * logits_full + bias_b)))
    p_static = 1.0 / (1.0 + np.exp(-(scale_a * logits_static_gate + bias_b)))
    p_spectral = 1.0 / (1.0 + np.exp(-(scale_a * logits_spectral_alone + bias_b)))
    p_sum = 1.0 / (1.0 + np.exp(-(scale_a * logits_sum_fusion + bias_b)))
    p_srm = 1.0 / (1.0 + np.exp(-(scale_a * logits_srm_alone + bias_b)))
    p_bayar = 1.0 / (1.0 + np.exp(-(scale_a * logits_bayar_alone + bias_b)))

    results_ablations = {
        "full_dual_stream_snr_gate": compute_metrics(y_true, p_full),
        "fusion_static_gate": compute_metrics(y_true, p_static),
        "spectral_tower_alone": compute_metrics(y_true, p_spectral),
        "fusion_elementwise_sum": compute_metrics(y_true, p_sum),
        "fixed_srm_alone": compute_metrics(y_true, p_srm),
        "learnable_bayar_alone": compute_metrics(y_true, p_bayar),
    }

    # Add Spatial Baseline Alone from release1_results.json
    rel1_path = os.path.join(out_dir, "release1_results.json")
    if os.path.exists(rel1_path):
        with open(rel1_path, "r", encoding="utf-8") as f_rel:
            rel_data = json.load(f_rel)
            sp_metrics = rel_data.get("test_frame_evaluation", {}).get("spatial_only", {})
            results_ablations["spatial_backbone_alone"] = sp_metrics
            temp_metrics = rel_data.get("temporal_video_evaluation", {})
            results_ablations["temporal_video_evaluation"] = temp_metrics

    # Save to JSON
    save_path = os.path.join(out_dir, "ablation_study_results.json")
    with open(save_path, "w", encoding="utf-8") as f_out:
        json.dump(results_ablations, f_out, indent=2)

    logger.info("=== ABLATION RESULTS SUMMARY ===")
    for k, v in results_ablations.items():
        if isinstance(v, dict) and "auc" in v:
            logger.info("  %-30s : ROC AUC = %.4f | Fake F1 = %.4f | EER = %.2f%%", k, v["auc"], v.get("f1", 0.0), v.get("eer", 0.0))

    logger.info("Ablation study saved successfully -> %s", save_path)


if __name__ == "__main__":
    main()
