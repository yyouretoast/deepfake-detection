"""Stage 7 of the Release-1 pipeline: Leave-One-Target-Out cross-generator
domain generalization benchmark (5-fold).

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.release1.common import subsample_balanced
from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier
from src.dataset.loader import get_transforms
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import compute_eer, fit_temperature_log
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.optimization import get_differential_param_groups

logger = logging.getLogger("release1_runner")

def run_loto_experiment(
    data_root: str,
    splits_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loto_summary_path = os.path.join(output_dir, "loto_results.json")
    if not force_rerun and os.path.exists(loto_summary_path):
        logger.info("Found existing LOTO benchmark results: %s. Skipping LOTO.", loto_summary_path)
        with open(loto_summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    checkpoint_path = os.path.join(output_dir, "loto_checkpoint.json")
    loto_results: dict[str, Any] = {}
    if not force_rerun and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                loto_results = json.load(f)
            logger.info("Resuming LOTO: already completed %d folds (%s)", len(loto_results), list(loto_results.keys()))
        except Exception:
            loto_results = {}

    if smoke_test:
        epochs = 1
        batch_size = min(batch_size, 8)
        logger.info("--- [SMOKE TEST] Running 4-Fold LOMO Benchmark (1 Epoch, Subsampled) ---")
    else:
        logger.info("--- Running 4-Fold Leave-One-Manipulation-Out (LOMO) Generalization Benchmark ---")
    folds = [
        ("Fold 1", "deepfakes", "FF++ Deepfakes (Pairs 0-99)"),
        ("Fold 2", "face2face", "FF++ Face2Face (Pairs 100-399)"),
        ("Fold 3", "faceswap", "FF++ FaceSwap (Pairs 400-599)"),
        ("Fold 4", "neuraltextures", "FF++ NeuralTextures (Pairs 600-799)"),
    ]

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_samples = splits["train"]
    test_samples = splits["test"]

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
    for fold_name, holdout_key, display_name in folds:
        if holdout_key in loto_results:
            logger.info("Skipping %s (%s) - already completed in checkpoint.", fold_name, display_name)
            continue

        logger.info("\n=== %s: Holding out %s ===", fold_name, display_name)
        # Filter train split: pure FaceForensics++ (quarantine Celeb-DF completely and exclude held-out fakes)
        retained_train = []
        for s in train_samples:
            p, lbl = s[0], s[1]
            norm = p.replace("\\", "/").lower()
            parts = norm.split("/")
            # Exclude Celeb-DF completely from training to ensure pure FF++ evaluation
            if "celeb" in norm or (len(parts) > 1 and "id" in parts[1]):
                continue
            # Exclude held-out manipulation method fakes
            if lbl == 1.0 and DomainClassifier.matches_holdout(p, holdout_key):
                continue
            retained_train.append(s)

        # Held-out test fakes
        held_out_test_fakes = [s for s in test_samples if s[1] == 1.0 and DomainClassifier.matches_holdout(s[0], holdout_key)]
        all_test_reals = [s for s in test_samples if s[1] == 0.0 and s[0].replace("\\", "/").split("/")[1].isdigit() and len(s[0].replace("\\", "/").split("/")[1]) == 3]

        if smoke_test:
            retained_train = subsample_balanced(retained_train, 40, seed=42)
            held_out_test_fakes = subsample_balanced(held_out_test_fakes, 10, seed=42)
            all_test_reals = subsample_balanced(all_test_reals, 10, seed=42)

        # Canonical 1:1 balanced evaluation cohort (Rossler et al., ICCV 2019)
        rng = random.Random(42)
        n_eval = min(len(held_out_test_fakes), len(all_test_reals))
        eval_fakes = held_out_test_fakes[:n_eval]
        eval_reals = rng.sample(all_test_reals, n_eval)
        eval_samples = eval_fakes + eval_reals
        logger.info(
            "Fold %s: Retained Train = %d, Eval Set = %d (%d Fake, %d Real, 1:1 Balanced)",
            fold_name,
            len(retained_train),
            len(eval_samples),
            len(eval_fakes),
            len(eval_reals),
        )

        # Train compact model for 3 epochs (1 in smoke mode)
        model = HybridDeepfakeDetector(pretrained=True, frequency_backbone="resse").to(device)
        labels = [int(s[1]) for s in retained_train]
        w_real = 1.0 / max(1, len(labels) - sum(labels))
        w_fake = 1.0 / max(1, sum(labels))
        weights = [w_fake if y == 1 else w_real for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            FaceCropDataset(retained_train, data_root, is_train=True, transform=train_transform),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2 if os.name != "nt" else 0,
        )
        eval_loader = DataLoader(
            FaceCropDataset(eval_samples, data_root, is_train=False, transform=eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if os.name != "nt" else 0,
        )

        optimizer = torch.optim.AdamW(get_differential_param_groups(model, lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-2), weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        for _ep in range(epochs):
            model.train()
            for images, b_labels, v_flags in train_loader:
                images = images.to(device)
                b_labels = b_labels.to(device).unsqueeze(1)
                v_mask = v_flags.to(device).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs, aux_outputs = model(images, return_aux=True)
                    loss_main = (criterion(outputs, b_labels) * v_mask).sum() / torch.clamp(v_mask.sum(), min=1.0)
                    loss_aux = (criterion(aux_outputs, b_labels) * v_mask).sum() / torch.clamp(v_mask.sum(), min=1.0)
                    loss = loss_main + 0.30 * loss_aux

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Zero-shot evaluation
        model.eval()
        evaluator = ModelEvaluator(model, device=device)
        logits, targets, valid = evaluator.predict_loader(eval_loader)
        mask = valid > 0.0
        logits = logits[mask]
        targets = targets[mask]

        # Zero-shot evaluation using uncalibrated logits to prevent test-set temperature leakage
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.50).astype(int)
        t_fold = fit_temperature_log(logits, targets)  # Diagnostic post-hoc oracle check only
        if len(np.unique(targets)) > 1:
            fold_auc = float(roc_auc_score(targets, probs))
        else:
            fold_auc = 0.5
        bacc = float(balanced_accuracy_score(targets, preds))
        fold_f1 = float(f1_score(targets, preds, zero_division=0))
        fold_prec = float(precision_score(targets, preds, zero_division=0))
        eer, _ = compute_eer(targets, probs)

        logger.info("Fold %s (%s): T* = %.4f | Zero-Shot AUC = %.4f | Bal Acc = %.4f | F1 = %.4f | Prec = %.4f | EER = %.2f%%", fold_name, display_name, t_fold, fold_auc, bacc, fold_f1, fold_prec, eer * 100)
        loto_results[holdout_key] = {
            "fold_name": fold_name,
            "display_name": display_name,
            "holdout_samples": len(eval_fakes),
            "fitted_t_star": float(t_fold),
            "zero_shot_auc": fold_auc,
            "balanced_accuracy": bacc,
            "zero_shot_f1": fold_f1,
            "zero_shot_precision": fold_prec,
            "eer": eer,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(loto_results, f, indent=2)

    with open(loto_summary_path, "w", encoding="utf-8") as f:
        json.dump(loto_results, f, indent=2)

    return loto_results


# ---------------------------------------------------------------------------
# Stage 7: Robustness Stress-Testing (Table 4)
# ---------------------------------------------------------------------------

