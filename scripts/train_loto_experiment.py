"""
True Leave-One-Type-Out (LOTO) Cross-Generator Training & Evaluation Experiment.

Filters out 100% of FAKE samples (y=1) belonging to a target generator domain
(e.g., 'neuraltextures' or 'deepfakes') from training/validation splits via strict folder token matching,
while retaining ALL REAL samples (y=0).

Includes valid_flags masking across all evaluation loops, drop_last=True for DDP batch safety,
train_loader sampler set_epoch(epoch) for shuffle entropy, and exports results to JSON.

Usage:
    accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_loto_experiment.py --holdout neuraltextures --epochs 3
"""

import os
import sys
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import time
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from accelerate import Accelerator
from src.models.hybrid_detector import HybridDeepfakeDetector
from scripts.train_dual_stream_ddp import (
    LEARNING_RATE_BACKBONE,
    LEARNING_RATE_HEAD,
    dedupe_split,
    find_dataset_root,
    get_differential_param_groups,
    seed_everything,
    seed_worker,
    KaggleFastDataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMG_SIZE = 256
BATCH_SIZE = 16

seed_everything(42)

def matches_holdout_domain(rel_path: str, holdout_keyword: str) -> bool:
    """
    Strict directory token matching to identify holdout generator domain.
    Prevents false-positive matches on filename frames (e.g. 'id0_frame1').
    """
    path_norm = rel_path.replace("\\", "/")
    path_parts = [p.lower() for p in path_norm.split("/")]
    kw = holdout_keyword.lower()

    for p in path_parts:
        if kw == "neuraltextures" and (p == "neuraltextures" or p == "nt"):
            return True
        elif kw == "deepfakes" and (p == "deepfakes" or p == "df"):
            return True
        elif kw == "face2face" and (p == "face2face" or p == "f2f"):
            return True
        elif kw == "faceswap" and (p == "faceswap" or p == "fs"):
            return True
        elif kw == "celeb" and ("celeb" in p):
            return True

    if re.search(r"/(?:" + re.escape(kw) + r")/", path_norm, re.IGNORECASE):
        return True

    return False

def filter_loto_split_strict(samples, holdout_keyword):
    """
    Strictly filters out FAKE samples (y=1) matching holdout_keyword for training/val sets.
    Retains ALL REAL samples (y=0) in training set to preserve real face representation.
    """
    retained = []
    held_out_fakes = []

    for s in samples:
        path, label = s[0], s[1]
        is_match = matches_holdout_domain(path, holdout_keyword)
        if is_match and label == 1.0:
            held_out_fakes.append(s)
        else:
            retained.append(s)

    return retained, held_out_fakes

def main():
    parser = argparse.ArgumentParser(description="LOTO Dual-Stream Deepfake Training")
    parser.add_argument("--holdout", type=str, required=True, help="Generator domain to hold out (e.g. neuraltextures, deepfakes)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of LOTO training epochs")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    data_root = find_dataset_root()

    splits_path = os.path.join(data_root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    raw_train = dedupe_split(splits['train'])
    raw_val = dedupe_split(splits['val'])
    raw_test = dedupe_split(splits['test'])

    train_samples, train_heldout_fakes = filter_loto_split_strict(raw_train, args.holdout)
    val_samples, val_heldout_fakes = filter_loto_split_strict(raw_val, args.holdout)
    _, test_heldout_fakes = filter_loto_split_strict(raw_test, args.holdout)

    all_heldout_fakes = train_heldout_fakes + val_heldout_fakes + test_heldout_fakes
    test_reals = [s for s in raw_test if s[1] == 0.0]
    
    eval_target_samples = all_heldout_fakes + test_reals

    if accelerator.is_main_process:
        logging.info("="*65)
        logging.info("🚀 STARTING TRUE LOTO EXPERIMENT (HOLDOUT: %s)", args.holdout.upper())
        logging.info("  ├─ Retained Train Samples: %d (Filtered out %d FAKE '%s' samples)", len(train_samples), len(train_heldout_fakes), args.holdout)
        logging.info("  ├─ Retained Val Samples:   %d", len(val_samples))
        logging.info("  └─ Zero-Shot Test Evaluation Set: %d samples (%d Zero-Shot Fakes vs %d Reals)", 
                     len(eval_target_samples), len(all_heldout_fakes), len(test_reals))
        logging.info("="*65)

    if len(all_heldout_fakes) < 5:
        raise ValueError(f"Held-out fake sample count for keyword '{args.holdout}' is too small ({len(all_heldout_fakes)} < 5)!")

    train_ds = KaggleFastDataset(train_samples, data_root, is_train=True)
    val_ds = KaggleFastDataset(val_samples, data_root, is_train=False)
    target_ds = KaggleFastDataset(eval_target_samples, data_root, is_train=False)

    g = torch.Generator()
    g.manual_seed(42)

    # Added drop_last=True for DDP batch safety
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    target_loader = DataLoader(target_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    num_fake = sum(1 for s in train_samples if s[1] == 1)
    num_real = len(train_samples) - num_fake
    pos_weight_val = num_real / max(1, num_fake)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)

    model = HybridDeepfakeDetector()
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    model, optimizer, train_loader, val_loader, target_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, target_loader, scheduler
    )

    for epoch in range(args.epochs):
        # Added DDP sampler set_epoch for shuffle entropy
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = torch.tensor(0.0, device=accelerator.device)

        for images, labels, valid_flags in train_loader:
            labels = labels.unsqueeze(1)
            valid_flags = valid_flags.unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)

            with accelerator.autocast():
                outputs = model(images)
                loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight_tensor, reduction='none')
                loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.detach() * images.size(0)

        scheduler.step()

        total_train_loss = accelerator.reduce(running_loss, reduction='sum').item()
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for images, labels, valid_flags in val_loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                images, labels = images[mask], labels[mask]
                labels_un = labels.unsqueeze(1)
                outputs = model(images)
                preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels_un))
                val_preds.extend(preds_g.cpu().numpy())
                val_targets.extend(targets_g.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except Exception:
            val_auc = 0.5

        if accelerator.is_main_process:
            logging.info(f"LOTO Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f} | Retained Val AUC: {val_auc:.4f}")

    # Dynamic Validation Threshold Search on Retained Val Set
    best_thresh = 0.5
    best_val_f1 = 0.0
    for thresh in np.arange(0.01, 0.95, 0.01):
        f1 = f1_score(val_targets, (val_preds > thresh).astype(int), zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = thresh

    # Final Zero-Shot Evaluation on Held-Out Generator Set (Zero-Shot on Fake, In-Distribution on Real)
    model.eval()
    target_targets, target_preds = [], []
    with torch.no_grad():
        for images, labels, valid_flags in target_loader:
            mask = valid_flags.bool()
            if not mask.any():
                continue
            images, labels = images[mask], labels[mask]
            labels_un = labels.unsqueeze(1)
            outputs = model(images)
            preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels_un))
            target_preds.extend(preds_g.cpu().numpy())
            target_targets.extend(targets_g.cpu().numpy())

    if accelerator.is_main_process:
        target_preds = np.array(target_preds)
        target_targets = np.array(target_targets)

        zero_shot_auc = roc_auc_score(target_targets, target_preds)
        binary_preds = (target_preds > best_thresh).astype(int)
        zero_shot_f1 = f1_score(target_targets, binary_preds, zero_division=0)
        zero_shot_prec = precision_score(target_targets, binary_preds, zero_division=0)
        zero_shot_rec = recall_score(target_targets, binary_preds, zero_division=0)

        logging.info("\n" + "="*65)
        logging.info("🏆 ZERO-SHOT LOTO EVALUATION ON HELD-OUT '%s':", args.holdout.upper())
        logging.info("   (Zero-Shot on Fake Generator, In-Distribution on Real Faces)")
        logging.info("  ├─ Retained Val Optimal Threshold: %.2f", best_thresh)
        logging.info("  ├─ Generalization AUC:            %.4f", zero_shot_auc)
        logging.info("  ├─ Generalization F1:             %.4f", zero_shot_f1)
        logging.info("  ├─ Precision:                    %.4f", zero_shot_prec)
        logging.info("  └─ Recall:                       %.4f", zero_shot_rec)
        logging.info("="*65)

        # JSON Persistence
        res_file = '/kaggle/working/loto_results.json'
        results = []
        if os.path.exists(res_file):
            try:
                with open(res_file, 'r') as f:
                    results = json.load(f)
            except Exception:
                results = []

        results.append({
            "holdout": args.holdout,
            "threshold": float(best_thresh),
            "zero_shot_auc": float(zero_shot_auc),
            "zero_shot_f1": float(zero_shot_f1),
            "precision": float(zero_shot_prec),
            "recall": float(zero_shot_rec),
            "n_samples": len(eval_target_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        with open(res_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info("💾 Saved LOTO result entry to %s", res_file)

if __name__ == '__main__':
    main()
