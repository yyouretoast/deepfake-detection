"""
True Leave-One-Type-Out (LOTO) Cross-Generator Training & Evaluation Experiment.

Filters out 100% of a target generator domain (e.g., 'neuraltextures' or 'deepfakes') from
training/validation splits, trains a dual-stream detector for N fast epochs, and evaluates
true zero-shot cross-generator generalization on the held-out target set.

Usage:
    accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_loto_experiment.py --holdout neuraltextures --epochs 3
"""

import os
import sys

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMG_SIZE = 256
BATCH_SIZE = 16

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class KaggleFastDataset(Dataset):
    def __init__(self, samples, root_dir, is_train=True):
        self.samples = samples
        self.root_dir = root_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_rel, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path_rel)
        valid_flag = 1.0
        try:
            bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Image read failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
                rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.debug("Image load error %s: %s", full_path, e)
            valid_flag = 0.0
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.is_train and random.random() > 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(valid_flag, dtype=torch.float32)

def dedupe_split(split_list):
    seen, deduped = set(), []
    for entry in split_list:
        path = entry[0] if isinstance(entry, (list, tuple)) else entry
        if path not in seen:
            seen.add(path)
            deduped.append(entry)
    return deduped

def find_dataset_root():
    candidate_paths = [
        '/kaggle/working/local_crops',
        '/kaggle/input/datasets/yassinyasserr/deepfake-crops-512/deepfake_crops_512',
        '/kaggle/input/deepfake-crops-512/deepfake_crops_512',
        '/kaggle/input/deepfake_crops_512'
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, 'splits.json')):
            return p
    for r, d, f in os.walk('/kaggle/input'):
        if 'splits.json' in f:
            return r
    raise FileNotFoundError("Could not locate dataset containing splits.json under /kaggle/input")

def get_differential_param_groups(model):
    backbone_decay, backbone_nodecay = [], []
    head_decay, head_nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = len(param.shape) == 1 or name.endswith(".bias") or "bn" in name or "norm" in name
        if "spatial_backbone" in name:
            if is_no_decay:
                backbone_nodecay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_no_decay:
                head_nodecay.append(param)
            else:
                head_decay.append(param)

    return [
        {'params': backbone_decay, 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': backbone_nodecay, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': head_nodecay, 'lr': 1e-3, 'weight_decay': 0.0}
    ]

def filter_loto_split(samples, holdout_keyword):
    """Filters out samples matching holdout_keyword for training/validation sets."""
    keyword = holdout_keyword.lower()
    retained = [s for s in samples if keyword not in s[0].lower()]
    held_out = [s for s in samples if keyword in s[0].lower()]
    return retained, held_out

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

    train_samples, train_heldout = filter_loto_split(raw_train, args.holdout)
    val_samples, val_heldout = filter_loto_split(raw_val, args.holdout)
    _, test_target_heldout = filter_loto_split(raw_test, args.holdout)

    # Combine all held-out samples for dedicated zero-shot testing
    eval_target_samples = train_heldout + val_heldout + test_target_heldout

    if accelerator.is_main_process:
        logging.info("="*65)
        logging.info("🚀 STARTING TRUE LOTO EXPERIMENT (HOLDOUT: %s)", args.holdout.upper())
        logging.info("  ├─ Retained Train Samples: %d (Filtered out %d '%s' samples)", len(train_samples), len(train_heldout), args.holdout)
        logging.info("  ├─ Retained Val Samples:   %d", len(val_samples))
        logging.info("  └─ Dedicated Held-Out Zero-Shot Test Samples: %d", len(eval_target_samples))
        logging.info("="*65)

    if not eval_target_samples:
        raise ValueError(f"No samples matching holdout keyword '{args.holdout}' were found in dataset!")

    train_ds = KaggleFastDataset(train_samples, data_root, is_train=True)
    val_ds = KaggleFastDataset(val_samples, data_root, is_train=False)
    target_ds = KaggleFastDataset(eval_target_samples, data_root, is_train=False)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
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
        all_targets, all_preds = [], []
        with torch.no_grad():
            for images, labels, valid_flags in val_loader:
                labels = labels.unsqueeze(1)
                valid_flags = valid_flags.unsqueeze(1)
                outputs = model(images)
                preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels))
                all_preds.extend(preds_g.cpu().numpy())
                all_targets.extend(targets_g.cpu().numpy())

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            val_auc = 0.5

        if accelerator.is_main_process:
            logging.info(f"LOTO Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f} | Retained Val AUC: {val_auc:.4f}")

    # Final Zero-Shot Evaluation on Excluded Holdout Generator Set
    model.eval()
    target_targets, target_preds = [], []
    with torch.no_grad():
        for images, labels, valid_flags in target_loader:
            labels = labels.unsqueeze(1)
            outputs = model(images)
            preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels))
            target_preds.extend(preds_g.cpu().numpy())
            target_targets.extend(targets_g.cpu().numpy())

    if accelerator.is_main_process:
        target_preds = np.array(target_preds)
        target_targets = np.array(target_targets)

        if len(np.unique(target_targets)) > 1:
            zero_shot_auc = roc_auc_score(target_targets, target_preds)
            zero_shot_f1 = f1_score(target_targets, (target_preds > 0.01).astype(int))
            zero_shot_prec = precision_score(target_targets, (target_preds > 0.01).astype(int), zero_division=0)
            zero_shot_rec = recall_score(target_targets, (target_preds > 0.01).astype(int), zero_division=0)
            logging.info("\n" + "="*65)
            logging.info("🏆 ZERO-SHOT LOTO EVALUATION ON HELD-OUT '%s':", args.holdout.upper())
            logging.info("  ├─ Zero-Shot AUC:   %.4f", zero_shot_auc)
            logging.info("  ├─ Zero-Shot F1:    %.4f", zero_shot_f1)
            logging.info("  ├─ Precision:       %.4f", zero_shot_prec)
            logging.info("  └─ Recall:          %.4f", zero_shot_rec)
            logging.info("="*65)
        else:
            acc = np.mean((target_preds > 0.01).astype(int) == target_targets)
            logging.info("🏆 ZERO-SHOT ACCURACY ON HELD-OUT '%s': %.4f", args.holdout.upper(), acc)

if __name__ == '__main__':
    main()
