"""
Leave-One-Target-Out (LOTO) Cross-Generator Training & Evaluation Experiment.

Filters out FAKE samples (y=1) belonging to a target generator domain
(e.g., 'celeb' or 'neuraltextures') from training/validation splits,
while retaining ALL REAL samples (y=0) to preserve baseline real distributions.

Usage:
    accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_loto_experiment.py --holdout celeb --epochs 3
"""

import argparse
import json
import logging
import os
import re
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.train_dual_stream_ddp import (  # noqa: E402
    KaggleFastDataset,
    find_dataset_root,
    get_differential_param_groups,
    seed_everything,
    seed_worker,
)
from src.config import load_config  # noqa: E402
from src.dataset.loader import dedupe_split  # noqa: E402
from src.models.hybrid_detector import HybridDeepfakeDetector  # noqa: E402
from src.utils.checkpoint import fit_temperature_log  # noqa: E402

CONFIG = load_config()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMG_SIZE = CONFIG.get("preprocessing", {}).get("img_size", 256)
BATCH_SIZE = CONFIG.get("training", {}).get("batch_size", 16)

seed_everything(42)

def matches_holdout_domain(rel_path: str, holdout_keyword: str) -> bool:
    """
    Identifies if a sample path belongs to the holdout generator domain.
    Extracts relative path starting from 'fake/' or 'real/' to ignore parent dataset container folders.
    """
    path_norm = rel_path.replace("\\", "/").lower()
    kw = holdout_keyword.lower()

    # Isolate subpath relative to fake/ or real/
    if "fake/" in path_norm:
        sub_path = "fake/" + path_norm.split("fake/")[1]
    elif "real/" in path_norm:
        sub_path = "real/" + path_norm.split("real/")[1]
    else:
        sub_path = path_norm

    if kw in ("celeb", "celebdf", "celeb-df"):
        return bool(re.search(r"id\d+_id\d+", sub_path)) or "__" in sub_path or "celeb" in sub_path

    folder = sub_path.split("/")[1] if len(sub_path.split("/")) > 1 else ""
    if re.match(r"^\d{3}_\d{3}$", folder):
        pair_num = int(folder.split("_")[0])
        if kw in ("deepfakes", "df") and (0 <= pair_num <= 99 or (pair_num < 200 and kw in ("df", "deepfakes"))):
            return True
        elif kw in ("face2face", "f2f") and (100 <= pair_num <= 199 or (200 <= pair_num <= 399 and kw in ("f2f", "face2face"))):
            return True
        elif kw in ("faceswap", "fs") and (200 <= pair_num <= 299 or (400 <= pair_num <= 599 and kw in ("fs", "faceswap"))):
            return True
        elif kw in ("neuraltextures", "nt") and (300 <= pair_num <= 399 or (600 <= pair_num <= 799 and kw in ("nt", "neuraltextures"))):
            return True

    return kw in sub_path

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
    parser.add_argument("--holdout", "--holdout_domain", dest="holdout", type=str, required=True, help="Generator domain to hold out (e.g. celeb, neuraltextures, deepfakes, face2face, faceswap)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of LOTO training epochs")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json and cropped dataset")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    data_root = find_dataset_root(args.data_dir)

    splits_path = os.path.join(data_root, 'splits.json')
    if os.path.exists('/kaggle/working/splits.json'):
        splits_path = '/kaggle/working/splits.json'
    elif os.path.exists('./splits.json'):
        splits_path = './splits.json'

    if accelerator.is_main_process:
        logger.info(f"Loading LOTO dataset splits from: {splits_path}")

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
        logger.info("Starting LOTO experiment (Holdout domain: %s)", args.holdout.upper())
        logger.info("  Retained train samples: %d (Filtered out %d FAKE '%s' samples)", len(train_samples), len(train_heldout_fakes), args.holdout)
        logger.info("  Retained val samples:   %d", len(val_samples))
        logger.info("  Zero-shot test set:     %d samples (%d zero-shot fakes vs %d reals)",
                     len(eval_target_samples), len(all_heldout_fakes), len(test_reals))

    import gc
    del splits, raw_train, raw_val, raw_test
    gc.collect()

    if len(all_heldout_fakes) < 5:
        raise ValueError(f"Held-out fake sample count for keyword '{args.holdout}' is too small ({len(all_heldout_fakes)} < 5)!")

    train_ds = KaggleFastDataset(train_samples, data_root, is_train=True)
    val_ds = KaggleFastDataset(val_samples, data_root, is_train=False)
    target_ds = KaggleFastDataset(eval_target_samples, data_root, is_train=False)

    fold_seed = 42 + (abs(hash(args.holdout)) % 1000)
    seed_everything(fold_seed)

    g = torch.Generator()
    g.manual_seed(fold_seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    target_loader = DataLoader(
        target_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )

    num_fake = sum(1 for s in train_samples if s[1] == 1)
    num_real = len(train_samples) - num_fake
    pos_weight_val = num_real / max(1, num_fake)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)

    model = HybridDeepfakeDetector()
    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    model, optimizer, train_loader, val_loader, target_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, target_loader, scheduler
    )

    from tqdm import tqdm
    for epoch in range(args.epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = torch.tensor(0.0, device=accelerator.device)

        train_iter = tqdm(train_loader, desc=f"LOTO Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        for images, labels, valid_flags in train_iter:
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
            train_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        total_train_loss = accelerator.reduce(running_loss, reduction='sum').item()
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc="Validating", disable=not accelerator.is_main_process)
            for images, labels, valid_flags in val_iter:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                images, labels = images[mask], labels[mask]
                outputs = model(images).squeeze(-1)
                logits_g, targets_g = accelerator.gather_for_metrics((outputs, labels))
                if accelerator.is_main_process:
                    val_logits.extend(logits_g.cpu().numpy())
                    val_targets.extend(targets_g.cpu().numpy())

        if accelerator.is_main_process:
            val_logits = np.array(val_logits)
            val_targets = np.array(val_targets)
            val_probs_raw = 1.0 / (1.0 + np.exp(-val_logits))
            try:
                val_auc = roc_auc_score(val_targets, val_probs_raw)
            except (ValueError, TypeError, RuntimeError):
                val_auc = 0.5

            print(f"LOTO Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f} | Retained Val AUC: {val_auc:.4f}", flush=True)

    if accelerator.is_main_process:
        optimal_temp = fit_temperature_log(val_logits, val_targets)
        val_probs_calibrated = 1.0 / (1.0 + np.exp(-(val_logits / optimal_temp)))

        best_thresh = 0.5
        best_val_f1 = 0.0
        for thresh in np.arange(0.01, 0.95, 0.01):
            f1 = f1_score(val_targets, (val_probs_calibrated > thresh).astype(int), zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thresh = thresh

    # Final Zero-Shot Evaluation on Held-Out Generator Set
    model.eval()
    target_logits, target_targets = [], []
    with torch.no_grad():
        for images, labels, valid_flags in target_loader:
            mask = valid_flags.bool()
            if not mask.any():
                continue
            images, labels = images[mask], labels[mask]
            outputs = model(images).squeeze(-1)
            logits_g, targets_g = accelerator.gather_for_metrics((outputs, labels))
            if accelerator.is_main_process:
                target_logits.extend(logits_g.cpu().numpy())
                target_targets.extend(targets_g.cpu().numpy())

    if accelerator.is_main_process:
        target_logits = np.array(target_logits)[:len(eval_target_samples)]
        target_targets = np.array(target_targets)[:len(eval_target_samples)]

        target_probs_calibrated = 1.0 / (1.0 + np.exp(-(target_logits / optimal_temp)))
        zero_shot_auc = roc_auc_score(target_targets, target_probs_calibrated)
        binary_preds = (target_probs_calibrated > best_thresh).astype(int)
        zero_shot_f1 = f1_score(target_targets, binary_preds, zero_division=0)
        zero_shot_prec = precision_score(target_targets, binary_preds, zero_division=0)
        zero_shot_rec = recall_score(target_targets, binary_preds, zero_division=0)

        logger.info("Zero-Shot LOTO evaluation on held-out '%s':", args.holdout.upper())
        logger.info("  Threshold: %.2f | Temperature (T*): %.4f", best_thresh, optimal_temp)
        logger.info("  AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f", zero_shot_auc, zero_shot_f1, zero_shot_prec, zero_shot_rec)

        res_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else REPO_ROOT
        res_file = os.path.join(res_dir, "loto_results.json")
        results = []
        if os.path.exists(res_file):
            try:
                with open(res_file, 'r') as f:
                    results = json.load(f)
            except (json.JSONDecodeError, OSError):
                results = []

        results.append({
            "holdout": args.holdout,
            "threshold": float(best_thresh),
            "temperature": float(optimal_temp),
            "zero_shot_auc": float(zero_shot_auc),
            "zero_shot_f1": float(zero_shot_f1),
            "precision": float(zero_shot_prec),
            "recall": float(zero_shot_rec),
            "n_samples": len(eval_target_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        save_paths = [os.path.join(REPO_ROOT, "loto_results.json")]
        if os.path.exists("/kaggle/working"):
            save_paths.append("/kaggle/working/loto_results.json")
            save_paths.append("/kaggle/working/repo/loto_results.json")

        for p in save_paths:
            try:
                with open(p, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info("Saved LOTO result entry to %s", p)
            except OSError:
                pass

    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
