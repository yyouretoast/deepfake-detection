"""Leave-One-Technology-Out (LOTO) cross-generator domain generalization experiment."""

import argparse
import faulthandler
import json
import logging
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    faulthandler.enable()
except Exception:
    pass

for cache_dir in ["/root/.cache/torch/kernels", os.path.expanduser("~/.cache/torch/kernels")]:
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

from accelerate import Accelerator
import cv2
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

# Disable OpenCV multithreading to eliminate fork deadlocks in Linux containers (e.g. Kaggle)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier
from src.dataset.loader import get_transforms
from src.dataset.resolver import find_dataset_root, resolve_splits_path
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.ema import ExponentialMovingAverage
from src.training.loss import FocalLossWithLogits
from src.training.optimization import get_differential_param_groups
from src.training.trainer import DualStreamTrainer
from src.utils.checkpoint import fit_temperature_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


__all__ = [
    "matches_holdout_domain",
    "filter_loto_split_strict",
    "seed_worker",
    "main",
]


def matches_holdout_domain(path: str, holdout_keyword: str) -> bool:
    """Check if sample path belongs to the specified LOTO holdout domain."""
    return DomainClassifier.matches_holdout(path, holdout_keyword)


def filter_loto_split_strict(
    samples: list[tuple[str, float]], holdout: str
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Splits samples into retained subset and held-out fake generator samples."""
    retained = []
    held_out_fakes = []
    for s in samples:
        path, label = s[0], s[1]
        if label == 1.0 and matches_holdout_domain(path, holdout):
            held_out_fakes.append(s)
        else:
            retained.append(s)
    return retained, held_out_fakes


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Leave-One-Technology-Out (LOTO) experiment.")
    parser.add_argument("--holdout", type=str, required=True, help="Holdout generator keyword (e.g. deepfakes, face2face, faceswap, neuraltextures, celeb)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU (default: 8)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default 0 prevents fork deadlocks)")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing dataset and splits.json")
    parser.add_argument(
        "--frequency_backbone",
        type=str,
        default="resse",
        choices=["resse", "legacy"],
        help="Frequency stream architecture: resse or legacy",
    )
    parser.add_argument(
        "--hardened",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use degradation-hardened augmentations",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (default: fp16)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info("Initializing HybridDeepfakeDetector model (%s backbone)...", args.frequency_backbone)
    model = HybridDeepfakeDetector(frequency_backbone=args.frequency_backbone)
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    data_root = find_dataset_root(args.data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits.get("test", [])

    train_loto_samples = [
        s for s in train_samples
        if s and len(s) >= 2 and s[0] and not DomainClassifier.matches_holdout(s[0], args.holdout)
    ]
    val_loto_samples = [
        s for s in val_samples
        if s and len(s) >= 2 and s[0] and not DomainClassifier.matches_holdout(s[0], args.holdout)
    ]

    eval_target_samples = [
        s for s in test_samples
        if s and len(s) >= 2 and s[0] and DomainClassifier.matches_holdout(s[0], args.holdout)
    ]
    if not eval_target_samples:
        eval_target_samples = [
            s for s in val_samples
            if s and len(s) >= 2 and s[0] and DomainClassifier.matches_holdout(s[0], args.holdout)
        ]
    real_test_samples = [s for s in test_samples if s and len(s) >= 2 and s[1] == 0]
    eval_target_samples.extend(real_test_samples[: min(len(real_test_samples), max(500, len(eval_target_samples)))])

    del splits, train_samples, val_samples, test_samples
    import gc
    gc.collect()

    if accelerator.is_main_process:
        logger.info(
            "LOTO Experiment [Holdout: %s] | Train: %d, Val: %d, Zero-Shot Test: %d | BatchSize: %d | GradAccum: %d | MixedPrec: %s",
            args.holdout,
            len(train_loto_samples),
            len(val_loto_samples),
            len(eval_target_samples),
            args.batch_size,
            args.gradient_accumulation_steps,
            args.mixed_precision,
        )

    train_transform, eval_transform = get_transforms(img_size=256, hardened=args.hardened)
    train_ds = FaceCropDataset(train_loto_samples, data_root, is_train=True, transform=train_transform)
    val_ds = FaceCropDataset(val_loto_samples, data_root, is_train=False, transform=eval_transform)

    fold_seed = 42 + sum(ord(c) for c in args.holdout)
    g_train = torch.Generator()
    g_train.manual_seed(fold_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_worker,
    )

    num_fake = sum(1 for s in train_loto_samples if s[1] == 1)
    num_real = len(train_loto_samples) - num_fake
    pos_weight_val = min(float(num_real / max(1, num_fake)), 3.0)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight_tensor)

    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    ema = (
        ExponentialMovingAverage(accelerator.unwrap_model(model), decay=0.999)
        if accelerator.is_main_process
        else None
    )

    trainer = DualStreamTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        ema=ema,
        max_grad_norm=1.0,
    )

    save_path = f"./models/dual_stream_loto_{args.holdout}.pth"
    trainer.fit(num_epochs=args.epochs, save_path=save_path, checkpoint_dir="./checkpoints_loto", patience=3)

    # Zero-shot evaluation on holdout domain (deferred preparation prevents DDP / RNG interference)
    eval_ds = FaceCropDataset(eval_target_samples, data_root, is_train=False, transform=eval_transform)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, worker_init_fn=seed_worker
    )
    eval_loader = accelerator.prepare(eval_loader)

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        with accelerator.autocast():
            for images, labels, valid_flags in eval_loader:
                labels = labels.unsqueeze(1) if labels.ndim == 1 else labels
                outputs = model(images)
                gathered_logits, gathered_labels = accelerator.gather_for_metrics((outputs, labels))
                all_logits.extend(gathered_logits.cpu().reshape(-1).tolist())
                all_targets.extend(gathered_labels.cpu().reshape(-1).tolist())

    if accelerator.is_main_process:
        eval_logits = np.array(all_logits).flatten()
        eval_targets = np.array(all_targets).flatten()

        optimal_temp = fit_temperature_log(eval_logits, eval_targets) if len(np.unique(eval_targets)) > 1 else 1.0
        eval_probs = 1.0 / (1.0 + np.exp(-(eval_logits / optimal_temp)))

        try:
            zero_shot_auc = float(roc_auc_score(eval_targets, eval_probs)) if len(np.unique(eval_targets)) > 1 else 0.5
        except (ValueError, TypeError, RuntimeError):
            zero_shot_auc = 0.5

        eval_preds = (eval_probs >= 0.5).astype(int)
        zero_shot_f1 = float(f1_score(eval_targets, eval_preds, zero_division=0))
        zero_shot_prec = float(precision_score(eval_targets, eval_preds, zero_division=0))
        zero_shot_rec = float(recall_score(eval_targets, eval_preds, zero_division=0))

        # Youden's J optimal threshold calibration
        opt_thresh = 0.5
        opt_f1 = zero_shot_f1
        opt_prec = zero_shot_prec
        opt_rec = zero_shot_rec
        if len(np.unique(eval_targets)) > 1:
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(eval_targets, eval_probs)
                valid_idx = np.isfinite(thresholds)
                if np.any(valid_idx):
                    j_scores = tpr[valid_idx] - fpr[valid_idx]
                    best_j_idx = np.argmax(j_scores)
                    opt_thresh = float(thresholds[valid_idx][best_j_idx])
                    opt_preds = (eval_probs >= opt_thresh).astype(int)
                    opt_f1 = float(f1_score(eval_targets, opt_preds, zero_division=0))
                    opt_prec = float(precision_score(eval_targets, opt_preds, zero_division=0))
                    opt_rec = float(recall_score(eval_targets, opt_preds, zero_division=0))
            except Exception as e:
                logger.warning("Error computing Youden optimal threshold: %s", e)

        logger.info(
            "Holdout [%s] Default (tau=0.50) -> AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            args.holdout,
            zero_shot_auc,
            zero_shot_f1,
            zero_shot_prec,
            zero_shot_rec,
        )
        logger.info(
            "Holdout [%s] Optimal (tau*=%.4f) -> F1: %.4f | Precision: %.4f | Recall: %.4f",
            args.holdout,
            opt_thresh,
            opt_f1,
            opt_prec,
            opt_rec,
        )

        candidates = [
            "/kaggle/working/loto_results.json",
            "/kaggle/working/repo/loto_results.json",
            os.path.join(REPO_ROOT, "results", "loto_results.json"),
            os.path.join(REPO_ROOT, "loto_results.json"),
            "./results/loto_results.json",
            "./loto_results.json",
        ]
        results = []
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list) and len(loaded) > 0:
                            results = loaded
                            break
                except (json.JSONDecodeError, OSError):
                    continue

        results = [r for r in results if r.get("holdout", "").lower() != args.holdout.lower()]
        results.append({
            "holdout": args.holdout,
            "threshold": 0.5,
            "optimal_threshold": float(opt_thresh),
            "temperature": float(optimal_temp),
            "zero_shot_auc": float(zero_shot_auc),
            "zero_shot_f1": float(zero_shot_f1),
            "precision": float(zero_shot_prec),
            "recall": float(zero_shot_rec),
            "optimal_f1": float(opt_f1),
            "optimal_precision": float(opt_prec),
            "optimal_recall": float(opt_rec),
            "n_samples": len(eval_target_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        save_paths = [
            os.path.join(REPO_ROOT, "loto_results.json"),
            os.path.join(REPO_ROOT, "results", "loto_results.json"),
        ]
        if os.path.exists("/kaggle/working"):
            save_paths.append("/kaggle/working/loto_results.json")
            save_paths.append("/kaggle/working/repo/loto_results.json")

        for p in save_paths:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                with open(p, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info("Saved LOTO result entry to %s", p)
            except OSError:
                pass

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
