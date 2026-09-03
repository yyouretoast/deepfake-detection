"""Leave-One-Technology-Out (LOTO) cross-generator domain generalization experiment."""

import argparse
import json
import logging
import os
import random
import sys
import time

from accelerate import Accelerator
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Leave-One-Technology-Out (LOTO) experiment.")
    parser.add_argument("--holdout", type=str, required=True, help="Holdout generator keyword (e.g. deepfakes, face2face, faceswap, neuraltextures, celeb)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing dataset and splits.json")
    parser.add_argument(
        "--frequency_backbone",
        type=str,
        default="resse",
        choices=["resse", "legacy"],
        help="Frequency stream architecture: resse or legacy",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    data_root = find_dataset_root(args.data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits.get("test", [])

    train_loto_samples = [s for s in train_samples if not DomainClassifier.matches_holdout(s[0], args.holdout)]
    val_loto_samples = [s for s in val_samples if not DomainClassifier.matches_holdout(s[0], args.holdout)]

    eval_target_samples = [s for s in test_samples if DomainClassifier.matches_holdout(s[0], args.holdout)]
    if not eval_target_samples:
        eval_target_samples = [s for s in val_samples if DomainClassifier.matches_holdout(s[0], args.holdout)]
    real_test_samples = [s for s in test_samples if s[1] == 0]
    eval_target_samples.extend(real_test_samples[: min(len(real_test_samples), max(500, len(eval_target_samples)))])

    if accelerator.is_main_process:
        logger.info(
            "LOTO Experiment [Holdout: %s] | Train: %d, Val: %d, Zero-Shot Test: %d",
            args.holdout,
            len(train_loto_samples),
            len(val_loto_samples),
            len(eval_target_samples),
        )

    train_transform, eval_transform = get_transforms(img_size=256)
    train_ds = FaceCropDataset(train_loto_samples, data_root, is_train=True, transform=train_transform)
    val_ds = FaceCropDataset(val_loto_samples, data_root, is_train=False, transform=eval_transform)
    eval_ds = FaceCropDataset(eval_target_samples, data_root, is_train=False, transform=eval_transform)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )

    num_fake = sum(1 for s in train_loto_samples if s[1] == 1)
    num_real = len(train_loto_samples) - num_fake
    pos_weight_val = min(float(num_real / max(1, num_fake)), 3.0)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)

    model = HybridDeepfakeDetector(frequency_backbone=args.frequency_backbone)
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight_tensor)

    ema = ExponentialMovingAverage(model, decay=0.999) if accelerator.is_main_process else None

    if accelerator.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model, optimizer, train_loader, val_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, eval_loader, scheduler
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

    # Zero-shot evaluation on holdout domain
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
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

        logger.info(
            "Holdout [%s] Final Metrics -> AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            args.holdout,
            zero_shot_auc,
            zero_shot_f1,
            zero_shot_prec,
            zero_shot_rec,
        )

        candidates = [
            "/kaggle/working/loto_results.json",
            "/kaggle/working/repo/loto_results.json",
            os.path.join(REPO_ROOT, "loto_results.json"),
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
            "temperature": float(optimal_temp),
            "zero_shot_auc": float(zero_shot_auc),
            "zero_shot_f1": float(zero_shot_f1),
            "precision": float(zero_shot_prec),
            "recall": float(zero_shot_rec),
            "n_samples": len(eval_target_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        save_paths = [os.path.join(REPO_ROOT, "loto_results.json")]
        if os.path.exists("/kaggle/working"):
            save_paths.append("/kaggle/working/loto_results.json")
            save_paths.append("/kaggle/working/repo/loto_results.json")

        for p in save_paths:
            try:
                with open(p, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info("Saved LOTO result entry to %s", p)
            except OSError:
                pass

    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
