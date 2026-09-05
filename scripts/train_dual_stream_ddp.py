"""Train Dual-Stream Deepfake Detector with Distributed Data Parallel (DDP) and FP16/BF16 AMP."""

import argparse
import json
import logging
import os
import random
import sys

from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import get_transforms
from src.dataset.resolver import find_dataset_root, resolve_splits_path
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.ema import ExponentialMovingAverage
from src.training.loss import FocalLossWithLogits
from src.training.optimization import create_scheduler, get_differential_param_groups
from src.training.trainer import DualStreamTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 8
BEST_MODEL_WEIGHTS_PATH = "./models/dual_stream_best.pth"
CHECKPOINT_DIR = "./checkpoints_ddp"

__all__ = [
    "find_dataset_root",
    "get_differential_param_groups",
    "ExponentialMovingAverage",
    "FocalLossWithLogits",
    "seed_worker",
    "main",
]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dual-Stream Deepfake Detector DDP")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json and cropped dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size per process")
    parser.add_argument("--save_path", type=str, default=BEST_MODEL_WEIGHTS_PATH, help="Path to save best weights")
    parser.add_argument(
        "--frequency_backbone",
        type=str,
        default="resse",
        choices=["resse", "legacy"],
        help="Frequency stream architecture: resse (~2.9M ResSE tower) or legacy (90k CNN)",
    )
    parser.add_argument(
        "--hardened",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use degradation-hardened augmentations (JPEG, blur, dropout)",
    )
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)

    data_root = find_dataset_root(args.data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]

    train_transform, eval_transform = get_transforms(img_size=IMG_SIZE, hardened=args.hardened)
    train_ds = FaceCropDataset(train_samples, data_root, is_train=True, transform=train_transform)
    val_ds = FaceCropDataset(val_samples, data_root, is_train=False, transform=eval_transform)

    g = torch.Generator()
    g.manual_seed(42)

    # 50/50 Balanced Sampling: Eliminates +1.455 Bayesian log-prior bias and equalizes gradient variance
    labels = [int(s[1]) for s in train_samples]
    num_fake = sum(labels)
    num_real = len(labels) - num_fake
    logger.info("Training Split: %d Real, %d Fake (Total: %d)", num_real, num_fake, len(labels))

    weight_real = 1.0 / max(1, num_real)
    weight_fake = 1.0 / max(1, num_fake)
    sample_weights = [weight_fake if y == 1 else weight_real for y in labels]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = HybridDeepfakeDetector(pretrained=True, frequency_backbone=args.frequency_backbone)
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = create_scheduler(optimizer, warmup_epochs=1, total_epochs=args.epochs)
    # With 50/50 balanced batches, pos_weight=None ensures symmetric gradient updates on hard examples
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=None)

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

    trainer.fit(
        num_epochs=args.epochs,
        save_path=args.save_path,
        checkpoint_dir=CHECKPOINT_DIR,
        patience=3,
    )


if __name__ == "__main__":
    main()
