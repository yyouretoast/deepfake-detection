"""
Dual-Stream Deepfake Detector Training Pipeline.
Trains ConvNeXt-Small (Spatial) + Depthwise SRM/Bayar 2D FFT (Frequency) via DistributedDataParallel.

Usage:
    accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu train_accelerate.py
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Optional

# Ensure repository root is on sys.path for DDP spawned subprocesses
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.dataset.loader import dedupe_split, get_transforms  # noqa: E402
from src.models.hybrid_detector import HybridDeepfakeDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3  # stop if val AUC does not improve for this many epochs
GRAD_ACCUM_STEPS = 4  # simulate effective batch size of 64 (16 * 4) without extra memory
T_MAX_TOTAL = 15
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_HEAD = 1e-3
CHECKPOINT_STATE_DIR = os.getenv("CHECKPOINT_STATE_DIR", "./models/checkpoint_state")
BEST_MODEL_WEIGHTS_PATH = os.getenv("BEST_MODEL_WEIGHTS_PATH", "./models/dual_stream_best.pth")


class FocalLossWithLogits(nn.Module):
    """Focal Loss for binary classification with unreduced mask support."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        return focal_weight * bce_loss


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) of model parameters for smoother validation inference."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(42)


class KaggleFastDataset(Dataset):
    def __init__(self, samples, root_dir, is_train=True, transform=None):
        self.samples = samples
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

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

            if self.transform is not None:
                augmented = self.transform(image=rgb)
                aug_tensor = augmented["image"]
                tensor = aug_tensor.float() / 255.0 if aug_tensor.dtype == torch.uint8 else aug_tensor.float()
            else:
                if self.is_train and random.random() > 0.5:
                    rgb = np.ascontiguousarray(np.fliplr(rgb))
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        except (OSError, ValueError, cv2.error):
            valid_flag = 0.0
            tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)

        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(valid_flag, dtype=torch.float32)


def find_dataset_root(custom_dir: Optional[str] = None) -> str:
    """Locate dataset directory containing splits.json."""
    candidates = []
    if custom_dir:
        candidates.append(custom_dir)
    env_dir = os.getenv("DATASET_ROOT")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        "./data/cropped",
        "./data",
        "/kaggle/working/local_crops",
        "/kaggle/input/deepfake-face-crops-256/deepfake_crops_512",
        "/kaggle/input/datasets/yassinyasserr/deepfake-face-crops-256/deepfake_crops_512",
        "/kaggle/input/deepfake-face-crops-256",
        "/kaggle/input/datasets/yassinyasserr/deepfake-dataset/deepfake_crops_512",
        "/kaggle/input/deepfake-dataset/deepfake_crops_512",
        "/kaggle/input/datasets/yassinyasserr/deepfake-crops-512/deepfake_crops_512",
        "/kaggle/input/deepfake-crops-512/deepfake_crops_512",
        "/kaggle/input/deepfake_crops_512",
    ])
    def is_valid_root(p: str) -> bool:
        if not p or not os.path.exists(os.path.join(p, "splits.json")):
            return False
        return os.path.exists(os.path.join(p, "fake")) or os.path.exists(os.path.join(p, "real"))

    for p in candidates:
        if is_valid_root(p):
            return p
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "splits.json")):
            return p
    if os.path.exists("/kaggle/input"):
        for r, d, f in os.walk("/kaggle/input"):
            if "splits.json" in f and ("fake" in d or "real" in d):
                return r
        for r, d, f in os.walk("/kaggle/input"):
            if "splits.json" in f:
                return r
    raise FileNotFoundError("Could not locate dataset containing splits.json. Specify via --data_dir or DATASET_ROOT environment variable.")


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
        {'params': backbone_decay, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': 1e-4},
        {'params': backbone_nodecay, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': LEARNING_RATE_HEAD, 'weight_decay': 1e-4},
        {'params': head_nodecay, 'lr': LEARNING_RATE_HEAD, 'weight_decay': 0.0}
    ]


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Stream Deepfake Detector DDP")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json and cropped dataset")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    data_root = find_dataset_root(args.data_dir)

    if accelerator.is_main_process:
        logger.info(f"Verified Dataset Root: {data_root}")
        os.makedirs(os.path.dirname(BEST_MODEL_WEIGHTS_PATH), exist_ok=True)
        os.makedirs(CHECKPOINT_STATE_DIR, exist_ok=True)

    splits_path = os.path.join(data_root, 'splits.json')
    if os.path.exists('/kaggle/working/splits.json'):
        splits_path = '/kaggle/working/splits.json'
    elif os.path.exists('./splits.json'):
        splits_path = './splits.json'

    if accelerator.is_main_process:
        logger.info(f"Loading splits from: {splits_path}")

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    train_samples = dedupe_split(splits['train'])
    val_samples = dedupe_split(splits['val'])

    if accelerator.is_main_process:
        logger.info(f"Deduplicated Splits Loaded — Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    train_transform, eval_transform = get_transforms(img_size=IMG_SIZE)
    train_ds = KaggleFastDataset(train_samples, data_root, is_train=True, transform=train_transform)
    val_ds = KaggleFastDataset(val_samples, data_root, is_train=False, transform=eval_transform)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g
    )

    num_fake = sum(1 for s in train_samples if s[1] == 1)
    num_real = len(train_samples) - num_fake
    pos_weight_val = num_real / max(1, num_fake)

    if accelerator.is_main_process:
        logger.info(f"Class Distribution — Real: {num_real}, Fake: {num_fake} | Calculated pos_weight: {pos_weight_val:.4f}")

    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight_tensor)

    model = HybridDeepfakeDetector()
    ema = ExponentialMovingAverage(model, decay=0.999)
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX_TOTAL, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[1]
    )

    # SyncBatchNorm: synchronises batch norm running statistics across all DDP processes.
    # Without this, each GPU computes its own BN stats from a sub-batch, which severely
    # degrades convergence when per-GPU batch sizes are small (e.g. 16).
    if accelerator.num_processes > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    start_epoch = 0
    best_val_auc = 0.0
    epochs_without_improvement = 0  # early-stopping counter

    if os.path.exists(BEST_MODEL_WEIGHTS_PATH):
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(BEST_MODEL_WEIGHTS_PATH, map_location='cpu'))
            if accelerator.is_main_process:
                logger.info(f"Loaded existing model weights from {BEST_MODEL_WEIGHTS_PATH}")
        except (OSError, RuntimeError, ValueError) as e:
            if accelerator.is_main_process:
                logger.warning(f"Could not load model weights: {e}")

    if accelerator.is_main_process:
        logger.info(f"Starting DDP Run: Epochs {start_epoch + 1} to {NUM_EPOCHS}...")

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = torch.tensor(0.0, device=accelerator.device)
        failed_reads_tensor = torch.tensor(0.0, device=accelerator.device)

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, (images, labels, valid_flags) in train_pbar:
            labels = labels.unsqueeze(1)
            valid_flags = valid_flags.unsqueeze(1)
            failed_reads_tensor += (1.0 - valid_flags).sum()

            # Gradient accumulation: accumulate over GRAD_ACCUM_STEPS micro-batches
            # before updating weights. This simulates a larger effective batch size
            # without requiring more GPU memory per step.
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(images)
                    loss_unreduced = criterion(outputs, labels)
                    loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    ema.update(accelerator.unwrap_model(model))

            running_loss += loss.detach() * images.size(0)

            if accelerator.is_main_process and (step + 1) % 200 == 0:
                current_avg_loss = (running_loss / ((step + 1) * images.size(0) * accelerator.num_processes)).item()
                logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step+1}/{len(train_loader)}] - Current Loss: {current_avg_loss:.4f}")
                sys.stdout.flush()

        scheduler.step()

        total_train_loss = accelerator.reduce(running_loss, reduction='sum').item()
        train_loss = total_train_loss / len(train_loader.dataset)
        total_train_failures = int(accelerator.reduce(failed_reads_tensor, reduction='sum').item())

        model.eval()
        val_shadow_backup = ema.apply_shadow(accelerator.unwrap_model(model))
        val_loss_tensor = torch.tensor(0.0, device=accelerator.device)
        val_failures_tensor = torch.tensor(0.0, device=accelerator.device)
        all_targets, all_preds = [], []

        with torch.no_grad():
            for images, labels, valid_flags in val_loader:
                labels = labels.unsqueeze(1)
                valid_flags = valid_flags.unsqueeze(1)
                val_failures_tensor += (1.0 - valid_flags).sum()

                outputs = model(images)
                loss_unreduced = criterion(outputs, labels)
                loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
                val_loss_tensor += loss.detach() * images.size(0)

                preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels))
                all_preds.extend(preds_g.cpu().numpy())
                all_targets.extend(targets_g.cpu().numpy())

        ema.restore(accelerator.unwrap_model(model), val_shadow_backup)

        total_val_loss = accelerator.reduce(val_loss_tensor, reduction='sum').item()
        val_loss = total_val_loss / len(val_loader.dataset)
        total_val_failures = int(accelerator.reduce(val_failures_tensor, reduction='sum').item())

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except (ValueError, TypeError, RuntimeError):
            val_auc = 0.5

        if accelerator.is_main_process:
            if total_train_failures > 0:
                logger.warning(f"Epoch {epoch+1} Train Read Failures: {total_train_failures}")
            if total_val_failures > 0:
                logger.warning(f"Epoch {epoch+1} Val Read Failures: {total_val_failures}")

            current_lr_head = optimizer.param_groups[2]['lr']
            logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Head LR: {current_lr_head:.6f}")

            if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_without_improvement = 0
                    val_shadow_backup = ema.apply_shadow(accelerator.unwrap_model(model))
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), BEST_MODEL_WEIGHTS_PATH)
                    ema.restore(accelerator.unwrap_model(model), val_shadow_backup)
                    logger.info(f"Saved Checkpoint (Val AUC: {val_auc:.4f}) to {BEST_MODEL_WEIGHTS_PATH}")
            else:
                    epochs_without_improvement += 1
                    logger.info(
                        f"No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs "
                        f"(best Val AUC: {best_val_auc:.4f})"
                    )
                    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                        logger.info(
                            f"Early stopping triggered after {epoch+1} epochs "
                            f"(patience={EARLY_STOPPING_PATIENCE}, best Val AUC={best_val_auc:.4f})."
                        )
                        break

    if accelerator.is_main_process:
        total_mins = (time.time() - start_time) / 60
        logger.info(f"Training Complete in {total_mins:.2f} mins. Peak Val AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    main()
