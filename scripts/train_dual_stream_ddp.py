"""
Dual-Stream Deepfake Detector Training Pipeline.
Trains ConvNeXt-Small (Spatial) + Depthwise SRM/Bayar 2D FFT (Frequency) via DistributedDataParallel.

Usage:
    accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu train_accelerate.py
"""

import os
import sys

# Ensure repository root is on sys.path for DDP spawned subprocesses
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import time
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from accelerate import Accelerator
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.dataset.loader import dedupe_split
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5
T_MAX_TOTAL = 15
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_HEAD = 1e-3
CHECKPOINT_STATE_DIR = '/kaggle/working/checkpoint_state'
BEST_MODEL_WEIGHTS_PATH = '/kaggle/working/dual_stream_best.pth'


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        except Exception:
            valid_flag = 0.0
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        if self.is_train and random.random() > 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(valid_flag, dtype=torch.float32)


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
        {'params': backbone_decay, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': 1e-4},
        {'params': backbone_nodecay, 'lr': LEARNING_RATE_BACKBONE, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': LEARNING_RATE_HEAD, 'weight_decay': 1e-4},
        {'params': head_nodecay, 'lr': LEARNING_RATE_HEAD, 'weight_decay': 0.0}
    ]


def main():
    accelerator = Accelerator(mixed_precision='fp16')
    data_root = find_dataset_root()

    if accelerator.is_main_process:
        logging.info(f"Verified Dataset Root: {data_root}")

    splits_path = os.path.join(data_root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    train_samples = dedupe_split(splits['train'])
    val_samples = dedupe_split(splits['val'])

    if accelerator.is_main_process:
        logging.info(f"Deduplicated Splits Loaded — Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    train_ds = KaggleFastDataset(train_samples, data_root, is_train=True)
    val_ds = KaggleFastDataset(val_samples, data_root, is_train=False)

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
        logging.info(f"Class Distribution — Real: {num_real}, Fake: {num_fake} | Calculated pos_weight: {pos_weight_val:.4f}")

    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)

    model = HybridDeepfakeDetector()
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX_TOTAL, eta_min=1e-6)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    start_epoch = 0
    best_val_auc = 0.0

    if os.path.exists(BEST_MODEL_WEIGHTS_PATH):
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(BEST_MODEL_WEIGHTS_PATH, map_location='cpu'))
            if accelerator.is_main_process:
                logging.info(f"Loaded existing model weights from {BEST_MODEL_WEIGHTS_PATH}")
        except Exception as e:
            if accelerator.is_main_process:
                logging.warning(f"Could not load model weights: {e}")

    if accelerator.is_main_process:
        logging.info(f"Starting DDP Run: Epochs {start_epoch + 1} to {NUM_EPOCHS}...")

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = torch.tensor(0.0, device=accelerator.device)
        failed_reads_tensor = torch.tensor(0.0, device=accelerator.device)

        for images, labels, valid_flags in tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            labels = labels.unsqueeze(1)
            valid_flags = valid_flags.unsqueeze(1)
            failed_reads_tensor += (1.0 - valid_flags).sum()
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
        total_train_failures = int(accelerator.reduce(failed_reads_tensor, reduction='sum').item())

        model.eval()
        val_loss_tensor = torch.tensor(0.0, device=accelerator.device)
        val_failures_tensor = torch.tensor(0.0, device=accelerator.device)
        all_targets, all_preds = [], []

        with torch.no_grad():
            for images, labels, valid_flags in val_loader:
                labels = labels.unsqueeze(1)
                valid_flags = valid_flags.unsqueeze(1)
                val_failures_tensor += (1.0 - valid_flags).sum()

                outputs = model(images)
                loss_unreduced = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight_tensor, reduction='none')
                loss = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
                val_loss_tensor += loss.detach() * images.size(0)

                preds_g, targets_g = accelerator.gather_for_metrics((torch.sigmoid(outputs), labels))
                all_preds.extend(preds_g.cpu().numpy())
                all_targets.extend(targets_g.cpu().numpy())

        total_val_loss = accelerator.reduce(val_loss_tensor, reduction='sum').item()
        val_loss = total_val_loss / len(val_loader.dataset)
        total_val_failures = int(accelerator.reduce(val_failures_tensor, reduction='sum').item())

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            val_auc = 0.5

        if accelerator.is_main_process:
            if total_train_failures > 0:
                logging.warning(f"Epoch {epoch+1} Train Read Failures: {total_train_failures}")
            if total_val_failures > 0:
                logging.warning(f"Epoch {epoch+1} Val Read Failures: {total_val_failures}")

            current_lr_head = optimizer.param_groups[2]['lr']
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Head LR: {current_lr_head:.6f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), BEST_MODEL_WEIGHTS_PATH)
                logging.info(f"Saved Checkpoint (Val AUC: {val_auc:.4f}) to {BEST_MODEL_WEIGHTS_PATH}")

    if accelerator.is_main_process:
        total_mins = (time.time() - start_time) / 60
        logging.info(f"Training Complete in {total_mins:.2f} mins. Peak Val AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    main()
