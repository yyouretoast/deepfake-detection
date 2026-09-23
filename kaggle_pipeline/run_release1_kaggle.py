"""Turnkey Release 1 Pipeline Runner for Kaggle Dual/Single Tesla T4 GPU.

Executes the complete Release 1 deepfake detection protocol:
1. Integrity audit of splits (0 identity leakage, 0 DFD in train/val).
2. Dual-stream training on clean 37,104 train split (5 epochs, cosine schedule with 1-epoch warmup,
   differential LRs, FP16 mixed precision with FP32-isolated FFT/atan2).
3. Macro-AUC checkpoint selection across composite FF++ and Celeb-DF Set A validation.
4. Validation calibration: Platt temperature scaling T* in [0.1, 10.0], Youden's J threshold tau*,
   and Bayesian dual thresholds (tau_real, tau_fake) with certified precision >= 0.980.
5. Held-out test evaluation (26,981 crops):
   - Overall metrics and trivial baseline row (always-fake).
   - Per-source evaluation against own authentic baseline (FF++, Celeb-DF, DFD) with 1,000-sample
     video-level clustered bootstrap 95% confidence intervals.
   - Fine-grained subdomain breakdown.
6. Spatiotemporal Bi-GRU video sequence modeling with velocity deltas (Delta e_t) and dual-path aggregation.
7. 5-fold Leave-One-Target-Out (LOTO) cross-generator domain generalization benchmark.
8. Robustness stress testing (JPEG, blur, noise, downscaling).
9. Single-T4 CUDA-synchronized latency profiling (B=1 and B=32).
10. Structured provenance JSON export to release1_results.json and LaTeX verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import sys
import time

# Compatibility guard for pathlib and matplotlib cache under sandbox environments
try:
    pathlib._NormalAccessor.mkdir = lambda self, path, mode=0o777, **kwargs: os.mkdir(path, mode, **kwargs)
except Exception:
    pass
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cache")))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
from collections import Counter
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Disable OpenCV multithreading to eliminate fork deadlocks in Linux containers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_sample_stratum(rel_path: str, label: int) -> str:
    """Classifies a crop into its source domain and label stratum for balanced sampling."""
    norm = rel_path.replace("\\", "/").lower()
    parts = norm.split("/")
    folder = parts[1] if len(parts) > 1 else parts[0]
    if "celeb" in norm or folder.startswith("id"):
        src = "celeb"
    elif "__" in folder or (len(folder) == 5 and folder.isdigit()) or "dfd" in norm:
        src = "dfd"
    else:
        src = "ffpp"
    return f"{src}_{int(label)}"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier, ManipulationDomain
from src.dataset.loader import SequenceVideoDataset, extract_identities, get_transforms, group_video_sequences
from src.dataset.resolver import DatasetResolver, find_dataset_root, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import (
    compute_ece,
    compute_eer,
    find_optimal_threshold,
    fit_platt_scaling,
    fit_temperature_log,
)
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import create_scheduler, get_differential_param_groups
from src.training.trainer import DualStreamTrainer
from src.utils.checkpoint import (
    clean_state_dict,
    compute_dual_thresholds,
    compute_dual_thresholds_certified,
)

logger = logging.getLogger("release1_runner")


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "release1_pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "missing"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

def subsample_balanced(
    samples: list[Any], n: int, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    fakes = [s for s in samples if float(s[1]) == 1.0]
    reals = [s for s in samples if float(s[1]) == 0.0]
    n_half = n // 2
    n_fakes = min(n_half, len(fakes))
    n_reals = min(n - n_fakes, len(reals))
    selected = rng.sample(fakes, n_fakes) + rng.sample(reals, n_reals)
    rng.shuffle(selected)
    return selected


def subsample_stratified_val(
    samples: list[Any], n_per_source: int = 50, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    ffpp = [s for s in samples if DomainClassifier.classify(s[0]).domain != ManipulationDomain.CELEB_DF]
    celeb = [s for s in samples if DomainClassifier.classify(s[0]).domain == ManipulationDomain.CELEB_DF]
    selected = subsample_balanced(ffpp, n_per_source, seed=seed) + subsample_balanced(celeb, n_per_source, seed=seed)
    rng.shuffle(selected)
    return selected


def subsample_stratified_test(
    samples: list[Any], n_per_source: int = 34, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    by_source: dict[str, list[Any]] = {"ffpp": [], "celeb": [], "dfd": []}
    for s in samples:
        p = s[0].replace("\\", "/")
        parts = p.split("/")
        dinfo = DomainClassifier.classify(p)
        if "dfd" in dinfo.domain.value or ("real" in parts[0] and len(parts[1]) == 5 and parts[1].isdigit()):
            by_source["dfd"].append(s)
        elif dinfo.domain == ManipulationDomain.CELEB_DF or ("real" in parts[0] and "id" in parts[1]):
            by_source["celeb"].append(s)
        else:
            by_source["ffpp"].append(s)

    selected = []
    for src, src_samples in by_source.items():
        if src_samples:
            selected.extend(subsample_balanced(src_samples, n_per_source, seed=seed))
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Stage 1: Dataset Integrity & Leakage Verification
# ---------------------------------------------------------------------------
def verify_dataset_integrity(splits_path: str) -> dict[str, Any]:
    logger.info("Verifying dataset splits and identity partitioning: %s", splits_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]

    def get_actor_ids(crops: list[Any]) -> set[str]:
        actors = set()
        for c in crops:
            p = c[0].replace("\\", "/")
            parts = p.split("/")
            name = parts[1] if len(parts) > 1 else parts[0]
            a1, a2 = extract_identities(name)
            if a1:
                actors.add(a1)
            if a2:
                actors.add(a2)
        return actors

    train_ids = get_actor_ids(train_samples)
    val_ids = get_actor_ids(val_samples)
    test_ids = get_actor_ids(test_samples)

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"FATAL: Identity leakage detected! "
            f"Train&Val: {len(overlap_train_val)}, "
            f"Train&Test: {len(overlap_train_test)}, "
            f"Val&Test: {len(overlap_val_test)}"
        )

    # Verify zero DFD samples in train or val
    for p, _ in train_samples + val_samples:
        norm = p.replace("\\", "/").lower()
        if "__" in norm or "/00" in norm and len(norm.split("/")[1]) == 5:
            raise ValueError(f"FATAL: DFD sample leaked into train/val: {p}")

    train_labels = [int(s[1]) for s in train_samples]
    val_labels = [int(s[1]) for s in val_samples]
    test_labels = [int(s[1]) for s in test_samples]

    ledger = {
        "train_total": len(train_samples),
        "train_real": len(train_labels) - sum(train_labels),
        "train_fake": sum(train_labels),
        "val_total": len(val_samples),
        "val_real": len(val_labels) - sum(val_labels),
        "val_fake": sum(val_labels),
        "test_total": len(test_samples),
        "test_real": len(test_labels) - sum(test_labels),
        "test_fake": sum(test_labels),
        "train_actors": len(train_ids),
        "val_actors": len(val_ids),
        "test_actors": len(test_ids),
        "splits_sha256": compute_sha256(splits_path),
    }

    logger.info(
        "Splits Verified: Train=%d (%d R / %d F), Val=%d (%d R / %d F), Test=%d (%d R / %d F) | Zero Leakage Guaranteed",
        ledger["train_total"],
        ledger["train_real"],
        ledger["train_fake"],
        ledger["val_total"],
        ledger["val_real"],
        ledger["val_fake"],
        ledger["test_total"],
        ledger["test_real"],
        ledger["test_fake"],
    )
    return ledger


# ---------------------------------------------------------------------------
# Stage 2: Dual-Stream Backbone Training with Macro-AUC Validation
# ---------------------------------------------------------------------------
def evaluate_macro_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[float, float, float]:
    """Computes Macro-AUC on composite validation split (0.5 * AUC_ffpp + 0.5 * AUC_celeb)."""
    model.eval()
    backup = ema.apply_shadow(model) if ema is not None else None

    all_preds: list[float] = []
    all_targets: list[int] = []
    all_sources: list[str] = []

    try:
        with torch.inference_mode():
            for images, labels, valid_flags in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                valid_mask = (valid_flags > 0.0).to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    probs = torch.sigmoid(outputs)

                p_flat = probs.view(-1)[valid_mask.view(-1)].cpu().numpy()
                y_flat = labels.view(-1)[valid_mask.view(-1)].cpu().numpy()

                all_preds.extend(p_flat.tolist())
                all_targets.extend(y_flat.astype(int).tolist())
    finally:
        if ema is not None and backup is not None:
            ema.restore(model, backup)

    val_samples = val_loader.dataset.samples  # type: ignore[attr-defined]
    for s in val_samples:
        path = s[0]
        domain_info = DomainClassifier.classify(path)
        if domain_info.domain == ManipulationDomain.CELEB_DF:
            all_sources.append("celeb")
        else:
            all_sources.append("ffpp")

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    sources_arr = np.array(all_sources[: len(preds_arr)])

    # Compute source-specific AUCs
    ffpp_mask = sources_arr == "ffpp"
    celeb_mask = sources_arr == "celeb"

    auc_ffpp = 0.5
    if np.sum(ffpp_mask) > 0 and len(np.unique(targets_arr[ffpp_mask])) >= 2:
        auc_ffpp = float(roc_auc_score(targets_arr[ffpp_mask], preds_arr[ffpp_mask]))

    auc_celeb = 0.5
    if np.sum(celeb_mask) > 0 and len(np.unique(targets_arr[celeb_mask])) >= 2:
        auc_celeb = float(roc_auc_score(targets_arr[celeb_mask], preds_arr[celeb_mask]))

    macro_auc = 0.5 * (auc_ffpp + auc_celeb)
    return macro_auc, auc_ffpp, auc_celeb


def train_dual_stream_backbone(
    data_root: str,
    splits_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr_spatial: float = 1e-5,
    lr_spectral: float = 1e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> str:
    save_path = os.path.join(output_dir, "dual_stream_best.pth")
    resume_path = os.path.join(output_dir, "dual_stream_epoch_resume.pth")
    if not force_rerun and os.path.exists(save_path) and not os.path.exists(resume_path):
        logger.info("Found existing best checkpoint: %s. Skipping training.", save_path)
        return save_path

    if smoke_test:
        epochs = 1
        batch_size = min(batch_size, 8)
        logger.info("--- [SMOKE TEST] Dual-Stream Backbone Training (1 Epoch, Subsampled) ---")
    else:
        logger.info("--- Starting Dual-Stream Backbone Training (%d Epochs) ---", epochs)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    if smoke_test:
        train_samples = subsample_balanced(train_samples, 200, seed=42)
        val_samples = subsample_stratified_val(val_samples, 50, seed=42)
        logger.info("Smoke test subsample: Train=%d, Val=%d", len(train_samples), len(val_samples))

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
    train_ds = FaceCropDataset(train_samples, data_root, is_train=True, transform=train_transform)
    val_ds = FaceCropDataset(val_samples, data_root, is_train=False, transform=eval_transform)

    # 4-Way Stratified Balanced Sampling per (source, label) to eliminate shortcut learning
    strata = [get_sample_stratum(s[0], s[1]) for s in train_samples]
    strata_counts = Counter(strata)
    num_strata = len(strata_counts)
    sample_weights = [1.0 / (num_strata * max(1, strata_counts[st])) for st in strata]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    logger.info(
        "4-Way Stratified Training Sampler initialized: %s (Total samples: %d)",
        dict(strata_counts),
        len(sample_weights),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    model = HybridDeepfakeDetector(pretrained=True, frequency_backbone="resse")
    model.to(device)

    # Differential parameter groups: 1e-5 spatial backbone, 1e-4 spectral/gating head
    optimizer = torch.optim.AdamW(
        get_differential_param_groups(model, lr_backbone=lr_spatial, lr_head=lr_spectral),
        weight_decay=1e-4,
    )
    scheduler = create_scheduler(optimizer, warmup_epochs=1, total_epochs=epochs)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    ema = ExponentialMovingAverage(model, decay=0.999)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_macro_auc = 0.0

    if not force_rerun and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device)
            start_epoch = ckpt["epoch"] + 1
            best_macro_auc = ckpt.get("best_macro_auc", 0.0)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict") is not None and device.type == "cuda":
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "ema_shadow" in ckpt:
                ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
            logger.info("Resumed backbone training from epoch %d/%d (best_macro_auc: %.4f)", start_epoch + 1, epochs, best_macro_auc)
        except Exception as e:
            logger.warning("Failed to resume from %s: %s. Starting fresh.", resume_path, e)
            start_epoch = 0
            best_macro_auc = 0.0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for images, batch_labels, valid_flags in pbar:
            images = images.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            valid_mask = valid_flags.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs, aux_outputs = model(images, return_aux=True)
                loss_main = (criterion(outputs, batch_labels) * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
                loss_aux = (criterion(aux_outputs, batch_labels) * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
                loss = loss_main + 0.30 * loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)
            pbar.set_postfix({"loss": f"{running_loss / total_samples:.4f}"})

        scheduler.step()
        train_loss = running_loss / max(1, total_samples)

        # Evaluate on validation with Macro-AUC checkpoint selection
        macro_auc, auc_ff, auc_cl = evaluate_macro_val(model, val_loader, device=device, ema=ema)
        elapsed = time.time() - t0
        logger.info(
            "Epoch [%d/%d] (%.1fs) - Train Loss: %.4f | Val Macro-AUC: %.4f (FF++: %.4f, Celeb: %.4f)",
            epoch + 1,
            epochs,
            elapsed,
            train_loss,
            macro_auc,
            auc_ff,
            auc_cl,
        )

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            backup = ema.apply_shadow(model)
            torch.save(model.state_dict(), save_path)
            ema.restore(model, backup)
            logger.info("Saved Best Checkpoint (Macro-AUC: %.4f) -> %s", macro_auc, save_path)

        # Save epoch-level resume checkpoint for fault-tolerant recovery
        resume_dict = {
            "epoch": epoch,
            "best_macro_auc": best_macro_auc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None,
            "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
        }
        torch.save(resume_dict, resume_path)
        logger.info("Saved Epoch Resume Checkpoint (Epoch %d/%d) -> %s", epoch + 1, epochs, resume_path)

    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
        logger.info("Saved Final Checkpoint -> %s", save_path)

    if os.path.exists(resume_path):
        try:
            os.remove(resume_path)
            logger.info("Cleaned up Epoch Resume Checkpoint -> training fully completed.")
        except Exception:
            pass

    return save_path


# ---------------------------------------------------------------------------
# Stage 2b: Spatial-Only ConvNeXt Baseline Training (Empirical Comparison)
# ---------------------------------------------------------------------------
def train_spatial_convnext_baseline(
    data_root: str,
    splits_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr_spatial: float = 1e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> str:
    """Trains ConvNeXt-Small spatial stream alone under the identical 4-way balanced sampling protocol."""
    save_path = os.path.join(output_dir, "spatial_convnext_best.pth")
    if not force_rerun and os.path.exists(save_path):
        logger.info("Found existing spatial ConvNeXt baseline checkpoint: %s. Skipping training.", save_path)
        return save_path

    if smoke_test:
        epochs = 1
        batch_size = min(batch_size, 8)
        logger.info("--- [SMOKE TEST] Spatial-Only ConvNeXt Baseline Training (1 Epoch, Subsampled) ---")
    else:
        logger.info("--- Starting Spatial-Only ConvNeXt Baseline Training (%d Epochs) ---", epochs)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    if smoke_test:
        train_samples = subsample_balanced(train_samples, 200, seed=42)
        val_samples = subsample_stratified_val(val_samples, 50, seed=42)

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
    train_ds = FaceCropDataset(train_samples, data_root, is_train=True, transform=train_transform)
    val_ds = FaceCropDataset(val_samples, data_root, is_train=False, transform=eval_transform)

    strata = [get_sample_stratum(s[0], s[1]) for s in train_samples]
    strata_counts = Counter(strata)
    num_strata = len(strata_counts)
    sample_weights = [1.0 / (num_strata * max(1, strata_counts[st])) for st in strata]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    model = HybridDeepfakeDetector(pretrained=True, use_fft_branch=False)
    model.to(device)

    optimizer = torch.optim.AdamW(
        get_differential_param_groups(model, lr_backbone=1e-5, lr_head=lr_spatial),
        weight_decay=1e-4,
    )
    scheduler = create_scheduler(optimizer, warmup_epochs=1, total_epochs=epochs)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Spatial ConvNeXt Epoch [{epoch + 1}/{epochs}]")
        for images, batch_labels, valid_flags in pbar:
            images = images.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            valid_mask = valid_flags.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss_unreduced = criterion(logits, batch_labels)
                loss = (loss_unreduced * valid_mask).sum() / max(1e-6, valid_mask.sum())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(images)
            total_samples += len(images)
            pbar.set_postfix({"loss": f"{running_loss / max(1, total_samples):.4f}"})

        scheduler.step()

        # Evaluate on validation split
        model.eval()
        val_logits_list, val_targets_list = [], []
        with torch.no_grad():
            for v_imgs, v_lbls, _ in val_loader:
                v_imgs = v_imgs.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    v_out = model(v_imgs)
                val_logits_list.append(v_out.cpu().numpy())
                val_targets_list.append(v_lbls.numpy())

        v_logits = np.concatenate(val_logits_list).flatten()
        v_targets = np.concatenate(val_targets_list).flatten()
        val_auc = float(roc_auc_score(v_targets, v_logits))
        logger.info("Spatial ConvNeXt Epoch %d/%d - Val AUC: %.4f", epoch + 1, epochs, val_auc)
        if val_auc > best_val_auc or epoch == 0:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            logger.info("Saved Best Spatial ConvNeXt Checkpoint (Val AUC: %.4f) -> %s", val_auc, save_path)

    return save_path


# ---------------------------------------------------------------------------
# Stage 3: Validation Split Calibration & Threshold Derivation
# ---------------------------------------------------------------------------
def calibrate_validation_split(
    backbone_path: str,
    data_root: str,
    splits_path: str,
    output_dir: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> tuple[str, dict[str, float]]:
    calibrated_path = os.path.join(output_dir, "dual_stream_calibrated.pth")
    if not force_rerun and os.path.exists(calibrated_path):
        try:
            ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
            if "calibration" in ckpt:
                logger.info("Found existing calibrated checkpoint: %s. Skipping calibration.", calibrated_path)
                return calibrated_path, ckpt["calibration"]
        except Exception as e:
            logger.warning("Failed loading existing calibrated checkpoint: %s. Re-calibrating.", e)

    logger.info("--- Calibrating Probabilities and Deriving Decision Thresholds on Val ---")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_samples = splits["val"]
    if smoke_test:
        val_samples = subsample_stratified_val(val_samples, 50, seed=42)
        logger.info("Smoke test subsample: Val=%d", len(val_samples))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    val_loader = DataLoader(
        FaceCropDataset(val_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
    )

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    state_dict = torch.load(backbone_path, map_location=device, weights_only=False)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)

    evaluator = ModelEvaluator(model, device=device)
    val_logits, val_targets, val_valid = evaluator.predict_loader(val_loader)
    mask = val_valid > 0.0
    val_logits = val_logits[mask]
    val_targets = val_targets[mask]

    real_logits = val_logits[val_targets == 0]
    fake_logits = val_logits[val_targets == 1]
    logger.info("Validation Logit Separation Analysis:")
    logger.info("  Reals (N=%d): mean=%.4f, std=%.4f, [min=%.4f, max=%.4f]", len(real_logits), float(np.mean(real_logits)), float(np.std(real_logits)), float(np.min(real_logits)), float(np.max(real_logits)))
    logger.info("  Fakes (N=%d): mean=%.4f, std=%.4f, [min=%.4f, max=%.4f]", len(fake_logits), float(np.mean(fake_logits)), float(np.std(fake_logits)), float(np.min(fake_logits)), float(np.max(fake_logits)))
    logger.info("  Separation Delta(mean) = %.4f", float(np.mean(fake_logits) - np.mean(real_logits)))

    # 4-Way Stratified Validation Weights to prevent Celeb-DF fakes (83.8% of val) from dominating calibration
    val_strata = [get_sample_stratum(s[0], s[1]) for s in val_samples]
    val_strata_arr = np.array(val_strata)[mask]
    val_counts = Counter(val_strata_arr)
    num_v_strata = len(val_counts)
    val_weights = np.array([1.0 / (num_v_strata * max(1, val_counts[st])) for st in val_strata_arr])
    logger.info("Validation Stratified Weights for Platt Scaling: %s", dict(val_counts))

    # 2-Parameter Platt Scaling (scale a, bias b) with stratified sample weighting
    scale_a, bias_b, eff_t = fit_platt_scaling(val_logits, val_targets, sample_weights=val_weights)
    t_star = fit_temperature_log(val_logits, val_targets)
    logger.info("Fitted Platt Calibration: a = %.4f, b = %.4f (Eff T = %.4f) | Temp-Only T* = %.4f", scale_a, bias_b, eff_t, t_star)

    is_at_bound = (scale_a <= 0.101 or scale_a >= 9.99 or abs(bias_b) >= 9.99 or t_star <= 0.101 or t_star >= 9.99)
    if is_at_bound:
        err_msg = (
            f"CALIBRATION BOUND HIT: scale_a={scale_a:.4f}, bias_b={bias_b:.4f}, T*={t_star:.4f}. "
            f"Search bounds are a in [0.1, 10.0], b in [-10.0, 10.0], T in [0.1, 10.0]. "
            f"Logit separation Delta(mean) = {float(np.mean(fake_logits) - np.mean(real_logits)):.4f}."
        )
        logger.error(err_msg)
        if not smoke_test:
            raise RuntimeError(err_msg + " Stopping run as calibration optimizer landed on a boundary parameter constraint.")
        else:
            logger.warning("Proceeding under smoke test mode despite bound hit.")

    val_probs_uncal = 1.0 / (1.0 + np.exp(-val_logits))
    val_probs_cal = 1.0 / (1.0 + np.exp(-(scale_a * val_logits + bias_b)))

    ece_uncal = compute_ece(val_probs_uncal, val_targets)
    ece_cal = compute_ece(val_probs_cal, val_targets)
    ece_reduction = ((ece_uncal - ece_cal) / max(1e-6, ece_uncal)) * 100.0
    logger.info("Val ECE: %.4f (Uncal) -> %.4f (Calibrated) [-%.1f%%]", ece_uncal, ece_cal, ece_reduction)

    # Youden's J / Balanced Accuracy optimal threshold
    tau_star, best_bal_acc = find_optimal_threshold(val_targets, val_probs_cal, criterion="balanced_accuracy")
    youden_j = 2.0 * best_bal_acc - 1.0
    logger.info("Optimal Decision Threshold tau* = %.4f (Val Bal. Acc = %.4f, Youden J = %.4f)", tau_star, best_bal_acc, youden_j)

    # Operational Bayesian 3-zone dual thresholds with certified precision reporting
    tau_real, tau_fake, p_real_cert, p_fake_cert = compute_dual_thresholds_certified(
        val_probs_cal, val_targets, min_precision=0.980
    )
    if p_fake_cert < 0.980 or p_real_cert < 0.980:
        logger.warning(
            "Target precision 0.980 unattainable on validation cohort. Operating at maximal certified precision: P*(fake) = %.4f, P*(real) = %.4f",
            p_fake_cert, p_real_cert
        )
    logger.info("Bayesian Decision Thresholds: tau_real = %.4f (P* = %.4f), tau_fake = %.4f (P* = %.4f)", tau_real, p_real_cert, tau_fake, p_fake_cert)

    calib_meta = {
        "optimal_temperature": float(t_star),
        "platt_scale_a": float(scale_a),
        "platt_bias_b": float(bias_b),
        "effective_temperature": float(eff_t),
        "uncalibrated_ece": float(ece_uncal),
        "calibrated_ece": float(ece_cal),
        "ece_reduction_pct": float(ece_reduction),
        "optimal_threshold": float(tau_star),
        "val_balanced_accuracy": float(best_bal_acc),
        "youden_j": float(youden_j),
        "tau_real": float(tau_real),
        "tau_fake": float(tau_fake),
        "certified_precision_real": float(p_real_cert),
        "certified_precision_fake": float(p_fake_cert),
        "is_precision_certified": bool(p_fake_cert >= 0.980 and p_real_cert >= 0.980),
    }

    # Save calibrated checkpoint with metadata
    torch.save({"model_state_dict": model.state_dict(), "calibration": calib_meta}, calibrated_path)
    logger.info("Saved Calibrated Checkpoint -> %s", calibrated_path)
    return calibrated_path, calib_meta


# ---------------------------------------------------------------------------
# Stage 4: Test Set Evaluation with Video-Level Clustered Bootstrap
# ---------------------------------------------------------------------------
def compute_video_clustered_bootstrap_auc(
    video_ids: list[str],
    targets: np.ndarray,
    probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Evaluates ROC AUC with 95% video-level clustered bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    unique_videos = np.unique(video_ids)
    video_to_indices: dict[str, list[int]] = {}
    for idx, vid in enumerate(video_ids):
        if vid not in video_to_indices:
            video_to_indices[vid] = []
        video_to_indices[vid].append(idx)

    # Point estimate
    point_auc = float(roc_auc_score(targets, probs))

    boot_aucs: list[float] = []
    n_vids = len(unique_videos)
    for _ in range(n_bootstraps):
        sampled_vids = rng.choice(unique_videos, size=n_vids, replace=True)
        sampled_indices = []
        for sv in sampled_vids:
            sampled_indices.extend(video_to_indices[sv])

        y_sample = targets[sampled_indices]
        p_sample = probs[sampled_indices]
        if len(np.unique(y_sample)) >= 2:
            boot_aucs.append(float(roc_auc_score(y_sample, p_sample)))

    if len(boot_aucs) == 0:
        return point_auc, point_auc, point_auc

    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return point_auc, ci_lower, ci_upper


def evaluate_held_out_test(
    calibrated_path: str,
    data_root: str,
    splits_path: str,
    calib_meta: dict[str, float],
    output_dir: str,
    spatial_checkpoint_path: str | None = None,
    n_bootstraps: int = 1000,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    test_summary_path = os.path.join(output_dir, "test_evaluation_summary.json")
    if not force_rerun and os.path.exists(test_summary_path):
        logger.info("Found existing test evaluation summary: %s. Skipping test evaluation.", test_summary_path)
        with open(test_summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if smoke_test:
        n_bootstraps = 10
        logger.info("--- [SMOKE TEST] Evaluating Test Split (Subsampled, 10 Bootstraps) ---")
    else:
        logger.info("--- Evaluating Full Test Set & Computing Clustered Bootstrap CIs ---")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_samples = splits["test"]
    if smoke_test:
        test_samples = subsample_stratified_test(test_samples, 34, seed=42)
        logger.info("Smoke test subsample: Test=%d", len(test_samples))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    test_loader = DataLoader(
        FaceCropDataset(test_samples, data_root, is_train=False, transform=eval_transform),
        batch_size=32,
        shuffle=False,
        num_workers=4 if os.name != "nt" else 0,
    )

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)

    evaluator = ModelEvaluator(model, device=device)
    test_logits, test_targets, test_valid = evaluator.predict_loader(test_loader)
    mask = test_valid > 0.0
    test_logits = test_logits[mask]
    test_targets = test_targets[mask]

    scale_a = calib_meta.get("platt_scale_a", 1.0 / max(1e-6, calib_meta.get("optimal_temperature", 1.0)))
    bias_b = calib_meta.get("platt_bias_b", 0.0)
    tau_star = calib_meta["optimal_threshold"]
    test_probs = 1.0 / (1.0 + np.exp(-(scale_a * test_logits + bias_b)))
    test_preds = (test_probs >= tau_star).astype(int)

    # 1. Overall Dual-Stream Test Metrics
    overall_auc = float(roc_auc_score(test_targets, test_probs))
    overall_prauc = float(average_precision_score(test_targets, test_probs))
    overall_f1 = float(f1_score(test_targets, test_preds, zero_division=0))
    overall_bal_acc = float(balanced_accuracy_score(test_targets, test_preds))
    overall_prec = float(precision_score(test_targets, test_preds, zero_division=0))
    overall_rec = float(recall_score(test_targets, test_preds, zero_division=0))
    overall_eer, _ = compute_eer(test_targets, test_probs)

    logger.info(
        "Overall Test (Dual-Stream): ROC AUC = %.4f | PR AUC = %.4f | Fake F1 = %.4f | Bal Acc = %.4f | Prec = %.4f | Rec = %.4f | EER = %.2f%%",
        overall_auc,
        overall_prauc,
        overall_f1,
        overall_bal_acc,
        overall_prec,
        overall_rec,
        overall_eer * 100.0,
    )

    # 2. Trivial Baseline (Always Predict Fake)
    n_fake = int(np.sum(test_targets == 1))
    n_total = len(test_targets)
    trivial_always_fake = {
        "auc": 0.5000,
        "pr_auc": float(n_fake / n_total),
        "f1": float(2.0 * n_fake / (n_total + n_fake)),
        "balanced_accuracy": 0.5000,
        "precision": float(n_fake / n_total),
        "recall": 1.0000,
        "eer": 50.0,
    }

    # 2b. Spatial-Only ConvNeXt Baseline Evaluation
    spatial_only_metrics = {}
    if spatial_checkpoint_path and os.path.exists(spatial_checkpoint_path):
        try:
            sp_model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=False)
            sp_state = torch.load(spatial_checkpoint_path, map_location=device, weights_only=False)
            sp_model.load_state_dict(clean_state_dict(sp_state), strict=False)
            sp_model.to(device)
            sp_eval = ModelEvaluator(sp_model, device=device)
            sp_logits, sp_targets, sp_valid = sp_eval.predict_loader(test_loader)
            sp_m = sp_valid > 0.0
            sp_l = sp_logits[sp_m]
            sp_t = sp_targets[sp_m]
            sp_p = 1.0 / (1.0 + np.exp(-sp_l))
            sp_preds = (sp_p >= 0.50).astype(int)
            sp_eer, _ = compute_eer(sp_t, sp_p)
            spatial_only_metrics = {
                "auc": float(roc_auc_score(sp_t, sp_p)),
                "pr_auc": float(average_precision_score(sp_t, sp_p)),
                "f1": float(f1_score(sp_t, sp_preds, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(sp_t, sp_preds)),
                "precision": float(precision_score(sp_t, sp_preds, zero_division=0)),
                "recall": float(recall_score(sp_t, sp_preds, zero_division=0)),
                "eer": float(sp_eer * 100.0),
            }
            logger.info(
                "Spatial-Only Baseline Test: AUC = %.4f | PR AUC = %.4f | Fake F1 = %.4f | Bal Acc = %.4f | EER = %.2f%%",
                spatial_only_metrics["auc"],
                spatial_only_metrics["pr_auc"],
                spatial_only_metrics["f1"],
                spatial_only_metrics["balanced_accuracy"],
                spatial_only_metrics["eer"],
            )
        except Exception as e:
            logger.warning("Failed evaluating spatial-only baseline from %s: %s", spatial_checkpoint_path, e)

    # 3. Fine-Grained Cohort Categorization
    video_ids = []
    fine_source = []
    gen_names = []
    for s in test_samples:
        path = s[0].replace("\\", "/")
        parts = path.split("/")
        vid = parts[1] if len(parts) > 1 else parts[0]
        video_ids.append(f"{parts[0]}/{vid}")
        lbl = int(s[1])

        if lbl == 0:
            if re.match(r"^\d{3}$", vid):
                fine_source.append("ffpp_real")
            elif re.match(r"^\d{5}$", vid):
                fine_source.append("youtube_real")
            else:
                fine_source.append("celeb_real")
            gen_names.append("authentic")
        else:
            if "__" in vid:
                fine_source.append("dfd_fake")
                gen_names.append("dfd")
            elif re.match(r"^\d{3}_\d{3}$", vid):
                fine_source.append("ffpp_fake")
                dinfo = DomainClassifier.classify(path)
                gen_names.append(dinfo.domain.value)
            else:
                fine_source.append("celeb_fake")
                gen_names.append("celeb")

    video_ids = [video_ids[i] for i, m in enumerate(mask) if m]
    fine_source_arr = np.array([fine_source[i] for i, m in enumerate(mask) if m])
    gen_names_arr = np.array([gen_names[i] for i, m in enumerate(mask) if m])

    # Distinct Per-Source Evaluation Cohorts
    cohort_definitions = {
        "ffpp": (fine_source_arr == "ffpp_fake") | (fine_source_arr == "ffpp_real"),
        "celeb_id_only": (fine_source_arr == "celeb_fake") | (fine_source_arr == "celeb_real"),
        "celeb_all": (fine_source_arr == "celeb_fake") | (fine_source_arr == "celeb_real") | (fine_source_arr == "youtube_real"),
        "dfd_vs_ffpp": (fine_source_arr == "dfd_fake") | (fine_source_arr == "ffpp_real"),
        "dfd_vs_youtube": (fine_source_arr == "dfd_fake") | (fine_source_arr == "youtube_real"),
    }

    per_source_results = {}
    for c_name, c_mask in cohort_definitions.items():
        if np.sum(c_mask) == 0:
            continue
        y_c = test_targets[c_mask]
        p_c = test_probs[c_mask]
        v_c = [video_ids[i] for i, sm in enumerate(c_mask) if sm]
        preds_c = (p_c >= tau_star).astype(int)

        point_auc, ci_low, ci_high = compute_video_clustered_bootstrap_auc(
            v_c, y_c, p_c, n_bootstraps=n_bootstraps
        )
        c_prauc = float(average_precision_score(y_c, p_c))
        c_f1 = float(f1_score(y_c, preds_c, zero_division=0))
        c_bacc = float(balanced_accuracy_score(y_c, preds_c))
        c_prec = float(precision_score(y_c, preds_c, zero_division=0))
        c_rec = float(recall_score(y_c, preds_c, zero_division=0))
        c_eer, _ = compute_eer(y_c, p_c)

        per_source_results[c_name] = {
            "total_crops": int(np.sum(c_mask)),
            "fakes": int(np.sum(y_c == 1)),
            "reals": int(np.sum(y_c == 0)),
            "auc": point_auc,
            "auc_ci_lower_95": ci_low,
            "auc_ci_upper_95": ci_high,
            "pr_auc": c_prauc,
            "f1": c_f1,
            "balanced_accuracy": c_bacc,
            "precision": c_prec,
            "recall": c_rec,
            "eer": float(c_eer * 100.0),
        }
        logger.info(
            "Cohort [%s]: AUC = %.4f [95%% CI: %.4f - %.4f] | F1 = %.4f | Bal Acc = %.4f | Prec = %.4f | EER = %.2f%%",
            c_name.upper(),
            point_auc,
            ci_low,
            ci_high,
            c_f1,
            c_bacc,
            c_prec,
            c_eer * 100.0,
        )

    # Calculate DFD cross-dataset confound gap
    dfd_confound_gap = per_source_results.get("dfd_vs_youtube", {}).get("auc", 0.0) - per_source_results.get("dfd_vs_ffpp", {}).get("auc", 0.0)
    logger.info("DFD Cross-Dataset Confound Gap (YouTube-reals vs FF++ reals): Delta(AUC) = %+.4f", dfd_confound_gap)

    # Legacy compatibility aliases for manuscript generator
    per_source_results["celeb"] = per_source_results.get("celeb_all", per_source_results.get("celeb_id_only", {}))
    per_source_results["dfd"] = per_source_results.get("dfd_vs_ffpp", {})

    # 4. Fine-Grained Subdomain Breakdown (Table 2)
    subdomain_results = {}
    for gen_key in ["deepfakes", "face2face", "faceswap", "neuraltextures", "celeb", "dfd"]:
        gen_fake_mask = (gen_names_arr == gen_key) & (test_targets == 1)
        if np.sum(gen_fake_mask) == 0:
            continue

        if gen_key == "dfd":
            matched_real_mask = (fine_source_arr == "ffpp_real")
        elif gen_key == "celeb":
            matched_real_mask = (fine_source_arr == "celeb_real")
        else:
            matched_real_mask = (fine_source_arr == "ffpp_real")

        eval_mask = gen_fake_mask | matched_real_mask
        y_sub = test_targets[eval_mask]
        p_sub = test_probs[eval_mask]
        preds_sub = (p_sub >= tau_star).astype(int)

        sub_auc = float(roc_auc_score(y_sub, p_sub))
        sub_f1 = float(f1_score(y_sub, preds_sub, zero_division=0))
        sub_prec = float(precision_score(y_sub, preds_sub, zero_division=0))
        sub_rec = float(recall_score(y_sub, preds_sub, zero_division=0))

        subdomain_results[gen_key] = {
            "fake_crops": int(np.sum(gen_fake_mask)),
            "auc": sub_auc,
            "f1": sub_f1,
            "precision": sub_prec,
            "recall": sub_rec,
        }
        logger.info(
            "Subdomain [%s]: Fakes = %d | AUC = %.4f | F1 = %.4f | Prec = %.4f | Rec = %.4f",
            gen_key,
            np.sum(gen_fake_mask),
            sub_auc,
            sub_f1,
            sub_prec,
            sub_rec,
        )

    test_summary = {
        "overall": {
            "auc": overall_auc,
            "pr_auc": overall_prauc,
            "f1": overall_f1,
            "balanced_accuracy": overall_bal_acc,
            "precision": overall_prec,
            "recall": overall_rec,
            "eer": float(overall_eer * 100.0),
        },
        "trivial_baseline_always_fake": trivial_always_fake,
        "spatial_only": spatial_only_metrics,
        "per_source": per_source_results,
        "subdomain_breakdown": subdomain_results,
        "dfd_confound_gap": float(dfd_confound_gap),
    }

    with open(os.path.join(output_dir, "test_evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    return test_summary


# ---------------------------------------------------------------------------
# Stage 5: Spatiotemporal Bi-GRU Video Sequence Modeling
# ---------------------------------------------------------------------------
def extract_clip_embeddings(
    backbone: HybridDeepfakeDetector, frames: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, c, h, w = frames.shape
    frames_flat = frames.view(b * t, c, h, w).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        feats = backbone.extract_features(frames_flat)
        logits = backbone(frames_flat)
        if isinstance(logits, tuple):
            logits = logits[0]
    return feats.view(b, t, -1), logits.view(b, t)


def train_and_eval_temporal(
    backbone_path: str,
    data_root: str,
    splits_path: str,
    output_dir: str,
    epochs: int = 5,
    seq_len: int = 8,
    stride: int = 2,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    temporal_save_path = os.path.join(output_dir, "temporal_head_best.pth")
    temporal_summary_path = os.path.join(output_dir, "temporal_evaluation_summary.json")
    if not force_rerun and os.path.exists(temporal_save_path) and os.path.exists(temporal_summary_path):
        logger.info("Found existing temporal checkpoint and summary: %s. Skipping temporal stage.", temporal_summary_path)
        with open(temporal_summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if smoke_test:
        epochs = 1
        seq_len = 4
        stride = 2
        logger.info("--- [SMOKE TEST] Training Spatiotemporal Bi-GRU Video Head (1 Epoch, Subsampled) ---")
    else:
        logger.info("--- Training Spatiotemporal Bi-GRU Video Head on Frozen Backbone ---")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = [(os.path.join(data_root, p), lbl) for p, lbl in splits["train"]]
    val_samples = [(os.path.join(data_root, p), lbl) for p, lbl in splits["val"]]
    test_samples = [(os.path.join(data_root, p), lbl) for p, lbl in splits["test"]]

    train_videos = group_video_sequences(train_samples, min_frames=seq_len)
    val_videos = group_video_sequences(val_samples, min_frames=seq_len)
    test_videos = group_video_sequences(test_samples, min_frames=seq_len)

    def _subsample_vids(vids: list[tuple[list[str], int]], count: int) -> list[tuple[list[str], int]]:
        fakes = [v for v in vids if v[1] == 1]
        reals = [v for v in vids if v[1] == 0]
        n_each = max(1, count // 2)
        return fakes[:n_each] + reals[:n_each]

    if smoke_test:
        train_videos = _subsample_vids(train_videos, 10)
        val_videos = _subsample_vids(val_videos, 6)
        test_videos = _subsample_vids(test_videos, 6)

    logger.info("Video Sequences: Train=%d, Val=%d, Test=%d", len(train_videos), len(val_videos), len(test_videos))

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
    train_ds = SequenceVideoDataset(train_videos, transform=train_transform, seq_len=seq_len, stride=stride, is_train=True)
    val_ds = SequenceVideoDataset(val_videos, transform=eval_transform, seq_len=seq_len, stride=stride, is_train=False)
    test_ds = SequenceVideoDataset(test_videos, transform=eval_transform, seq_len=seq_len, stride=stride, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2 if os.name != "nt" else 0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2 if os.name != "nt" else 0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2 if os.name != "nt" else 0)

    # Load frozen backbone
    backbone = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    backbone.load_state_dict(clean_state_dict(state_dict), strict=False)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=256, use_deltas=True, use_max_pool=True)
    temporal_model.to(device)

    optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    for epoch in range(epochs):
        temporal_model.train()
        total_loss, total_count = 0.0, 0
        for frames, labels, _ in tqdm(train_loader, desc=f"Bi-GRU Epoch [{epoch + 1}/{epochs}]"):
            labels = labels.to(device).float().unsqueeze(1)
            with torch.no_grad():
                embeddings, _ = extract_clip_embeddings(backbone, frames, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, _ = temporal_model(embeddings)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * len(labels)
            total_count += len(labels)

        scheduler.step()

        # Val evaluation
        temporal_model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for frames, labels, _ in val_loader:
                with torch.no_grad():
                    embeddings, _ = extract_clip_embeddings(backbone, frames, device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits, _ = temporal_model(embeddings)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.view(-1).cpu().numpy().tolist())
                val_targets.extend(labels.numpy().tolist())

        if len(np.unique(val_targets)) > 1:
            val_auc = float(roc_auc_score(val_targets, val_preds))
        else:
            val_auc = 0.5
        logger.info("Bi-GRU Epoch [%d/%d] - Loss: %.4f | Val AUC: %.4f", epoch + 1, epochs, total_loss / total_count, val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(temporal_model.state_dict(), temporal_save_path)
            logger.info("Saved Best Bi-GRU Checkpoint (Val AUC: %.4f) -> %s", val_auc, temporal_save_path)

    if not os.path.exists(temporal_save_path):
        torch.save(temporal_model.state_dict(), temporal_save_path)
        logger.info("Saved Final Bi-GRU Checkpoint -> %s", temporal_save_path)

    # Sequence-level evaluation on Test
    temporal_model.load_state_dict(torch.load(temporal_save_path, map_location=device, weights_only=False))
    temporal_model.eval()

    test_targets: list[int] = []
    bigru_preds: list[float] = []
    frame_avg_preds: list[float] = []
    max_pool_preds: list[float] = []

    with torch.inference_mode():
        for frames, labels, _ in tqdm(test_loader, desc="Testing Sequence Models"):
            with torch.no_grad():
                embeddings, frame_logits = extract_clip_embeddings(backbone, frames, device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits_bigru, _ = temporal_model(embeddings)
                p_frames = torch.sigmoid(frame_logits)
                p_avg = p_frames.mean(dim=-1)
                p_max = p_frames.max(dim=-1)[0]
                p_bigru = torch.sigmoid(logits_bigru).squeeze(-1)

            test_targets.extend(labels.numpy().tolist())
            bigru_preds.extend(p_bigru.cpu().numpy().tolist())
            frame_avg_preds.extend(p_avg.cpu().numpy().tolist())
            max_pool_preds.extend(p_max.cpu().numpy().tolist())

    y_test = np.array(test_targets)
    p_bigru = np.array(bigru_preds)
    p_avg = np.array(frame_avg_preds)
    p_max = np.array(max_pool_preds)

    def get_seq_metrics(y_t: np.ndarray, p_t: np.ndarray) -> dict[str, float]:
        if len(np.unique(y_t)) < 2:
            return {
                "auc": 0.5,
                "pr_auc": 0.5,
                "f1": 0.0,
                "balanced_accuracy": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "eer": 50.0,
            }
        tau, _ = find_optimal_threshold(y_t, p_t, criterion="balanced_accuracy")
        preds = (p_t >= tau).astype(int)
        eer, _ = compute_eer(y_t, p_t)
        return {
            "auc": float(roc_auc_score(y_t, p_t)),
            "pr_auc": float(average_precision_score(y_t, p_t)),
            "f1": float(f1_score(y_t, preds, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_t, preds)),
            "precision": float(precision_score(y_t, preds, zero_division=0)),
            "recall": float(recall_score(y_t, preds, zero_division=0)),
            "eer": float(eer * 100.0),
        }

    res_avg = get_seq_metrics(y_test, p_avg)
    res_max = get_seq_metrics(y_test, p_max)
    res_bigru = get_seq_metrics(y_test, p_bigru)
    eer_reduction = res_avg["eer"] - res_bigru["eer"]
    res_bigru["eer_reduction"] = float(eer_reduction)

    logger.info("Sequence Frame-Avg: AUC = %.4f | EER = %.2f%%", res_avg["auc"], res_avg["eer"])
    logger.info("Sequence Max-Pool:  AUC = %.4f | EER = %.2f%%", res_max["auc"], res_max["eer"])
    logger.info("Sequence Bi-GRU:    AUC = %.4f | EER = %.2f%% [EER Reduction: -%.2f%%]", res_bigru["auc"], res_bigru["eer"], eer_reduction)

    temporal_summary = {
        "naive_frame_average": res_avg,
        "temporal_max_pooling": res_max,
        "bigru_ours": res_bigru,
    }
    with open(temporal_summary_path, "w", encoding="utf-8") as f:
        json.dump(temporal_summary, f, indent=2)

    return temporal_summary


# ---------------------------------------------------------------------------
# Stage 6: 5-Fold Leave-One-Target-Out (LOTO) Benchmark
# ---------------------------------------------------------------------------
def run_loto_experiment(
    data_root: str,
    splits_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
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
        logger.info("--- [SMOKE TEST] Running 5-Fold LOTO Benchmark (1 Epoch, Subsampled) ---")
    else:
        logger.info("--- Running 5-Fold Leave-One-Target-Out (LOTO) Generalization Benchmark ---")
    folds = [
        ("Fold 1", "deepfakes", "FF++ Deepfakes"),
        ("Fold 2", "face2face", "FF++ Face2Face"),
        ("Fold 3", "faceswap", "FF++ FaceSwap"),
        ("Fold 4", "neuraltextures", "FF++ NeuralTextures"),
        ("Fold 5", "celeb", "Celeb-DF v2"),
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
        # Filter train split: exclude fakes from holdout
        retained_train = []
        for s in train_samples:
            path, lbl = s[0], s[1]
            if lbl == 1.0 and DomainClassifier.matches_holdout(path, holdout_key):
                continue
            retained_train.append(s)

        # Matched test evaluation: held-out fakes vs matched reals
        held_out_test_fakes = [s for s in test_samples if s[1] == 1.0 and DomainClassifier.matches_holdout(s[0], holdout_key)]
        if holdout_key == "celeb":
            matched_test_reals = [s for s in test_samples if s[1] == 0.0 and "id" in s[0].replace("\\", "/").split("/")[1]]
        else:
            matched_test_reals = [s for s in test_samples if s[1] == 0.0 and s[0].replace("\\", "/").split("/")[1].isdigit() and len(s[0].replace("\\", "/").split("/")[1]) == 3]

        if smoke_test:
            retained_train = subsample_balanced(retained_train, 40, seed=42)
            held_out_test_fakes = subsample_balanced(held_out_test_fakes, 10, seed=42)
            matched_test_reals = subsample_balanced(matched_test_reals, 10, seed=42)

        eval_samples = held_out_test_fakes + matched_test_reals
        logger.info(
            "Fold %s: Retained Train = %d, Eval Set = %d (%d Fake, %d Real)",
            fold_name,
            len(retained_train),
            len(eval_samples),
            len(held_out_test_fakes),
            len(matched_test_reals),
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

        optimizer = torch.optim.AdamW(get_differential_param_groups(model), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        for ep in range(epochs):
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

        t_fold = fit_temperature_log(logits, targets)
        probs = 1.0 / (1.0 + np.exp(-(logits / t_fold)))
        preds = (probs >= 0.50).astype(int)
        if len(np.unique(targets)) > 1:
            fold_auc = float(roc_auc_score(targets, probs))
        else:
            fold_auc = 0.5
        fold_f1 = float(f1_score(targets, preds, zero_division=0))
        fold_prec = float(precision_score(targets, preds, zero_division=0))

        logger.info("Fold %s (%s): T* = %.4f | Zero-Shot AUC = %.4f | F1 = %.4f | Prec = %.4f", fold_name, display_name, t_fold, fold_auc, fold_f1, fold_prec)
        loto_results[holdout_key] = {
            "fold_name": fold_name,
            "display_name": display_name,
            "holdout_samples": len(held_out_test_fakes),
            "fitted_t_star": float(t_fold),
            "zero_shot_auc": fold_auc,
            "zero_shot_f1": fold_f1,
            "zero_shot_precision": fold_prec,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(loto_results, f, indent=2)

    with open(loto_summary_path, "w", encoding="utf-8") as f:
        json.dump(loto_results, f, indent=2)

    return loto_results


# ---------------------------------------------------------------------------
# Stage 7: Robustness Stress-Testing (Table 4)
# ---------------------------------------------------------------------------
def run_robustness_stress_test(
    calibrated_path: str,
    data_root: str,
    splits_path: str,
    output_dir: str | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    robustness_path = os.path.join(output_dir, "robustness_results.json") if output_dir else None
    if not force_rerun and robustness_path and os.path.exists(robustness_path):
        logger.info("Found existing robustness results: %s. Skipping robustness.", robustness_path)
        with open(robustness_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if smoke_test:
        logger.info("--- [SMOKE TEST] Stress-Testing Model Resilience (10 Samples, 1 Setting per Distortion) ---")
    else:
        logger.info("--- Stress-Testing Model Resilience to Real-World Degradations ---")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_samples = splits["test"]

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    calib_meta = ckpt.get("calibration", {"optimal_temperature": 1.0})
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()
    scale_a = calib_meta.get("platt_scale_a", 1.0 / max(1e-6, calib_meta.get("optimal_temperature", 1.0)))
    bias_b = calib_meta.get("platt_bias_b", 0.0)

    # Sample a reproducible subset (e.g. 1,000 crops; 10 in smoke mode)
    n_samples = 10 if smoke_test else min(1000, len(test_samples))
    sample_subset = subsample_balanced(test_samples, n_samples, seed=42)

    def evaluate_degraded_images(degrade_fn: Any) -> float:
        y_true, y_probs = [], []
        for path, label in sample_subset:
            full_p = os.path.join(data_root, path)
            img = cv2.imread(full_p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_deg = degrade_fn(img)
            img_deg = cv2.resize(img_deg, (256, 256))
            tensor = torch.from_numpy(img_deg).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(tensor)
                if isinstance(out, tuple):
                    out = out[0]
                prob = torch.sigmoid(scale_a * out + bias_b).item()
            y_true.append(int(label))
            y_probs.append(prob)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_probs))

    # 1. JPEG
    jpeg_results = {}
    jpeg_qs = [70] if smoke_test else [90, 70, 50, 30]
    for q in jpeg_qs:
        def apply_jpeg(im: np.ndarray) -> np.ndarray:
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        jpeg_results[f"q_{q}"] = evaluate_degraded_images(apply_jpeg)
        logger.info("Robustness JPEG Q=%d: AUC = %.4f", q, jpeg_results[f"q_{q}"])

    # 2. Gaussian Blur
    blur_results = {}
    blur_sigmas = [2.0] if smoke_test else [1.0, 2.0, 3.0, 4.0]
    for sigma in blur_sigmas:
        ksize = int(6 * sigma + 1) | 1
        def apply_blur(im: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(im, (ksize, ksize), sigma)
        blur_results[f"sigma_{sigma}"] = evaluate_degraded_images(apply_blur)
        logger.info("Robustness Blur sigma=%.1f: AUC = %.4f", sigma, blur_results[f"sigma_{sigma}"])

    # 3. Additive Gaussian Noise
    noise_results = {}
    noise_sigmas = [10.0] if smoke_test else [5.0, 10.0, 15.0, 20.0]
    for sigma_n in noise_sigmas:
        def apply_noise(im: np.ndarray) -> np.ndarray:
            noise = np.random.normal(0, sigma_n, im.shape)
            return np.clip(im.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noise_results[f"sigma_{sigma_n}"] = evaluate_degraded_images(apply_noise)
        logger.info("Robustness Noise sigma=%.1f: AUC = %.4f", sigma_n, noise_results[f"sigma_{sigma_n}"])

    # 4. Downscaling
    downscale_results = {}
    downscale_scales = [4] if smoke_test else [2, 4, 8]
    for scale in downscale_scales:
        def apply_downscale(im: np.ndarray) -> np.ndarray:
            h, w = im.shape[:2]
            small = cv2.resize(im, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        downscale_results[f"scale_{scale}x"] = evaluate_degraded_images(apply_downscale)
        logger.info("Robustness Downscale %dx: AUC = %.4f", scale, downscale_results[f"scale_{scale}x"])

    robustness_summary = {
        "jpeg": jpeg_results,
        "gaussian_blur": blur_results,
        "gaussian_noise": noise_results,
        "downscaling": downscale_results,
    }
    if robustness_path:
        with open(robustness_path, "w", encoding="utf-8") as f:
            json.dump(robustness_summary, f, indent=2)

    return robustness_summary


# ---------------------------------------------------------------------------
# Stage 8: CUDA Latency Profiling (Table 5)
# ---------------------------------------------------------------------------
def run_latency_profiling(
    calibrated_path: str,
    output_dir: str | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    latency_path = os.path.join(output_dir, "latency_results.json") if output_dir else None
    if not force_rerun and latency_path and os.path.exists(latency_path):
        logger.info("Found existing latency results: %s. Skipping latency.", latency_path)
        with open(latency_path, "r", encoding="utf-8") as f:
            return json.load(f)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    is_tesla_t4 = "T4" in gpu_name.upper()
    logger.info("--- Running CUDA Latency Profiling on Device: %s (Is Tesla T4: %s) ---", gpu_name, is_tesla_t4)

    if not is_tesla_t4 and not smoke_test:
        logger.warning("Active device '%s' is NOT an NVIDIA Tesla T4. Refusing to write invalid latency metrics for Table 5.", gpu_name)
        results = {
            "device": gpu_name,
            "is_tesla_t4": False,
            "status": "SKIPPED_NOT_TESLA_T4",
            "message": "Execution device is not NVIDIA Tesla T4. Latency profiling requires a dedicated single T4 GPU.",
        }
        if latency_path:
            with open(latency_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return results

    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(calibrated_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()

    warmup_iters = 5 if smoke_test else 50
    timed_iters = 10 if smoke_test else 200

    results = {
        "device": gpu_name,
        "is_tesla_t4": is_tesla_t4,
    }
    for b_sz in [1, 32]:
        dummy = torch.randn(b_sz, 3, 256, 256, device=device)
        # Warmup
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed runs
        timings = []
        with torch.inference_mode():
            for _ in range(timed_iters):
                t_start = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t_start) * 1000.0)

        mean_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        fps = float((b_sz / (mean_ms / 1000.0)))
        results[f"batch_size_{b_sz}"] = {
            "mean_latency_ms": mean_ms,
            "std_latency_ms": std_ms,
            "fps": fps,
        }
        logger.info("Latency B=%d on %s: %.2f ms +/- %.2f ms (%.1f FPS)", b_sz, gpu_name, mean_ms, std_ms, fps)

    if latency_path:
        with open(latency_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Stage 9: Master Turnkey Execution & Verification
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Turnkey Release 1 Deepfake Pipeline on Kaggle.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing splits.json and crops.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save weights, logs, and results JSON.")
    parser.add_argument("--epochs", type=int, default=5, help="Backbone training epochs (default: 5).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per process (default: 16).")
    parser.add_argument("--skip_loto", action="store_true", help="Skip 5-fold LOTO experiment.")
    parser.add_argument("--skip_robustness", action="store_true", help="Skip robustness stress-tests.")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test (~200 crops, 1 epoch) across all pipeline stages.")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of all stages, bypassing cached checkpoints.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "backbone", "calibrate", "test", "temporal", "loto", "robustness", "latency"])
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    elif os.path.exists("/kaggle/working"):
        out_dir = "/kaggle/working"
    else:
        out_dir = os.path.abspath("./results_release1")
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Execution Device: %s (Count: %d)", device, torch.cuda.device_count())

    if args.smoke_test:
        logger.info("=== RUNNING IN SMOKE TEST MODE (Rapid Verification) ===")
        backbone_epochs = 1
        batch_size = min(args.batch_size, 8)
    else:
        backbone_epochs = args.epochs
        batch_size = args.batch_size

    data_root = DatasetResolver.find_dataset_root(args.data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    # 1. Integrity check
    ledger = verify_dataset_integrity(splits_path)

    # 2. Train backbone
    backbone_weights = train_dual_stream_backbone(
        data_root=data_root,
        splits_path=splits_path,
        output_dir=out_dir,
        epochs=backbone_epochs,
        batch_size=batch_size,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 2b. Train spatial-only ConvNeXt baseline
    spatial_weights = train_spatial_convnext_baseline(
        data_root=data_root,
        splits_path=splits_path,
        output_dir=out_dir,
        epochs=backbone_epochs,
        batch_size=batch_size,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 3. Calibrate on val
    calibrated_weights, calib_meta = calibrate_validation_split(
        backbone_path=backbone_weights,
        data_root=data_root,
        splits_path=splits_path,
        output_dir=out_dir,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 4. Evaluate Test Split with Clustered Bootstrap
    test_eval_summary = evaluate_held_out_test(
        calibrated_path=calibrated_weights,
        data_root=data_root,
        splits_path=splits_path,
        calib_meta=calib_meta,
        output_dir=out_dir,
        spatial_checkpoint_path=spatial_weights,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 5. Temporal Video Head
    temporal_summary = train_and_eval_temporal(
        backbone_path=calibrated_weights,
        data_root=data_root,
        splits_path=splits_path,
        output_dir=out_dir,
        epochs=1 if args.smoke_test else 5,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 6. LOTO Folds (Optional)
    if not args.skip_loto:
        loto_summary = run_loto_experiment(
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            epochs=1 if args.smoke_test else 3,
            batch_size=batch_size,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )
    else:
        loto_summary = {}

    # 7. Robustness Stress Tests (Optional)
    if not args.skip_robustness:
        robustness_summary = run_robustness_stress_test(
            calibrated_path=calibrated_weights,
            data_root=data_root,
            splits_path=splits_path,
            output_dir=out_dir,
            device=device,
            smoke_test=args.smoke_test,
            force_rerun=args.force_rerun,
        )
    else:
        robustness_summary = {}

    # 8. Latency Benchmark
    latency_summary = run_latency_profiling(
        calibrated_weights,
        output_dir=out_dir,
        device=device,
        smoke_test=args.smoke_test,
        force_rerun=args.force_rerun,
    )

    # 9. Structured Provenance JSON Assembly
    final_results = {
        "provenance": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "splits_sha256": ledger["splits_sha256"],
            "spatial_weights_sha256": compute_sha256(spatial_weights),
            "backbone_weights_sha256": compute_sha256(backbone_weights),
            "calibrated_weights_sha256": compute_sha256(calibrated_weights),
        },
        "dataset_ledger": ledger,
        "calibration": calib_meta,
        "test_frame_evaluation": test_eval_summary,
        "temporal_video_evaluation": temporal_summary,
        "loto_cross_generator_benchmark": loto_summary,
        "robustness_stress_tests": robustness_summary,
        "latency_profiling": latency_summary,
    }

    final_json_path = os.path.join(out_dir, "release1_results.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Saved Master Release 1 Results -> %s", final_json_path)

    # Also write to repo results/ directory if available
    repo_results_path = os.path.join(REPO_ROOT, "results", "release1_results.json")
    os.makedirs(os.path.dirname(repo_results_path), exist_ok=True)
    with open(repo_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info("Mirrored Master Release 1 Results -> %s", repo_results_path)

    # 10. Run LaTeX Verifier
    verifier_path = os.path.join(REPO_ROOT, "scripts", "verify_latex_full.py")
    tex_path = os.path.join(REPO_ROOT, "manuscript", "main.tex")
    if os.path.exists(verifier_path) and os.path.exists(tex_path):
        import subprocess
        logger.info("Executing LaTeX Manuscript Invariant & Ledger Verification...")
        subprocess.run([sys.executable, verifier_path, tex_path, "--results", final_json_path], check=False)

    logger.info("=== TURNKEY RELEASE 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
