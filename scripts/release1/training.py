"""Stage 2 of the Release-1 pipeline: dual-stream backbone training, spatial
ConvNeXt-baseline training, and macro-AUC checkpoint evaluation on validation.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from scripts.release1.common import (
    get_sample_stratum,
    seed_worker,
    subsample_balanced,
    subsample_stratified_val,
)
from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import get_transforms
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.training.ema import ExponentialMovingAverage
from src.training.optimization import create_scheduler, get_differential_param_groups

logger = logging.getLogger("release1_runner")

def evaluate_macro_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[float, float, float]:
    """Compute macro-AUC on the validation split as 0.5 * AUC_ffpp + 0.5 * AUC_celeb.

    Degenerate branches (empty/single-class, or an exact 0.5 AUC despite two classes)
    fall back to 0.5 and emit a warning so the fallback is never silent.
    """
    model.eval()
    backup = ema.apply_shadow(model) if ema is not None else None

    all_preds: list[float] = []
    all_targets: list[int] = []
    all_valid: list[bool] = []
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
                all_valid.extend((valid_flags.view(-1) > 0.0).cpu().numpy().tolist())
    finally:
        if ema is not None and backup is not None:
            ema.restore(model, backup)

    val_samples = val_loader.dataset.samples  # type: ignore[attr-defined]
    for s in val_samples:
        stratum = get_sample_stratum(s[0], int(s[1]))
        all_sources.append("celeb" if stratum.startswith("celeb") else "ffpp")

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    if len(all_valid) == len(all_sources):
        sources_arr = np.array(all_sources)[np.array(all_valid)]
    else:
        sources_arr = np.array(all_sources[: len(preds_arr)])

    # Compute source-specific AUCs
    ffpp_mask = sources_arr == "ffpp"
    celeb_mask = sources_arr == "celeb"

    auc_ffpp = 0.5
    if np.sum(ffpp_mask) > 0 and len(np.unique(targets_arr[ffpp_mask])) >= 2:
        auc_ffpp = float(roc_auc_score(targets_arr[ffpp_mask], preds_arr[ffpp_mask]))
    else:
        logger.warning(
            "evaluate_macro_val: FF++ branch degenerate (n=%d, classes=%d); using 0.5 fallback.",
            int(np.sum(ffpp_mask)),
            len(np.unique(targets_arr[ffpp_mask])) if np.sum(ffpp_mask) else 0,
        )

    auc_celeb = 0.5
    if np.sum(celeb_mask) > 0 and len(np.unique(targets_arr[celeb_mask])) >= 2:
        auc_celeb = float(roc_auc_score(targets_arr[celeb_mask], preds_arr[celeb_mask]))
        if auc_celeb == 0.5:
            logger.warning(
                "evaluate_macro_val: Celeb-DF AUC is exactly 0.5 despite two classes (n=%d); "
                "treating the branch as degenerate and reporting the 0.5 fallback.",
                int(np.sum(celeb_mask)),
            )
    else:
        logger.warning(
            "evaluate_macro_val: Celeb-DF branch degenerate (n=%d, classes=%d); using 0.5 fallback. "
            "Macro-AUC is effectively FF++-only.",
            int(np.sum(celeb_mask)),
            len(np.unique(targets_arr[celeb_mask])) if np.sum(celeb_mask) else 0,
        )

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
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> str:
    """Trains ConvNeXt-Small spatial stream alone under the identical 4-way balanced sampling protocol."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

