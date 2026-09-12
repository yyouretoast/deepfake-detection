"""Surgical Fold 3 crash diagnostic with synchronous CUDA and per-operation logging.

This script reproduces the EXACT same execution path as train_loto_experiment.py
but stops after 3 batches and prints a flush-buffered diagnostic between every
single GPU operation. It sets CUDA_LAUNCH_BLOCKING=1 so any CUDA error is caught
synchronously with a proper stack trace instead of killing the session silently.
"""
import faulthandler
import gc
import json
import logging
import os
import random
import sys
import time
import traceback

# CRITICAL: Force synchronous CUDA execution so errors are caught at the exact op
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # device-side assertions
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    faulthandler.enable()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fold3_debug")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2  # noqa: E402

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np  # noqa: E402
import torch  # noqa: E402

STEP = 0


def checkpoint(msg: str) -> None:
    global STEP
    STEP += 1
    gpu_mem = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        gpu_mem = f" | GPU: {alloc:.0f}MB alloc, {reserved:.0f}MB reserved"
    print(f"  [STEP {STEP:03d}] {msg}{gpu_mem}", flush=True)


def main() -> None:
    print("=" * 70, flush=True)
    print("FOLD 3 CRASH DIAGNOSTIC (CUDA_LAUNCH_BLOCKING=1)", flush=True)
    print("=" * 70, flush=True)

    # ── Phase 1: Hardware ──
    print("\n[PHASE 1/7] Hardware Check", flush=True)
    checkpoint("PyTorch version: " + torch.__version__)
    checkpoint(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        checkpoint(f"GPU: {torch.cuda.get_device_name(0)}")
        checkpoint(f"GPU count: {torch.cuda.device_count()}")
        checkpoint(f"CUDA version: {torch.version.cuda}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint(f"Using device: {device}")

    # ── Phase 2: Accelerator ──
    print("\n[PHASE 2/7] Creating Accelerator (fp16, grad_accum=2)", flush=True)
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
    checkpoint(f"Accelerator device: {accelerator.device}")
    checkpoint(f"Accelerator distributed_type: {accelerator.distributed_type}")
    checkpoint(f"Accelerator num_processes: {accelerator.num_processes}")
    checkpoint(f"Accelerator mixed_precision: {accelerator.mixed_precision}")

    # ── Phase 3: Model ──
    print("\n[PHASE 3/7] Creating Model + Optimizer + Scheduler", flush=True)
    from src.models.hybrid_detector import HybridDeepfakeDetector
    from src.training.optimization import get_differential_param_groups

    checkpoint("Instantiating HybridDeepfakeDetector(resse)...")
    model = HybridDeepfakeDetector(frequency_backbone="resse")
    checkpoint(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint("Creating AdamW optimizer...")
    optimizer = torch.optim.AdamW(get_differential_param_groups(model))
    checkpoint("Optimizer created.")

    checkpoint("Creating CosineAnnealingLR scheduler...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    checkpoint("Scheduler created.")

    # ── Phase 4: Data ──
    print("\n[PHASE 4/7] Loading Data (Fold 3 = FaceSwap Holdout)", flush=True)
    from src.dataset.datasets import FaceCropDataset
    from src.dataset.domains import DomainClassifier
    from src.dataset.loader import get_transforms
    from src.dataset.resolver import find_dataset_root, resolve_splits_path

    data_root = find_dataset_root()
    splits_path = resolve_splits_path(data_root=data_root)
    checkpoint(f"Dataset root: {data_root}")
    checkpoint(f"Splits path: {splits_path}")

    with open(splits_path, "r") as f:
        splits = json.load(f)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits.get("test", [])
    checkpoint(f"Splits loaded: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    holdout = "faceswap"
    train_loto = [
        s for s in train_samples if s and len(s) >= 2 and s[0] and not DomainClassifier.matches_holdout(s[0], holdout)
    ]
    checkpoint(f"Fold 3 train: {len(train_loto)} samples")

    del splits, train_samples, val_samples, test_samples
    gc.collect()
    checkpoint("Freed splits manifest from memory")

    train_transform, _ = get_transforms(img_size=256, hardened=True)
    train_ds = FaceCropDataset(train_loto, data_root, is_train=True, transform=train_transform)
    checkpoint(f"Dataset created: {len(train_ds)} samples")

    fold_seed = 42 + sum(ord(c) for c in holdout)
    g_train = torch.Generator()
    g_train.manual_seed(fold_seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_train,
    )
    checkpoint(f"DataLoader created: {len(train_loader)} batches")

    # ── Phase 5: Loss + Prepare ──
    print("\n[PHASE 5/7] Setting up Loss + accelerator.prepare()", flush=True)
    from src.training.loss import FocalLossWithLogits

    num_fake = sum(1 for s in train_loto if s[1] == 1)
    num_real = len(train_loto) - num_fake
    pos_weight_val = min(float(num_real / max(1, num_fake)), 3.0)
    pos_weight_tensor = torch.tensor([pos_weight_val], device=accelerator.device)
    criterion = FocalLossWithLogits(gamma=2.0, pos_weight=pos_weight_tensor)
    checkpoint(f"FocalLoss created: pos_weight={pos_weight_val:.3f}")

    checkpoint("Calling accelerator.prepare(model, optimizer, train_loader, scheduler)...")
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    checkpoint("accelerator.prepare() completed!")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        checkpoint("CUDA synchronized after prepare()")

    # ── Phase 6: Forward + Backward on 3 Batches ──
    print("\n[PHASE 6/7] Running 3 Training Batches (the exact crash zone)", flush=True)
    model.train()
    checkpoint("model.train() set")

    train_iter = iter(train_loader)
    for batch_idx in range(3):
        print(f"\n  --- BATCH {batch_idx} ---", flush=True)

        # Step A: Load batch
        try:
            checkpoint(f"Batch {batch_idx}: calling next(train_iter)...")
            images, labels, valid_flags = next(train_iter)
            checkpoint(
                f"Batch {batch_idx}: yielded! images={tuple(images.shape)}, "
                f"labels={tuple(labels.shape)}, valid={int(valid_flags.sum())}/{len(valid_flags)}"
            )
        except Exception as e:
            checkpoint(f"Batch {batch_idx}: EXCEPTION loading data: {e}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            return

        labels = labels.unsqueeze(1) if labels.ndim == 1 else labels
        valid_flags = valid_flags.unsqueeze(1) if valid_flags.ndim == 1 else valid_flags

        # Step B: Forward pass
        try:
            checkpoint(f"Batch {batch_idx}: entering accumulate + autocast...")
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    unwrapped = accelerator.unwrap_model(model)
                    has_aux = (
                        getattr(unwrapped, "frequency_backbone", None) == "resse"
                        and getattr(unwrapped, "use_fft_branch", False)
                    )
                    checkpoint(f"Batch {batch_idx}: has_aux={has_aux}, starting forward pass...")

                    if has_aux:
                        outputs, aux_outputs = model(images, return_aux=True)
                        checkpoint(
                            f"Batch {batch_idx}: forward pass done! "
                            f"outputs={tuple(outputs.shape)}, aux={tuple(aux_outputs.shape)}"
                        )
                    else:
                        outputs = model(images)
                        checkpoint(f"Batch {batch_idx}: forward pass done! outputs={tuple(outputs.shape)}")

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    checkpoint(f"Batch {batch_idx}: CUDA sync after forward pass OK")

                    # Step C: Loss
                    checkpoint(f"Batch {batch_idx}: computing loss...")
                    try:
                        loss_main = criterion(outputs, labels, valid_flags=valid_flags)
                    except TypeError:
                        loss_unreduced = criterion(outputs, labels)
                        if loss_unreduced.ndim > 0:
                            loss_main = (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
                        else:
                            loss_main = loss_unreduced

                    if has_aux:
                        try:
                            loss_aux = criterion(aux_outputs, labels, valid_flags=valid_flags)
                        except TypeError:
                            loss_aux_unreduced = criterion(aux_outputs, labels)
                            if loss_aux_unreduced.ndim > 0:
                                loss_aux = (loss_aux_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
                            else:
                                loss_aux = loss_aux_unreduced
                        loss = loss_main + 0.3 * loss_aux
                    else:
                        loss = loss_main

                    checkpoint(f"Batch {batch_idx}: loss={float(loss.detach().item()):.4f}")

                # Step D: Backward pass
                checkpoint(f"Batch {batch_idx}: starting accelerator.backward(loss)...")
                accelerator.backward(loss)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                checkpoint(f"Batch {batch_idx}: backward pass completed!")

                # Step E: Optimizer step (if sync_gradients)
                if accelerator.sync_gradients:
                    checkpoint(f"Batch {batch_idx}: sync_gradients=True, clipping + stepping...")
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    checkpoint(f"Batch {batch_idx}: optimizer step completed!")
                else:
                    checkpoint(f"Batch {batch_idx}: sync_gradients=False (accumulating)")

        except Exception as e:
            checkpoint(f"Batch {batch_idx}: EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as e2:
                    checkpoint(f"CUDA sync after error also failed: {e2}")
            return

        checkpoint(f"Batch {batch_idx}: FULLY COMPLETED")

    # ── Phase 7: Summary ──
    print("\n[PHASE 7/7] Summary", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        checkpoint(f"Final GPU: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB alloc, "
                   f"{torch.cuda.max_memory_allocated(0) / 1024**2:.0f}MB peak")
    checkpoint("ALL 3 BATCHES COMPLETED SUCCESSFULLY — NO CRASH!")
    print("=" * 70, flush=True)
    print("If you see this message, the crash is NOT in the forward/backward pass.", flush=True)
    print("The crash must be in the DataLoader iteration or tqdm wrapper.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
