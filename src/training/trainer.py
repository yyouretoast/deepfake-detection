"""Reusable Distributed Data Parallel (DDP) Trainer for Dual-Stream Deepfake Detectors."""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.ema import ExponentialMovingAverage

logger = logging.getLogger(__name__)


class DualStreamTrainer:
    """Encapsulates multi-GPU DDP training, gradient accumulation, EMA, metrics, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_loader: Any,
        val_loader: Any,
        accelerator: Any,
        ema: Optional[ExponentialMovingAverage] = None,
        max_grad_norm: float = 1.0,
        aux_loss_weight: float = 0.3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.ema = ema
        self.max_grad_norm = max_grad_norm
        self.aux_loss_weight = aux_loss_weight

    def _compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, valid_flags: torch.Tensor
    ) -> torch.Tensor:
        try:
            return self.criterion(outputs, labels, valid_flags=valid_flags)
        except TypeError:
            loss_unreduced = self.criterion(outputs, labels)
            if loss_unreduced.ndim > 0:
                return (loss_unreduced * valid_flags).sum() / valid_flags.sum().clamp(min=1.0)
            return loss_unreduced

    def train_one_epoch(self, epoch: int, total_epochs: int) -> dict[str, float]:
        """Runs a single training epoch with gradient accumulation and EMA updates."""
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()
        running_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_failures = torch.tensor(0.0, device=self.accelerator.device)

        desc = f"Epoch [{epoch + 1}/{total_epochs}]"
        train_iter = tqdm(self.train_loader, desc=desc, disable=not self.accelerator.is_main_process)

        for images, labels, valid_flags in train_iter:
            labels = labels.unsqueeze(1) if labels.ndim == 1 else labels
            valid_flags = valid_flags.unsqueeze(1) if valid_flags.ndim == 1 else valid_flags

            num_corrupt = (valid_flags == 0.0).sum()
            total_failures += num_corrupt

            with self.accelerator.accumulate(self.model):
                unwrapped = self.accelerator.unwrap_model(self.model)
                has_aux = (
                    getattr(unwrapped, "frequency_backbone", None) == "resse"
                    and getattr(unwrapped, "use_fft_branch", False)
                )

                if has_aux:
                    outputs, aux_outputs = self.model(images, return_aux=True)
                    loss_main = self._compute_loss(outputs, labels, valid_flags)
                    loss_aux = self._compute_loss(aux_outputs, labels, valid_flags)
                    loss = loss_main + self.aux_loss_weight * loss_aux
                else:
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, labels, valid_flags)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.accelerator.unwrap_model(self.model))

            running_loss += loss.detach()

        epoch_loss = float(self.accelerator.reduce(running_loss, reduction="sum").item()) / len(
            self.train_loader.dataset
        )
        failures = int(self.accelerator.reduce(total_failures, reduction="sum").item())
        return {"train_loss": epoch_loss, "failures": failures}

    @torch.inference_mode()
    def evaluate(self, loader: Optional[Any] = None) -> dict[str, float]:
        """Evaluates model (applying EMA shadow weights if present) and computes loss and AUC."""
        eval_loader = loader if loader is not None else self.val_loader
        self.model.eval()

        val_loss_tensor = torch.tensor(0.0, device=self.accelerator.device)
        val_failures_tensor = torch.tensor(0.0, device=self.accelerator.device)
        all_preds = []
        all_targets = []

        unwrapped = self.accelerator.unwrap_model(self.model)
        backup = self.ema.apply_shadow(unwrapped) if self.ema is not None else None

        try:
            for images, labels, valid_flags in eval_loader:
                labels = labels.unsqueeze(1) if labels.ndim == 1 else labels
                valid_flags = valid_flags.unsqueeze(1) if valid_flags.ndim == 1 else valid_flags

                val_failures_tensor += (valid_flags == 0.0).sum()

                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels, valid_flags)
                val_loss_tensor += loss.detach()

                probs = torch.sigmoid(outputs)
                gathered_probs, gathered_labels = self.accelerator.gather_for_metrics((probs, labels))

                all_preds.extend(gathered_probs.cpu().reshape(-1).tolist())
                all_targets.extend(gathered_labels.cpu().reshape(-1).tolist())
        finally:
            if self.ema is not None and backup is not None:
                self.ema.restore(unwrapped, backup)

        total_val_loss = float(self.accelerator.reduce(val_loss_tensor, reduction="sum").item())
        val_loss = total_val_loss / max(1, len(eval_loader.dataset))
        failures = int(self.accelerator.reduce(val_failures_tensor, reduction="sum").item())

        preds_arr = np.array(all_preds).flatten()
        targets_arr = np.array(all_targets).flatten()

        try:
            if len(np.unique(targets_arr)) >= 2:
                val_auc = float(roc_auc_score(targets_arr, preds_arr))
            else:
                val_auc = 0.5
        except (ValueError, TypeError, RuntimeError):
            val_auc = 0.5

        return {"val_loss": val_loss, "val_auc": val_auc, "failures": failures}

    def fit(
        self,
        num_epochs: int,
        save_path: str,
        checkpoint_dir: str,
        patience: int = 3,
    ) -> float:
        """Runs the full multi-epoch training and validation loop with early stopping."""
        best_val_auc = 0.0
        epochs_without_improvement = 0

        if self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            os.makedirs(os.path.abspath(checkpoint_dir), exist_ok=True)

        for epoch in range(num_epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch, num_epochs)

            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            val_auc = val_metrics["val_auc"]
            val_loss = val_metrics["val_loss"]
            train_loss = train_metrics["train_loss"]
            should_stop = torch.tensor(0, dtype=torch.int32, device=self.accelerator.device)

            if self.accelerator.is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] ({elapsed:.1f}s) - "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_without_improvement = 0
                    unwrapped = self.accelerator.unwrap_model(self.model)

                    backup = self.ema.apply_shadow(unwrapped) if self.ema is not None else None
                    torch.save(unwrapped.state_dict(), save_path)
                    if self.ema is not None and backup is not None:
                        self.ema.restore(unwrapped, backup)

                    logger.info(f"Saved Best Checkpoint (Val AUC: {val_auc:.4f}) to {save_path}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                        should_stop = torch.tensor(1, dtype=torch.int32, device=self.accelerator.device)

            if self.accelerator.num_processes > 1:
                should_stop = self.accelerator.reduce(should_stop, reduction="max")

            if should_stop.item() == 1:
                break

        return best_val_auc
