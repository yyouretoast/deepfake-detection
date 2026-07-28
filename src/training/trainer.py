from typing import Dict, Tuple, Any, Optional
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve

try:
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.config import load_config

logger = logging.getLogger(__name__)

def get_grad_scaler(device_type: str = "cuda", enabled: bool = True):
    """PyTorch 2.6+ compatible GradScaler helper."""
    use = enabled and torch.cuda.is_available()
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=use)
    return torch.cuda.amp.GradScaler(enabled=use)

def get_autocast(device_type: str = "cuda", enabled: bool = True):
    """PyTorch 2.6+ compatible autocast helper."""
    use = enabled and torch.cuda.is_available()
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=use)
    return torch.cuda.amp.autocast(enabled=use)

class TwoPhaseTrainer:
    """
    Modular Two-Phase Training Engine for Hybrid Deepfake Detector.
    Encapsulates Phase 1 (Warmup Classifier Head), Phase 2 (LLRD Fine-Tuning),
    AMP FP16 scaling, Macro F1 threshold calibration, and EER evaluation.
    Supports both 4D single-frame and 5D video sequence inputs [B, T, 3, H, W].
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config if config is not None else load_config()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.config.get("training", {}).get("use_amp", True)
        
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            logger.info("DataParallel: distributing across %d GPUs", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = get_grad_scaler("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp)

    def _forward_step(self, batch) -> torch.Tensor:
        """DataParallel-invariant model forward step handling 4D/5D tensors and optional padding masks."""
        if isinstance(batch, (tuple, list)):
            imgs = batch[0]
            padding_mask = batch[1] if len(batch) > 1 else None
        else:
            imgs, padding_mask = batch, None

        imgs = imgs.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        if imgs.ndim == 5 and hasattr(unwrapped, "forward_sequence"):
            return unwrapped.forward_sequence(imgs, padding_mask=padding_mask)
        return self.model(imgs)

    def _get_llrd_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        """Layer-wise LR decay with 5 tiers:
        - Tier 0: stem + stages 0-1 (lr_backbone * 0.2)
        - Tier 1: stage 2          (lr_backbone * 0.5)
        - Tier 2: stage 3          (lr_backbone * 1.0)
        - Tier 3: fusion/freq layers (lr_backbone * 1.0) — NOT head LR
        - Tier 4: classifier head  (lr_head)
        """
        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        stem_s01, s2, s3, fusion_params, head_params = [], [], [], [], []

        # Fusion/frequency layer prefixes — these are feature extractors, not the head
        FUSION_PREFIXES = (
            "freq_extractor", "cross_attn", "spatial_proj",
            "freq_proj", "temporal_encoder", "attn_out_proj",
        )

        for name, p in unwrapped.named_parameters():
            if not p.requires_grad:
                continue
            clean_name = name.replace("module.", "")
            if "spatial_backbone" in clean_name:
                if any(s in clean_name for s in ["stem", "stages.0", "stages.1"]):
                    stem_s01.append(p)
                elif "stages.2" in clean_name:
                    s2.append(p)
                elif "stages.3" in clean_name:
                    s3.append(p)
                else:
                    stem_s01.append(p)
            elif any(clean_name.startswith(pfx) for pfx in FUSION_PREFIXES):
                fusion_params.append(p)
            else:
                head_params.append(p)

        return [
            {'params': stem_s01,      'lr': lr_backbone * 0.2},
            {'params': s2,            'lr': lr_backbone * 0.5},
            {'params': s3,            'lr': lr_backbone * 1.0},
            {'params': fusion_params, 'lr': lr_backbone * 1.0},
            {'params': head_params,   'lr': lr_head},
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Evaluates single-pass predictions over validation loader."""
        self.model.eval()
        running_loss = 0.0
        val_probs, val_targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    imgs, labels, padding_mask = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = (imgs, padding_mask)
                else:
                    imgs, labels = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = imgs

                with get_autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp):
                    logits = self._forward_step(fwd_input)
                    loss = self.criterion(logits, labels.float())

                running_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        total_samples = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') and len(self.val_loader.dataset) > 0 else len(val_targets)
        val_loss = running_loss / max(total_samples, 1)
        
        probs_arr = np.array(val_probs)
        targets_arr = np.array(val_targets)
        
        val_auc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 0.5

        # Macro F1 Threshold Calibration
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, opt_thresh = 0.0, 0.5
        for t in thresholds:
            preds = (probs_arr >= t).astype(int)
            f1 = float(f1_score(targets_arr, preds, average='macro', zero_division=0))
            if f1 > best_f1:
                best_f1, opt_thresh = f1, float(t)

        # Report accuracy at fixed 0.5 threshold (honest metric — opt_thresh is for deployment only)
        val_preds_fixed = (probs_arr >= 0.5).astype(int)
        val_acc = float(np.mean(val_preds_fixed == targets_arr))

        # Equal Error Rate (EER) Calculation
        eer = 0.50
        if HAS_SCIPY and len(np.unique(targets_arr)) > 1:
            try:
                fpr, tpr, _ = roc_curve(targets_arr, probs_arr)
                eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
            except Exception as e:
                logger.warning("EER calculation fallback: %s", e)

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "optimal_threshold": opt_thresh,
            "macro_f1": best_f1,
            "eer": eer,
            "probs": probs_arr,
            "targets": targets_arr
        }

    def train(self) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Executes full Two-Phase training pipeline."""
        training_cfg = self.config.get("training", {})
        epochs_p1 = training_cfg.get("epochs_phase1", 3)
        epochs_p2 = training_cfg.get("epochs_phase2", 5)
        lr_p1 = training_cfg.get("lr_phase1", 1e-3)
        lr_backbone = training_cfg.get("lr_backbone", 1e-5)
        lr_head = training_cfg.get("lr_head", 1e-4)
        weight_decay = training_cfg.get("weight_decay", 1e-2)
        patience = training_cfg.get("patience", 4)

        best_val_auc = 0.0
        best_weights = None
        best_opt_thresh = 0.5

        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        # --- Phase 1: Classifier Head Warmup ---
        logger.info("Phase 1: Training classifier head (backbone frozen)")
        for p in unwrapped.spatial_backbone.parameters():
            p.requires_grad = False

        head_params = [p for n, p in self.model.named_parameters() if "spatial_backbone" not in n and p.requires_grad]
        optimizer_p1 = torch.optim.AdamW(head_params, lr=lr_p1, weight_decay=weight_decay)

        for epoch in range(epochs_p1):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{epochs_p1} [Train]")
            for batch in pbar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    imgs, labels, padding_mask = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = (imgs, padding_mask)
                else:
                    imgs, labels = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = imgs

                optimizer_p1.zero_grad()
                with get_autocast(device_type, enabled=self.use_amp):
                    logits = self._forward_step(fwd_input)
                    smooth_labels = labels * 0.95 + 0.025
                    loss = self.criterion(logits, smooth_labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer_p1)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer_p1)
                self.scaler.update()
                running_loss += loss.item()

            val_metrics = self.evaluate()
            logger.info("Phase 1 - Epoch %d/%d Complete | Val Acc: %.2f%% | Val AUC: %.4f",
                        epoch + 1, epochs_p1, val_metrics["val_acc"] * 100, val_metrics["val_auc"])

            if val_metrics["val_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_auc"]
                best_opt_thresh = val_metrics["optimal_threshold"]
                best_weights = copy.deepcopy(unwrapped.state_dict())

        # --- Phase 2: LLRD Fine-Tuning ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Phase 2: Fine-tuning all layers (with LLRD)")
        for p in unwrapped.spatial_backbone.parameters():
            p.requires_grad = True

        param_groups = self._get_llrd_param_groups(lr_backbone=lr_backbone, lr_head=lr_head)
        optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=epochs_p2, eta_min=1e-6)

        patience_counter = 0

        for epoch in range(epochs_p2):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Phase 2 - Epoch {epoch+1}/{epochs_p2} [Train]")
            for batch in pbar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    imgs, labels, padding_mask = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = (imgs, padding_mask)
                else:
                    imgs, labels = batch
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    fwd_input = imgs

                optimizer_p2.zero_grad()
                with get_autocast(device_type, enabled=self.use_amp):
                    logits = self._forward_step(fwd_input)
                    smooth_labels = labels * 0.95 + 0.025
                    loss = self.criterion(logits, smooth_labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer_p2)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer_p2)
                self.scaler.update()
                running_loss += loss.item()

            scheduler.step()
            val_metrics = self.evaluate()
            logger.info("Phase 2 - Epoch %d/%d Complete | Val Acc: %.2f%% | Val AUC: %.4f | Opt T*: %.4f",
                        epoch + 1, epochs_p2, val_metrics["val_acc"] * 100, val_metrics["val_auc"], val_metrics["optimal_threshold"])

            if val_metrics["val_auc"] > best_val_auc:
                best_val_auc = val_metrics["val_auc"]
                best_opt_thresh = val_metrics["optimal_threshold"]
                best_weights = copy.deepcopy(unwrapped.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at Epoch %d", epoch + 1)
                    break

        if best_weights is not None:
            unwrapped.load_state_dict(best_weights)

        final_val_metrics = self.evaluate()
        final_val_metrics["best_val_auc"] = best_val_auc
        final_val_metrics["optimal_threshold"] = best_opt_thresh

        checkpoint_data = {
            "state_dict": best_weights,
            "optimal_threshold": float(best_opt_thresh),
            "val_auc": float(best_val_auc),
            "config": self.config
        }

        return checkpoint_data, best_opt_thresh, final_val_metrics

def train_two_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, float]:
    """Functional wrapper for TwoPhaseTrainer for backward compatibility."""
    trainer = TwoPhaseTrainer(model, train_loader, val_loader, config=config, device=device)
    checkpoint_data, opt_thresh, metrics = trainer.train()
    return model, opt_thresh
