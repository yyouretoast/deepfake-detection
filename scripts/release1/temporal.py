"""Stage 5 of the Release-1 pipeline: clip embedding extraction and spatiotemporal
Bi-GRU video head training + evaluation.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.loader import SequenceVideoDataset, get_transforms, group_video_sequences
from src.evaluation.metrics import compute_eer, find_optimal_threshold
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import clean_state_dict

logger = logging.getLogger("release1_runner")

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
    device: torch.device | None = None,
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

