"""Train lightweight spatiotemporal Bi-GRU temporal consistency head on frozen dual-stream embeddings."""

import argparse
import json
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.loader import SequenceVideoDataset, get_transforms, group_video_sequences
from src.dataset.resolver import DatasetResolver, find_weights_path, resolve_splits_path
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bi-GRU Temporal Head on Frozen Backbone.")
    parser.add_argument("--backbone_weights", type=str, default=None, help="Path to trained backbone weights.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to cropped dataset root.")
    parser.add_argument("--save_path", type=str, default="weights/temporal_head_best.pth", help="Checkpoint save path.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of video sequences.")
    parser.add_argument("--seq_len", type=int, default=8, help="Number of frames per video sequence.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Bi-GRU head.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Bi-GRU hidden dimension.")
    return parser.parse_args()


def extract_clip_embeddings(
    backbone: HybridDeepfakeDetector, frames: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Extracts [B, T, 512] embedding sequence from [B, T, 3, H, W] video clip."""
    b, t, c, h, w = frames.shape
    frames_flat = frames.view(b * t, c, h, w).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            feats = backbone.extract_features(frames_flat)  # [B*T, 512]
    return feats.view(b, t, -1)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = DatasetResolver.find_dataset_root(args.data_root)
    weights_path = find_weights_path(args.backbone_weights)

    logger.info("Initializing backbone detector from %s", weights_path)
    backbone = HybridDeepfakeDetector(pretrained=False)
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        backbone.load_state_dict(clean_state_dict(state_dict), strict=False)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    splits_path = resolve_splits_path(data_root=data_root)
    logger.info("Loading sequence splits from: %s", splits_path)
    with open(splits_path, "r") as f:
        manifest = json.load(f)

    train_samples = [(os.path.join(data_root, p), lbl) for p, lbl in manifest["train"]]
    val_samples = [(os.path.join(data_root, p), lbl) for p, lbl in manifest["val"]]

    train_videos = group_video_sequences(train_samples, min_frames=args.seq_len)
    val_videos = group_video_sequences(val_samples, min_frames=args.seq_len)

    logger.info("Found %d training video clips and %d validation video clips.", len(train_videos), len(val_videos))

    train_transform, eval_transform = get_transforms(img_size=256, hardened=True)
    train_ds = SequenceVideoDataset(train_videos, transform=train_transform, seq_len=args.seq_len)
    val_ds = SequenceVideoDataset(val_videos, transform=eval_transform, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(args.epochs):
        temporal_model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")

        for frames, labels, _ in pbar:
            labels = labels.unsqueeze(1).float().to(device)
            embeddings = extract_clip_embeddings(backbone, frames, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, _ = temporal_model(embeddings)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        temporal_model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for frames, labels, _ in val_loader:
                labels = labels.unsqueeze(1).float().to(device)
                embeddings = extract_clip_embeddings(backbone, frames, device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits, _ = temporal_model(embeddings)
                    probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().squeeze(-1).tolist())
                val_targets.extend(labels.cpu().squeeze(-1).tolist())

        y_true = np.array(val_targets)
        y_score = np.array(val_preds)
        val_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
        logger.info("Epoch %d: Loss=%.4f, Val AUC=%.4f", epoch + 1, total_loss / max(1, len(train_loader)), val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": temporal_model.state_dict(),
                "val_auc": val_auc,
                "hidden_dim": args.hidden_dim,
                "seq_len": args.seq_len,
            }, args.save_path)
            logger.info("New best checkpoint saved to %s (AUC: %.4f)", args.save_path, best_auc)


if __name__ == "__main__":
    main()
