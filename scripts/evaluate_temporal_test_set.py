"""Evaluate trained Bi-GRU temporal consistency head on held-out test video sequences."""

import argparse
import json
import logging
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.loader import SequenceVideoDataset, get_transforms, group_video_sequences
from src.dataset.resolver import DatasetResolver, find_weights_path, resolve_splits_path
from src.evaluation.metrics import compute_eer
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bi-GRU Temporal Head on Test Video Sequences.")
    parser.add_argument("--backbone_weights", type=str, default=None, help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--temporal_weights", type=str, default=None, help="Path to temporal_head_best.pth")
    parser.add_argument("--data_root", type=str, default=None, help="Path to cropped dataset root.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="/kaggle/working/temporal_test_predictions.json",
        help="Path to export temporal test predictions JSON.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for video sequences.")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length in frames.")
    return parser.parse_args()


def extract_features_and_logits(
    backbone: HybridDeepfakeDetector, frames: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts features [B, T, 512] and single-frame logits [B, T] from clip [B, T, 3, H, W]."""
    b, t, c, h, w = frames.shape
    frames_flat = frames.view(b * t, c, h, w).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        feats = backbone.extract_features(frames_flat)
        logits = backbone(frames_flat)
        if isinstance(logits, tuple):
            logits = logits[0]
    return feats.view(b, t, -1), logits.view(b, t)


def evaluate_loader(
    backbone: HybridDeepfakeDetector,
    temporal_model: BiGRUTemporalDetector,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates video sequence loader, returning temporal raw logits, frame-avg probs, and targets."""
    temporal_logits, naive_probs, targets = [], [], []

    with torch.inference_mode():
        for frames, labels, _ in tqdm(loader, desc=desc):
            labels_np = labels.cpu().numpy()
            embeddings, frame_logits = extract_features_and_logits(backbone, frames, device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits_temporal, _ = temporal_model(embeddings)
                p_frame = torch.sigmoid(frame_logits).mean(dim=-1)

            temporal_logits.extend(logits_temporal.squeeze(-1).cpu().numpy().tolist())
            naive_probs.extend(p_frame.cpu().numpy().tolist())
            targets.extend(labels_np.tolist())

    return np.array(temporal_logits), np.array(naive_probs), np.array(targets)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using evaluation device: %s", device)

    data_root = DatasetResolver.find_dataset_root(args.data_root)
    backbone_path = find_weights_path(args.backbone_weights, data_root=data_root)
    temporal_path = find_weights_path(args.temporal_weights, data_root=data_root)

    logger.info("Backbone checkpoint: %s", backbone_path)
    logger.info("Temporal checkpoint: %s", temporal_path)

    # 1. Load Backbone
    backbone = HybridDeepfakeDetector(pretrained=False)
    if backbone_path and os.path.exists(backbone_path):
        b_ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)
        b_state = b_ckpt.get("model_state_dict", b_ckpt)
        backbone.load_state_dict(clean_state_dict(b_state), strict=False)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # 2. Load Bi-GRU Temporal Head
    hidden_dim = 256
    if temporal_path and os.path.exists(temporal_path):
        t_ckpt = torch.load(temporal_path, map_location="cpu", weights_only=False)
        t_state = t_ckpt.get("model_state_dict", t_ckpt)
        hidden_dim = t_ckpt.get("hidden_dim", 256)
        temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=hidden_dim)
        temporal_model.load_state_dict(clean_state_dict(t_state), strict=False)
    else:
        temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=hidden_dim)
    temporal_model.to(device).eval()

    # 3. Load Splits
    splits_path = resolve_splits_path(data_root=data_root)
    with open(splits_path, "r") as f:
        manifest = json.load(f)

    val_samples = [(os.path.join(data_root, p), lbl) for p, lbl in manifest["val"]]
    test_samples = [(os.path.join(data_root, p), lbl) for p, lbl in manifest["test"]]

    val_videos = group_video_sequences(val_samples, min_frames=args.seq_len)
    test_videos = group_video_sequences(test_samples, min_frames=args.seq_len)
    logger.info("Loaded %d validation video clips, %d test video clips.", len(val_videos), len(test_videos))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    val_ds = SequenceVideoDataset(val_videos, transform=eval_transform, seq_len=args.seq_len)
    test_ds = SequenceVideoDataset(test_videos, transform=eval_transform, seq_len=args.seq_len)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. Calibration on Validation Set via Balanced Platt Scaling
    logger.info("Computing validation calibration via Balanced Platt Scaling...")
    val_logits, _, val_targets = evaluate_loader(backbone, temporal_model, val_loader, device, desc="Val Sequence Eval")

    calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs")
    calibrator.fit(val_logits.reshape(-1, 1), val_targets)
    val_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]

    fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
    j_scores = tpr - fpr
    best_idx = int(np.nanargmax(j_scores))
    opt_thresh = float(thresholds[best_idx])
    val_bal_acc = float(balanced_accuracy_score(val_targets, (val_probs >= opt_thresh).astype(int)))
    logger.info("Calibrated Val Threshold tau* = %.4f (Val Balanced Acc: %.4f)", opt_thresh, val_bal_acc)

    # 5. Evaluate Held-Out Test Set
    logger.info("Evaluating on held-out test video sequences...")
    test_logits, test_naive_probs, test_targets = evaluate_loader(
        backbone, temporal_model, test_loader, device, desc="Test Sequence Eval"
    )

    test_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    test_auc = float(roc_auc_score(test_targets, test_probs))
    test_pr_auc = float(average_precision_score(test_targets, test_probs))
    naive_auc = float(roc_auc_score(test_targets, test_naive_probs))
    test_eer, eer_thresh = compute_eer(test_targets, test_probs)

    test_preds_opt = (test_probs >= opt_thresh).astype(int)
    test_preds_default = (test_probs >= 0.50).astype(int)
    bal_acc_opt = float(balanced_accuracy_score(test_targets, test_preds_opt))
    bal_acc_default = float(balanced_accuracy_score(test_targets, test_preds_default))

    print("\n" + "=" * 70)
    print("      HELD-OUT TEST SET TEMPORAL VIDEO BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Test Video Sequences:            {len(test_targets):,}")
    print(f"  Naive Frame-Averaging ROC AUC:   {naive_auc:.4f}")
    print(f"  Bi-GRU Video ROC AUC:            {test_auc:.4f}  (+{(test_auc - naive_auc)*100:+.2f}% vs Frame-Avg)")
    print(f"  Bi-GRU Video PR AUC:             {test_pr_auc:.4f}")
    print(f"  Bi-GRU Equal Error Rate (EER):   {test_eer:.4f} (at tau = {eer_thresh:.4f})")
    print("-" * 70)
    print("  Metrics at Default Threshold (0.50):")
    print(f"    Balanced Accuracy:             {bal_acc_default:.4f}")
    print(f"    Macro F1:                      {f1_score(test_targets, test_preds_default, average='macro', zero_division=0):.4f}")
    print(f"    Fake Precision / Recall:       {precision_score(test_targets, test_preds_default, zero_division=0):.4f} / {recall_score(test_targets, test_preds_default, zero_division=0):.4f}")
    print(f"  Metrics at Youden's J Threshold ({opt_thresh:.4f}):")
    print(f"    Balanced Accuracy:             {bal_acc_opt:.4f}")
    print(f"    Macro F1:                      {f1_score(test_targets, test_preds_opt, average='macro', zero_division=0):.4f}")
    print(f"    Fake Precision / Recall:       {precision_score(test_targets, test_preds_opt, zero_division=0):.4f} / {recall_score(test_targets, test_preds_opt, zero_division=0):.4f}")
    print("\nDetailed Test Classification Report (Optimal Threshold):")
    print(classification_report(test_targets, test_preds_opt, target_names=["Real", "Fake"], digits=4))

    real_indices = np.where(test_targets == 0)[0]
    fake_indices = np.where(test_targets == 1)[0]
    n_balanced = min(len(real_indices), len(fake_indices))
    if n_balanced > 0 and len(real_indices) != len(fake_indices):
        rng = np.random.default_rng(42)
        sampled_fake_idx = rng.choice(fake_indices, size=n_balanced, replace=False)
        balanced_idx = np.concatenate([real_indices, sampled_fake_idx])
        b_targets = test_targets[balanced_idx]
        b_preds = test_preds_opt[balanced_idx]
        b_probs = test_probs[balanced_idx]
        b_auc = float(roc_auc_score(b_targets, b_probs))
        b_bal_acc = float(balanced_accuracy_score(b_targets, b_preds))
        print(f"\nBalanced 1:1 Benchmark Subset ({n_balanced:,} Real vs {n_balanced:,} Fake):")
        print(f"    Balanced ROC AUC:     {b_auc:.4f}")
        print(f"    Balanced Accuracy:    {b_bal_acc:.4f}")
        print(classification_report(b_targets, b_preds, target_names=["Real", "Fake"], digits=4))

    # 6. Save JSON
    output_path = args.output_json
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results = {
        "probs_temporal": test_probs.tolist(),
        "probs_naive_avg": test_naive_probs.tolist(),
        "labels": test_targets.tolist(),
        "optimal_threshold": opt_thresh,
        "roc_auc": test_auc,
        "pr_auc": test_pr_auc,
        "naive_avg_roc_auc": naive_auc,
        "eer": test_eer,
        "eer_threshold": eer_thresh,
        "balanced_accuracy": bal_acc_opt,
        "balanced_accuracy_default": bal_acc_default,
        "n_samples": len(test_targets),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved temporal test predictions to %s", output_path)


if __name__ == "__main__":
    main()
