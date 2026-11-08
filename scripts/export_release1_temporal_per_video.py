"""Canonical Release-1 per-video temporal export.

Reproduces scripts/evaluate_temporal_test_set.py bit-for-bit (same loader, same
transforms, same calibration, same metrics), but loads the splits from the
repo's canonical CSVs (splits/val_split.csv, splits/test_split.csv) instead of
the stale Phase-1 splits.json inside deepfake_crops_512.

Sanity gates (must hold or the run is NOT canonical):
  * grouped test sequences == 2,248  (2 of 2,250 manifest videos have <8 frames)
  * temporal/exported video AUC == 0.8994
"""

import csv
import hashlib
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.evaluate_temporal_test_set import evaluate_loader
from src.dataset.loader import SequenceVideoDataset, get_transforms, group_video_sequences
from src.evaluation.metrics import compute_eer
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(REPO_ROOT, "deepfake_crops_512")
BACKBONE = os.path.join(REPO_ROOT, "models", "dual_stream_calibrated.pth")
TEMPORAL = os.path.join(REPO_ROOT, "models", "temporal_head_best.pth")
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
SEQ_LEN = 8
SCRATCH = os.path.join(REPO_ROOT, "results", "release1_run", "_inc_resume")
BATCH = 16

VAL_CACHE = os.path.join(SCRATCH, "val_cache.npz")
PARTS_DIR = os.path.join(SCRATCH, "test_parts")


def sha256(path: str, limit_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit_bytes))
    return h.hexdigest()[:16]


def load_split_csv(name: str) -> list[tuple[str, int]]:
    with open(os.path.join(SPLITS_DIR, f"{name}_split.csv"), newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [(os.path.join(DATA_ROOT, r[0]), int(r[1])) for r in rows]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    # Checkpoints (mirrors evaluate_temporal_test_set.py)
    b_ckpt = torch.load(BACKBONE, map_location="cpu", weights_only=False)
    backbone = HybridDeepfakeDetector(pretrained=False)
    backbone.load_state_dict(clean_state_dict(b_ckpt.get("model_state_dict", b_ckpt)), strict=False)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    t_ckpt = torch.load(TEMPORAL, map_location="cpu", weights_only=False)
    t_state = t_ckpt.get("model_state_dict", t_ckpt)
    hidden_dim = t_ckpt.get("hidden_dim", 256)
    use_deltas = "use_deltas" in t_ckpt and bool(t_ckpt["use_deltas"])
    if "use_deltas" not in t_ckpt:
        ih_weight = t_state.get("gru.weight_ih_l0", None)
        use_deltas = ih_weight is not None and ih_weight.shape[1] == 1024
    use_max_pool = bool(t_ckpt.get("use_max_pool", True))
    if "use_max_pool" not in t_ckpt:
        cls_weight = t_state.get("classifier.0.weight", None)
        use_max_pool = cls_weight is not None and cls_weight.shape[1] == hidden_dim * 4
    stride = int(t_ckpt.get("stride", 2))
    logger.info("Bi-GRU config: hidden_dim=%d use_deltas=%s use_max_pool=%s stride=%d", hidden_dim, use_deltas, use_max_pool, stride)
    temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=hidden_dim, use_deltas=use_deltas, use_max_pool=use_max_pool)
    temporal_model.load_state_dict(clean_state_dict(t_state), strict=False)
    temporal_model.to(device).eval()

    # Canonical splits from repo CSVs -> grouped video sequences
    val_samples = load_split_csv("val")
    test_samples = load_split_csv("test")
    val_videos = group_video_sequences(val_samples, min_frames=SEQ_LEN)
    test_videos = group_video_sequences(test_samples, min_frames=SEQ_LEN)
    logger.info("grouped video sequences: val=%d test=%d (expect test=2,248)", len(val_videos), len(test_videos))

    _, eval_transform = get_transforms(img_size=256, hardened=False)
    val_ds = SequenceVideoDataset(val_videos, transform=eval_transform, seq_len=SEQ_LEN, stride=stride, is_train=False)
    test_ds = SequenceVideoDataset(test_videos, transform=eval_transform, seq_len=SEQ_LEN, stride=stride, is_train=False)
    # 4 workers: each Windows dataloader process loads the full import stack (~GB), so fewer = less RAM.
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)

    os.makedirs(SCRATCH, exist_ok=True)

    # ---- Calibration on Validation Set via Balanced Platt Scaling ----
    if os.path.exists(VAL_CACHE):
        cached = np.load(VAL_CACHE)
        val_logits, val_targets = cached["logits"], cached["targets"]
        logger.info("reusing cached val pass: n=%d (delete %s to force recomputation)", len(val_targets), VAL_CACHE)
    else:
        val_logits, _, val_targets = evaluate_loader(backbone, temporal_model, val_loader, device, desc="Val Sequence Eval")
        np.savez_compressed(VAL_CACHE, logits=val_logits, targets=val_targets)
        logger.info("cached val pass -> %s", VAL_CACHE)

    calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs")
    calibrator.fit(val_logits.reshape(-1, 1), val_targets)
    val_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
    opt_thresh = float(thresholds[int(np.nanargmax(tpr - fpr))])
    logger.info("val calibrated AUC=%.4f  tau*=%.4f", roc_auc_score(val_targets, val_probs), opt_thresh)

    # ---- Test pass: per-batch incremental parts (crash-resilient + resumable) ----
    os.makedirs(PARTS_DIR, exist_ok=True)
    n_batches = len(test_loader)
    done = sorted(int(name.split("_")[1].split(".")[0]) for name in os.listdir(PARTS_DIR) if name.startswith("batch_"))
    logger.info("test pass: %d/%d batches already on disk, resuming", len(done), n_batches)

    from scripts.evaluate_temporal_test_set import extract_features_and_logits

    with torch.inference_mode():
        for idx, (frames, labels, _) in enumerate(test_loader):
            part_path = os.path.join(PARTS_DIR, f"batch_{idx:04d}.npz")
            if os.path.exists(part_path):
                continue
            frames, labels_np = frames.to(device), labels.numpy()
            embeddings, frame_logits = extract_features_and_logits(backbone, frames, device)
            logits_temporal, _ = temporal_model(embeddings)
            p_frame = torch.sigmoid(frame_logits).mean(dim=-1)
            np.savez_compressed(
                part_path,
                logits=logits_temporal.squeeze(-1).cpu().numpy(),
                naive=p_frame.cpu().numpy(),
                labels=labels_np,
            )
    logger.info("test pass complete: %d batches", n_batches)

    parts = [np.load(os.path.join(PARTS_DIR, f"batch_{i:04d}.npz")) for i in range(n_batches)]
    test_logits = np.concatenate([p["logits"] for p in parts])
    test_naive_probs = np.concatenate([p["naive"] for p in parts])
    test_targets = np.concatenate([p["labels"] for p in parts])

    test_probs = calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    auc = float(roc_auc_score(test_targets, test_probs))
    naive_auc = float(roc_auc_score(test_targets, test_naive_probs))
    eer, eer_thresh = compute_eer(test_targets, test_probs)

    logger.info("GATE n_sequences: got %d, expected 2248 -> %s", len(test_targets), "PASS" if len(test_targets) == 2248 else "FAIL")
    logger.info("GATE temporal AUC: got %.4f, expected 0.8994 -> %s", auc, "PASS" if abs(auc - 0.8994) < 0.0015 else "FAIL")
    logger.info("naive frame-average AUC: %.4f (expect ~0.8962)", naive_auc)

    out = {
        "probs_temporal": test_probs.tolist(),
        "probs_naive_avg": test_naive_probs.tolist(),
        "labels": test_targets.tolist(),
        "n_samples": len(test_targets),
        "optimal_threshold": opt_thresh,
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(test_targets, test_probs)),
        "naive_avg_roc_auc": naive_auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "balanced_accuracy": float(balanced_accuracy_score(test_targets, (test_probs >= opt_thresh).astype(int))),
        "stride": stride,
        "use_deltas": use_deltas,
        "use_max_pool": use_max_pool,
        "provenance": {
            "method": "scripts/export_release1_temporal_per_video.py",
            "generator_equivalent": "scripts/evaluate_temporal_test_set.py",
            "split_source": "splits/{val,test}_split.csv (canonical Release-1)",
            "backbone_sha256_16": sha256(BACKBONE),
            "temporal_sha256_16": sha256(TEMPORAL),
            "device": str(device),
        },
    }
    out_path = os.path.join(REPO_ROOT, "results", "release1_run", "temporal_test_predictions_release1.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    logger.info("saved %s", out_path)


if __name__ == "__main__":
    main()
