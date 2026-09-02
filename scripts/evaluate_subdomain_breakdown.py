"""Per-Generator Sub-Domain Evaluation Script for Dual-Stream Deepfake Detector."""

import json
import logging
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from scripts.train_dual_stream_ddp import KaggleFastDataset, dedupe_split, find_dataset_root
from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector

CONFIG = load_config()
BATCH_SIZE = CONFIG.get("training", {}).get("batch_size", 32) * 2


def categorize_sample_path(rel_path: str) -> str:
    path_norm = rel_path.replace("\\", "/").lower()

    if "real" in path_norm:
        return "Original Real Faces"
    elif "id" in path_norm or "__" in path_norm or "celeb" in path_norm:
        return "Celeb-DF v2 Synthesis"
    else:
        pair_match = re.search(r"(?:^|/)(\d{3})_\d{3}(?:/|$)", path_norm)
        if pair_match:
            pair_num = int(pair_match.group(1))
            if 0 <= pair_num <= 199:
                return "FF++ Deepfakes (Pairs 0-199)"
            elif 200 <= pair_num <= 399:
                return "FF++ Face2Face (Pairs 200-399)"
            elif 400 <= pair_num <= 599:
                return "FF++ FaceSwap (Pairs 400-599)"
            elif 600 <= pair_num <= 799:
                return "FF++ NeuralTextures (Pairs 600-799)"
            return "FF++ Numeric Manipulation Pairs"
        return "FF++ Deepfakes / Mixed"


def run_subdomain_evaluation() -> None:
    data_root = find_dataset_root()
    splits_path = os.path.join(data_root, "splits.json")
    if os.path.exists("/kaggle/working/splits.json"):
        splits_path = "/kaggle/working/splits.json"
    elif os.path.exists("./splits.json"):
        splits_path = "./splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    test_samples = dedupe_split(splits["test"])

    grouped_samples: dict[str, list] = {}
    for sample in test_samples:
        path = sample[0]
        group = categorize_sample_path(path)
        if group not in grouped_samples:
            grouped_samples[group] = []
        grouped_samples[group].append(sample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDeepfakeDetector().to(device)

    calibrated_ckpt_path = "/kaggle/working/dual_stream_calibrated.pth"
    if not os.path.exists(calibrated_ckpt_path):
        calibrated_ckpt_path = os.path.join(data_root, "dual_stream_calibrated.pth")
    if not os.path.exists(calibrated_ckpt_path):
        calibrated_ckpt_path = os.path.join(REPO_ROOT, "dual_stream_calibrated.pth")

    if os.path.exists(calibrated_ckpt_path):
        try:
            ckpt = torch.load(calibrated_ckpt_path, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning("weights_only=True failed, falling back safely: %s", e)
            ckpt = torch.load(calibrated_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = float(ckpt["optimal_threshold"])
        temp = float(ckpt["temperature"])
        print(f"Loaded Calibrated Checkpoint from {calibrated_ckpt_path} (Threshold={threshold:.2f}, Temp={temp:.4f})")
    else:
        raise FileNotFoundError(
            f"Calibrated checkpoint file not found at {calibrated_ckpt_path}! Please run scripts/evaluate_test_set.py first."
        )

    model.eval()

    print("\nPER-GENERATOR SUB-DOMAIN EVALUATION (2-Class AUC vs Real Faces)")

    real_samples = grouped_samples.get("Original Real Faces", [])
    real_logits_list, real_targets_list = [], []

    if real_samples:
        real_loader = DataLoader(
            KaggleFastDataset(real_samples, data_root, is_train=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )
        with torch.no_grad():
            for images, labels, valid_flags in real_loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                images, labels = images[mask].to(device), labels[mask]
                logits = model(images).squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = np.array([logits])
                real_logits_list.extend(logits)
                real_targets_list.extend(labels.numpy())

    real_logits = np.array(real_logits_list)
    real_targets = np.array(real_targets_list)

    for group_name in sorted(grouped_samples.keys()):
        if group_name == "Original Real Faces":
            continue

        fake_samples = grouped_samples[group_name]
        if len(fake_samples) < 5:
            continue

        loader = DataLoader(
            KaggleFastDataset(fake_samples, data_root, is_train=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )
        fake_logits_list, fake_targets_list = [], []

        with torch.no_grad():
            for images, labels, valid_flags in loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                images, labels = images[mask].to(device), labels[mask]
                logits = model(images).squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = np.array([logits])
                fake_logits_list.extend(logits)
                fake_targets_list.extend(labels.numpy())

        group_logits = np.concatenate([np.array(fake_logits_list), real_logits])
        group_targets = np.concatenate([np.array(fake_targets_list), real_targets])

        probs = 1.0 / (1.0 + np.exp(-(group_logits / temp)))
        preds = (probs > threshold).astype(int)

        auc_val = roc_auc_score(group_targets, probs)
        f1 = f1_score(group_targets, preds, zero_division=0)
        prec = precision_score(group_targets, preds, zero_division=0)
        rec = recall_score(group_targets, preds, zero_division=0)
        print(f"  {group_name:<36} | Fakes: {len(fake_samples):<5} | AUC: {auc_val:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    print("\nPer-generator sub-domain evaluation complete.")


if __name__ == "__main__":
    run_subdomain_evaluation()
