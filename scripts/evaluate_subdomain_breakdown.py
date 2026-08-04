"""
Per-Generator Sub-Domain Evaluation Script for Dual-Stream Deepfake Detector.
Evaluates the calibrated checkpoint on held-out test split samples partitioned by exact
generator manipulation category (FF++ Numeric Manipulation Pairs, Celeb-DF v2 Synthesis).
"""
import os
import sys
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config
from scripts.train_dual_stream_ddp import KaggleFastDataset, find_dataset_root, dedupe_split

CONFIG = load_config()
BATCH_SIZE = CONFIG.get("training", {}).get("batch_size", 32) * 2

def categorize_sample_path(rel_path: str) -> str:
    """
    Categorizes face crop samples into exact generator domains based on dataset structure.
    Celeb-DF v2: Contains 'id0_id' or double underscores '__'
    FF++ Numeric Pairs: Folder pattern like '000_003'
    """
    path_norm = rel_path.replace("\\", "/").lower()

    if "real" in path_norm:
        return "Original Real Faces"
    elif "id" in path_norm or "__" in path_norm or "celeb" in path_norm:
        return "Celeb-DF v2 Synthesis"
    else:
        folder = path_norm.split("/")[1] if len(path_norm.split("/")) > 1 else ""
        if re.match(r"^\d{3}_\d{3}$", folder):
            # Map known FF++ pair ranges if present
            pair_num = int(folder.split("_")[0]) if folder.split("_")[0].isdigit() else 0
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

def run_subdomain_evaluation():
    data_root = find_dataset_root()
    splits_path = os.path.join(data_root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    test_samples = dedupe_split(splits['test'])

    grouped_samples = {}
    for sample in test_samples:
        path, label = sample[0], sample[1]
        group = categorize_sample_path(path)
        if group not in grouped_samples:
            grouped_samples[group] = []
        grouped_samples[group].append(sample)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridDeepfakeDetector().to(device)

    calibrated_ckpt_path = '/kaggle/working/dual_stream_calibrated.pth'
    if not os.path.exists(calibrated_ckpt_path):
        calibrated_ckpt_path = os.path.join(data_root, 'dual_stream_calibrated.pth')

    if not os.path.exists(calibrated_ckpt_path):
        calibrated_ckpt_path = os.path.join(REPO_ROOT, 'dual_stream_calibrated.pth')

    if os.path.exists(calibrated_ckpt_path):
        ckpt = torch.load(calibrated_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        threshold = float(ckpt['optimal_threshold'])
        temp = float(ckpt['temperature'])
        print(f"✅ Loaded Calibrated Checkpoint from {calibrated_ckpt_path} (Threshold={threshold:.2f}, Temp={temp:.4f})")
    else:
        raise FileNotFoundError(f"Calibrated checkpoint file not found at {calibrated_ckpt_path}! Please run scripts/evaluate_test_set.py first.")

    model.eval()

    print("\n" + "="*75)
    print("🔬 PER-GENERATOR SUB-DOMAIN EVALUATION (HELD-OUT TEST SET)")
    print("="*75)

    for group_name in sorted(grouped_samples.keys()):
        group_list = grouped_samples[group_name]
        if len(group_list) < 5:
            continue

        loader = DataLoader(KaggleFastDataset(group_list, data_root, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        group_logits, group_targets = [], []

        with torch.no_grad():
            for images, labels, valid_flags in loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                images, labels = images[mask].to(device), labels[mask]
                logits = model(images).squeeze(-1).cpu().numpy()
                if logits.ndim == 0:
                    logits = np.array([logits])
                group_logits.extend(logits)
                group_targets.extend(labels.numpy())

        group_logits = np.array(group_logits)
        group_targets = np.array(group_targets)

        probs = 1.0 / (1.0 + np.exp(-(group_logits / temp)))
        preds = (probs > threshold).astype(int)

        unique_classes = np.unique(group_targets)
        
        if len(unique_classes) > 1:
            auc = roc_auc_score(group_targets, probs)
            f1 = f1_score(group_targets, preds)
            prec = precision_score(group_targets, preds, zero_division=0)
            rec = recall_score(group_targets, preds, zero_division=0)
            print(f"  📌 {group_name:<36} | Samples: {len(group_list):<5} | AUC: {auc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        else:
            acc = np.mean(preds == group_targets)
            cls_name = "Fake (1.0)" if unique_classes[0] == 1.0 else "Real (0.0)"
            print(f"  📌 {group_name:<36} | Samples: {len(group_list):<5} | Acc: {acc:.4f} (Single-Class: {cls_name})")

    print("="*75 + "\n✅ PER-GENERATOR SUB-DOMAIN EVALUATION COMPLETE!")

if __name__ == '__main__':
    run_subdomain_evaluation()
