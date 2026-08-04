"""
Per-Generator Sub-Domain Evaluation Script for Dual-Stream Deepfake Detector.
Evaluates the calibrated checkpoint on held-out test split samples partitioned by exact
generator manipulation category (FF++ Deepfakes, Face2Face, FaceSwap, NeuralTextures, Celeb-DF).
"""
import os
import sys
import re

# Ensure repository root is on sys.path for standalone subprocess execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.models.hybrid_detector import HybridDeepfakeDetector
from scripts.train_dual_stream_ddp import KaggleFastDataset, find_dataset_root, dedupe_split

def categorize_sample_path(rel_path: str) -> str:
    """
    Strict directory token matching to categorize face crop samples into exact generator domains.
    Prevents false-positive matches on filename frames (e.g. 'id0_frame1').
    """
    path_norm = rel_path.replace("\\", "/")
    path_parts = [p.lower() for p in path_norm.split("/")]

    for p in path_parts:
        if "celeb" in p:
            return "Celeb-DF"
        elif p == "deepfakes" or p == "df":
            return "FF++ Deepfakes"
        elif p == "face2face" or p == "f2f":
            return "FF++ Face2Face"
        elif p == "faceswap" or p == "fs":
            return "FF++ FaceSwap"
        elif p == "neuraltextures" or p == "nt":
            return "FF++ NeuralTextures"
        elif p == "face_shifter" or p == "faceshifter":
            return "FaceShifter"

    if re.search(r"/(?:deepfakes|df)/", path_norm, re.IGNORECASE):
        return "FF++ Deepfakes"
    elif re.search(r"/(?:face2face|f2f)/", path_norm, re.IGNORECASE):
        return "FF++ Face2Face"
    elif re.search(r"/(?:faceswap|fs)/", path_norm, re.IGNORECASE):
        return "FF++ FaceSwap"
    elif re.search(r"/(?:neuraltextures|nt)/", path_norm, re.IGNORECASE):
        return "FF++ NeuralTextures"
    elif re.search(r"/celeb", path_norm, re.IGNORECASE):
        return "Celeb-DF"

    return "Mixed / Original Real"

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

        loader = DataLoader(KaggleFastDataset(group_list, data_root, is_train=False), batch_size=32, shuffle=False, num_workers=2)
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
            print(f"  📌 {group_name:<24} | Samples: {len(group_list):<5} | AUC: {auc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        else:
            acc = np.mean(preds == group_targets)
            cls_name = "Fake (1.0)" if unique_classes[0] == 1.0 else "Real (0.0)"
            print(f"  📌 {group_name:<24} | Samples: {len(group_list):<5} | Acc: {acc:.4f} (Single-Class: {cls_name})")

    print("="*75 + "\n✅ PER-GENERATOR SUB-DOMAIN EVALUATION COMPLETE!")

if __name__ == '__main__':
    run_subdomain_evaluation()
