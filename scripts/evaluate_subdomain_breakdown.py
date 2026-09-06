"""Fine-grained subdomain breakdown evaluation across manipulation techniques and source generators."""

import argparse
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.domains import DomainClassifier, ManipulationDomain
from src.dataset.loader import dedupe_split
from src.dataset.resolver import find_dataset_root, find_weights_path, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import fit_temperature_log
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_subdomain_evaluation(
    data_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
    output_json: Optional[str] = None,
) -> None:
    data_root = find_dataset_root(data_dir)
    splits_path = resolve_splits_path(data_root=data_root)

    print(f"Loading test split from: {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    test_samples = dedupe_split(splits.get("test", []))
    val_samples = dedupe_split(splits.get("val", []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridDeepfakeDetector().to(device)

    resolved_weights = find_weights_path(weights_path, data_root)
    print(f"Loading weights from: {resolved_weights}")

    checkpoint = torch.load(resolved_weights, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.eval()

    evaluator = ModelEvaluator(model, device=device)

    # Calibration parameters: extract from checkpoint if calibrated, or fit on val split
    if "optimal_threshold" in checkpoint and "temperature" in checkpoint:
        best_thresh = float(checkpoint["optimal_threshold"])
        temp = float(checkpoint["temperature"])
        print(f"Using calibrated parameters from checkpoint: tau* = {best_thresh:.4f}, T* = {temp:.4f}")
    else:
        val_loader = DataLoader(FaceCropDataset(val_samples, data_root, is_train=False), batch_size=32, shuffle=False)
        val_logits, val_targets, val_valid = evaluator.predict_loader(val_loader)
        val_mask = val_valid > 0.0
        val_logits, val_targets = val_logits[val_mask], val_targets[val_mask]

        temp = fit_temperature_log(val_logits, val_targets) if len(np.unique(val_targets)) > 1 else 1.0
        val_probs = 1.0 / (1.0 + np.exp(-(val_logits / temp)))

        thresholds = np.linspace(0.1, 0.9, 81)
        best_thresh = 0.5
        best_f1 = 0.0
        for t in thresholds:
            f1 = f1_score(val_targets, (val_probs >= t).astype(int), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        print(f"Fitted calibration on val split: tau* = {best_thresh:.4f} (Macro F1 = {best_f1:.4f}), T* = {temp:.4f}")

    # Group test samples by domain using canonical DomainClassifier
    domain_buckets: dict[ManipulationDomain, list] = {d: [] for d in ManipulationDomain}
    real_samples = []

    for item in test_samples:
        path = item[0]
        if item[1] == 0:
            real_samples.append(item)
        else:
            info = DomainClassifier.classify(path)
            domain_buckets[info.domain].append(item)

    print("\n" + "=" * 80)
    print("        SUB-DOMAIN MANIPULATION BREAKDOWN EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Total Real Test Samples:      {len(real_samples)}")
    for domain, bucket in domain_buckets.items():
        if bucket:
            print(f"  {domain.value.upper():<20} Test Samples: {len(bucket)}")
    print("-" * 80)

    results = {}

    # Pre-evaluate real samples once to avoid redundant computation
    real_loader = DataLoader(FaceCropDataset(real_samples, data_root, is_train=False), batch_size=32, shuffle=False)
    real_logits, real_targets, real_valid = evaluator.predict_loader(real_loader)
    mask_r = real_valid > 0.0
    real_logits, real_targets = real_logits[mask_r], real_targets[mask_r]

    for domain, fake_items in domain_buckets.items():
        if not fake_items:
            continue
        fake_loader = DataLoader(FaceCropDataset(fake_items, data_root, is_train=False), batch_size=32, shuffle=False)
        fake_logits, fake_targets, fake_valid = evaluator.predict_loader(fake_loader)
        mask_f = fake_valid > 0.0
        fake_logits, fake_targets = fake_logits[mask_f], fake_targets[mask_f]

        logits = np.concatenate([fake_logits, real_logits])
        targets = np.concatenate([fake_targets, real_targets])

        probs = 1.0 / (1.0 + np.exp(-(logits / temp)))
        preds = (probs >= best_thresh).astype(int)

        try:
            auc_val = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5
        except (ValueError, TypeError, RuntimeError):
            auc_val = 0.5

        bal_acc = float(balanced_accuracy_score(targets, preds))
        f1 = float(f1_score(targets, preds, zero_division=0))
        prec = float(precision_score(targets, preds, zero_division=0))
        rec = float(recall_score(targets, preds, zero_division=0))

        domain_display = domain.value.upper()
        results[domain.value] = {
            "display_name": domain_display,
            "fakes_count": len(fake_items),
            "real_count": len(real_samples),
            "auc": auc_val,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        }
        print(f"  {domain_display:<20} | Fakes: {len(fake_items):<5} | AUC: {auc_val:.4f} | BalAcc: {bal_acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    print("=" * 80 + "\n")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved subdomain breakdown results to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-grained subdomain breakdown performance.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing dataset and splits.json")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to dual_stream_best.pth")
    parser.add_argument(
        "--output_json",
        type=str,
        default="/kaggle/working/subdomain_results.json" if os.path.exists("/kaggle/working") else "subdomain_results.json",
        help="Path to export subdomain results JSON",
    )
    args = parser.parse_args()

    run_subdomain_evaluation(data_dir=args.data_dir, weights_path=args.weights_path, output_json=args.output_json)


if __name__ == "__main__":
    main()
