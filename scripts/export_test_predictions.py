"""Exports per-sample test set predictions for downstream plot generation."""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.datasets import FaceCropDataset
from src.dataset.loader import dedupe_split
from src.dataset.resolver import find_dataset_root, find_weights_path, resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample test predictions for ROC/ECE plot generation.")
    parser.add_argument("--checkpoint", default=None, help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--data_root", default=None, help="Dataset root containing splits.json")
    parser.add_argument("--output_json", default="test_predictions.json", help="Output path for predictions JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_root = find_dataset_root(args.data_root)
    checkpoint_path = find_weights_path(args.checkpoint, data_root)

    splits_path = resolve_splits_path(data_root=data_root)
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_samples = dedupe_split(splits["test"])
    logger.info("Loaded %d test samples from %s", len(test_samples), splits_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    temperature = float(ckpt.get("temperature", 1.0))
    threshold = float(ckpt.get("optimal_threshold", 0.50))
    logger.info("Using temperature=%.4f, threshold=%.4f from checkpoint", temperature, threshold)

    model = HybridDeepfakeDetector(pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state), strict=False)
    model.eval()

    dataset = FaceCropDataset(test_samples, data_root, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluator = ModelEvaluator(model, device=device)
    all_logits, all_targets, all_valid = evaluator.predict_loader(loader)

    valid_mask = all_valid > 0.0
    logits_clean = all_logits[valid_mask]
    targets_clean = all_targets[valid_mask].astype(int)

    probs_uncal = 1.0 / (1.0 + np.exp(-logits_clean))
    probs_cal = 1.0 / (1.0 + np.exp(-(logits_clean / temperature)))

    output_data = {
        "probs_raw": probs_uncal.tolist(),
        "probs_cal": probs_cal.tolist(),
        "labels": targets_clean.tolist(),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "n_samples": int(len(targets_clean)),
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Exported %d predictions to %s", len(targets_clean), args.output_json)


if __name__ == "__main__":
    main()
