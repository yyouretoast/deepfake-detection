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
from src.dataset.resolver import resolve_splits_path
from src.evaluation.evaluator import ModelEvaluator
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256
TestDataset = FaceCropDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample test predictions for ROC/ECE plot generation.")
    parser.add_argument("--checkpoint", required=True, help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--data_root", required=True, help="Dataset root containing splits.json")
    parser.add_argument("--output_json", default="test_predictions.json", help="Output path for predictions JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    splits_path = resolve_splits_path(data_root=args.data_root)
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_samples = dedupe_split(splits["test"])
    logger.info("Loaded %d test samples from %s", len(test_samples), splits_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    temperature = float(ckpt.get("temperature", 1.0))
    threshold = float(ckpt.get("optimal_threshold", 0.50))
    logger.info("Using temperature=%.4f, threshold=%.4f from checkpoint", temperature, threshold)

    model = HybridDeepfakeDetector(pretrained=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(clean_state_dict(state), strict=False)
    model.eval()

    dataset = FaceCropDataset(test_samples, args.data_root, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluator = ModelEvaluator(model, device=device)
    all_logits, all_targets, all_valid = evaluator.predict_loader(loader)

    probs_uncal = 1.0 / (1.0 + np.exp(-all_logits))
    probs_cal = 1.0 / (1.0 + np.exp(-(all_logits / temperature)))

    records = []
    for i, (path, label) in enumerate(test_samples):
        if all_valid[i] > 0.0:
            records.append({
                "path": path,
                "label": int(label),
                "prob_uncal": float(probs_uncal[i]),
                "prob_cal": float(probs_cal[i]),
                "logit": float(all_logits[i]),
                "pred": int(probs_cal[i] >= threshold),
            })

    output_data = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "data_root": args.data_root,
            "temperature": temperature,
            "threshold": threshold,
            "n_samples": len(records),
        },
        "predictions": records,
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Exported %d predictions to %s", len(records), args.output_json)


if __name__ == "__main__":
    main()
