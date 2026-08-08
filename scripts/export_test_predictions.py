"""
Exports per-sample test set predictions for downstream plot generation.

Runs one clean inference pass over the held-out test split and saves raw
probabilities, calibrated probabilities, and ground truth labels to JSON.
The output file is consumed by scripts/generate_benchmark_plots.py to
produce ROC curves and ECE reliability diagrams.

Usage:
    python scripts/export_test_predictions.py \\
        --checkpoint /path/to/dual_stream_calibrated.pth \\
        --data_root /path/to/dataset \\
        --output_json test_predictions.json
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config
from src.utils.checkpoint import clean_state_dict
from src.dataset.loader import dedupe_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Self-contained dataset (same as evaluate_robustness.py).
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, samples, root_dir):
        self.samples = samples
        self.root_dir = root_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_rel, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path_rel)
        valid_flag = 1.0
        try:
            bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Image read failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
                rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            valid_flag = 0.0
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (
            tensor,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(valid_flag, dtype=torch.float32),
        )



# ---------------------------------------------------------------------------
# Prediction Exporter
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export per-sample test predictions for ROC/ECE plot generation."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--data_root", required=True,
                        help="Dataset root containing splits.json")
    parser.add_argument("--output_json", default="test_predictions.json",
                        help="Output path for predictions JSON")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    # Load splits
    splits_path = os.path.join(args.data_root, "splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json not found at {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    test_samples = dedupe_split(splits["test"])
    logging.info("Test split: %d samples", len(test_samples))

    # Load model
    config = load_config()
    backbone = config.get("model", {}).get("backbone", "convnext_small")
    model = HybridDeepfakeDetector(
        backbone_name=backbone, pretrained=False, use_fft_branch=True, config=config
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()

    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    temperature = float(checkpoint.get("temperature", 1.0))
    logging.info("Checkpoint loaded. Threshold=%.4f, Temperature=%.4f", threshold, temperature)

    # Inference
    dataset = TestDataset(test_samples, args.data_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for images, labels, valid_flags in loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                logits = model(images[mask].to(device)).squeeze(-1).float()
                all_logits.extend(logits.cpu().numpy().tolist())
                all_labels.extend(labels[mask].numpy().tolist())

    logits_arr = np.array(all_logits, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int32)

    probs_raw = (1.0 / (1.0 + np.exp(-logits_arr))).tolist()
    probs_cal = (1.0 / (1.0 + np.exp(-logits_arr / temperature))).tolist()

    output = {
        "probs_raw": probs_raw,
        "probs_cal": probs_cal,
        "labels": labels_arr.tolist(),
        "temperature": temperature,
        "threshold": threshold,
        "n_samples": len(all_labels),
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f)

    logging.info(
        "Saved %d predictions to %s (%.1f MB)",
        len(all_labels),
        args.output_json,
        os.path.getsize(args.output_json) / 1e6,
    )


if __name__ == "__main__":
    main()
