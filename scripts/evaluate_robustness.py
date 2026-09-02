"""Robustness evaluation script for Dual-Stream Deepfake Detector."""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any, Optional

import cv2
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import load_config
from src.dataset.loader import dedupe_split
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 256


class RobustnessDataset(Dataset):
    """Dataset loader applying optional image degradation prior to normalization."""

    def __init__(
        self,
        samples: list[tuple[str, float]],
        root_dir: str,
        degradation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.samples = samples
        self.root_dir = root_dir
        self.degradation_fn = degradation_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if self.degradation_fn is not None:
                rgb = self.degradation_fn(rgb)
        except (OSError, ValueError, cv2.error):
            valid_flag = 0.0
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (
            tensor,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(valid_flag, dtype=torch.float32),
        )


def jpeg_fn(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    """JPEG compression degradation at specified quality level (0-100)."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return fn


def blur_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Gaussian blur degradation at specified kernel sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        ksize = int(6 * sigma + 1) | 1
        return cv2.GaussianBlur(rgb, (ksize, ksize), sigma)

    return fn


def noise_fn(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Additive Gaussian noise degradation at specified pixel-space sigma."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*rgb.shape).astype(np.float32) * sigma
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return fn


def downscale_fn(scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """Downscale and re-upsample image back to original resolution."""

    def fn(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = cv2.resize(
            rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return fn


def run_eval(
    model: torch.nn.Module,
    samples: list[tuple[str, float]],
    root_dir: str,
    degradation_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    threshold: float,
    temperature: float,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
) -> tuple[float, float, int]:
    """Run an inference pass over evaluation samples under specified degradation."""
    if max_samples:
        reals = [s for s in samples if s[1] == 0]
        fakes = [s for s in samples if s[1] == 1]
        n_each = max_samples // 2
        capped = reals[:n_each] + fakes[:n_each]
    else:
        capped = samples

    dataset = RobustnessDataset(capped, root_dir, degradation_fn)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_probs: list[float] = []
    all_labels: list[float] = []
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            for images, labels, valid_flags in loader:
                mask = valid_flags.bool()
                if not mask.any():
                    continue
                logits = model(images[mask].to(device)).squeeze(-1).float()
                probs = torch.sigmoid(logits / temperature).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels[mask].numpy().tolist())

    if len(set(all_labels)) < 2:
        logger.warning("Only one class present in evaluated samples — AUC undefined.")
        return float("nan"), float("nan"), len(all_labels)

    auc = float(roc_auc_score(all_labels, all_probs))
    preds = (np.array(all_probs) > threshold).astype(int)
    f1 = float(f1_score(all_labels, preds, zero_division=0))
    return auc, f1, len(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation for Dual-Stream Deepfake Detector.")
    parser.add_argument("--checkpoint", required=True, help="Path to dual_stream_calibrated.pth")
    parser.add_argument("--data_root", required=True, help="Dataset root containing splits.json")
    parser.add_argument("--output_json", default="robustness_results.json", help="Output path for JSON results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap test samples for fast local runs (e.g. 500)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    splits_path = os.path.join(args.data_root, "splits.json")
    if os.path.exists("/kaggle/working/splits.json"):
        splits_path = "/kaggle/working/splits.json"
    elif os.path.exists("./splits.json"):
        splits_path = "./splits.json"
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json not found at {splits_path}")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_samples = dedupe_split(splits["test"])
    logger.info(f"Test split: {len(test_samples)} samples (capped to {args.max_samples or 'all'})")

    config = load_config()
    backbone = config.get("model", {}).get("backbone", "convnext_small")
    model = HybridDeepfakeDetector(
        backbone_name=backbone, pretrained=False, use_fft_branch=True, config=config
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    threshold = float(checkpoint.get("optimal_threshold", 0.5))
    temperature = float(checkpoint.get("temperature", 1.0))
    logger.info(f"Checkpoint loaded. Threshold={threshold:.4f}, Temperature={temperature:.4f}")

    sweeps: list[tuple[str, list[tuple[str, Optional[Callable[[np.ndarray], np.ndarray]]]]]] = [
        (
            "JPEG Compression",
            [
                ("Clean (no degradation)", None),
                ("Q=100", jpeg_fn(100)),
                ("Q=90", jpeg_fn(90)),
                ("Q=70", jpeg_fn(70)),
                ("Q=50", jpeg_fn(50)),
                ("Q=30", jpeg_fn(30)),
            ],
        ),
        (
            "Gaussian Blur",
            [
                ("Clean (no degradation)", None),
                ("σ=0.5", blur_fn(0.5)),
                ("σ=1.5", blur_fn(1.5)),
                ("σ=3.0", blur_fn(3.0)),
            ],
        ),
        (
            "Gaussian Noise",
            [
                ("Clean (no degradation)", None),
                ("σ=5", noise_fn(5)),
                ("σ=15", noise_fn(15)),
                ("σ=30", noise_fn(30)),
            ],
        ),
        (
            "Downscaling",
            [
                ("Clean (no degradation)", None),
                ("0.75×", downscale_fn(0.75)),
                ("0.50×", downscale_fn(0.50)),
                ("0.25×", downscale_fn(0.25)),
            ],
        ),
    ]

    all_results: dict[str, list[dict[str, Any]]] = {}
    for sweep_name, levels in sweeps:
        print(f"\n{'='*60}")
        print(f"  {sweep_name}")
        print(f"{'='*60}")
        print(f"  {'Level':<28} {'AUC':>8}  {'F1':>8}  {'N':>7}")
        print(f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*7}")

        sweep_results: list[dict[str, Any]] = []
        for label, deg_fn in levels:
            auc, f1, n = run_eval(
                model=model,
                samples=test_samples,
                root_dir=args.data_root,
                degradation_fn=deg_fn,
                threshold=threshold,
                temperature=temperature,
                device=device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
            print(f"  {label:<28} {auc:>8.4f}  {f1:>8.4f}  {n:>7}")
            sweep_results.append({"level": label, "auc": auc, "f1": f1, "n_samples": n})

        all_results[sweep_name] = sweep_results

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()

