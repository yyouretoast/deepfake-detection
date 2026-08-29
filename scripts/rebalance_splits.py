"""Rebalance dataset splits into stratified 80% Train / 10% Val / 10% Test with zero leakage."""

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_dual_stream_ddp import find_dataset_root
from src.dataset.loader import dedupe_split, perform_graph_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rebalance(data_dir: str = None, train_ratio: float = 0.80, val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42) -> None:
    data_root = find_dataset_root(data_dir)
    splits_path = os.path.join(data_root, "splits.json")
    logger.info(f"Loading existing splits from: {splits_path}")

    with open(splits_path, "r") as f:
        old_splits = json.load(f)

    all_samples = []
    for split_name in ["train", "val", "test"]:
        if split_name in old_splits:
            all_samples.extend(old_splits[split_name])

    all_samples = dedupe_split(all_samples)
    logger.info(f"Total Unique Samples Found: {len(all_samples):,}")

    train_samples, val_samples, test_samples = perform_graph_split(
        all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    def get_stats(samples):
        n_real = sum(1 for s in samples if s[1] == 0)
        n_fake = sum(1 for s in samples if s[1] == 1)
        return len(samples), n_real, n_fake

    t_tot, t_r, t_f = get_stats(train_samples)
    v_tot, v_r, v_f = get_stats(val_samples)
    te_tot, te_r, te_f = get_stats(test_samples)

    logger.info("=" * 65)
    logger.info(f"NEW STRATIFIED ZERO-LEAKAGE SPLIT BREAKDOWN:")
    logger.info(f"  TRAIN: {t_tot:,} samples ({t_tot/len(all_samples)*100:.1f}%) | Real: {t_r:,}, Fake: {t_f:,} (Ratio: {t_f/max(1, t_r):.2f})")
    logger.info(f"  VAL:   {v_tot:,} samples ({v_tot/len(all_samples)*100:.1f}%) | Real: {v_r:,}, Fake: {v_f:,} (Ratio: {v_f/max(1, v_r):.2f})")
    logger.info(f"  TEST:  {te_tot:,} samples ({te_tot/len(all_samples)*100:.1f}%) | Real: {te_r:,}, Fake: {te_f:,} (Ratio: {te_f/max(1, te_r):.2f})")
    logger.info("=" * 65)

    new_splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    # Save to working dir / data root
    out_paths = [splits_path]
    if os.path.exists("/kaggle/working"):
        out_paths.append("/kaggle/working/splits.json")
        out_paths.append("/kaggle/working/repo/splits.json")

    for p in out_paths:
        try:
            with open(p, "w") as f:
                json.dump(new_splits, f, indent=2)
            logger.info(f"Saved balanced splits to: {p}")
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebalance splits.json into 80/10/10 stratified zero-leakage splits")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing splits.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    rebalance(data_dir=args.data_dir, seed=args.seed)
