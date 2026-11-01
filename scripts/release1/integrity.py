"""Stage 1 of the Release-1 pipeline: splits integrity audit
(0 identity leakage, 0 DFD presence in train/val).

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from scripts.release1.common import compute_sha256
from src.dataset.loader import extract_identities

logger = logging.getLogger("release1_runner")

def verify_dataset_integrity(splits_path: str) -> dict[str, Any]:
    logger.info("Verifying dataset splits and identity partitioning: %s", splits_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]

    def get_actor_ids(crops: list[Any]) -> set[str]:
        actors = set()
        for c in crops:
            p = c[0].replace("\\", "/")
            parts = p.split("/")
            name = parts[1] if len(parts) > 1 else parts[0]
            a1, a2 = extract_identities(name)
            if a1:
                actors.add(a1)
            if a2:
                actors.add(a2)
        return actors

    train_ids = get_actor_ids(train_samples)
    val_ids = get_actor_ids(val_samples)
    test_ids = get_actor_ids(test_samples)

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"FATAL: Identity leakage detected! "
            f"Train&Val: {len(overlap_train_val)}, "
            f"Train&Test: {len(overlap_train_test)}, "
            f"Val&Test: {len(overlap_val_test)}"
        )

    # Verify zero DFD samples in train or val
    for p, _ in train_samples + val_samples:
        norm = p.replace("\\", "/").lower()
        if "__" in norm or "/00" in norm and len(norm.split("/")[1]) == 5:
            raise ValueError(f"FATAL: DFD sample leaked into train/val: {p}")

    train_labels = [int(s[1]) for s in train_samples]
    val_labels = [int(s[1]) for s in val_samples]
    test_labels = [int(s[1]) for s in test_samples]

    ledger = {
        "train_total": len(train_samples),
        "train_real": len(train_labels) - sum(train_labels),
        "train_fake": sum(train_labels),
        "val_total": len(val_samples),
        "val_real": len(val_labels) - sum(val_labels),
        "val_fake": sum(val_labels),
        "test_total": len(test_samples),
        "test_real": len(test_labels) - sum(test_labels),
        "test_fake": sum(test_labels),
        "train_actors": len(train_ids),
        "val_actors": len(val_ids),
        "test_actors": len(test_ids),
        "splits_sha256": compute_sha256(splits_path),
    }

    logger.info(
        "Splits Verified: Train=%d (%d R / %d F), Val=%d (%d R / %d F), Test=%d (%d R / %d F) | Zero Leakage Guaranteed",
        ledger["train_total"],
        ledger["train_real"],
        ledger["train_fake"],
        ledger["val_total"],
        ledger["val_real"],
        ledger["val_fake"],
        ledger["test_total"],
        ledger["test_real"],
        ledger["test_fake"],
    )
    return ledger


# ---------------------------------------------------------------------------
# Stage 2: Dual-Stream Backbone Training with Macro-AUC Validation
# ---------------------------------------------------------------------------

