"""Shared utilities for the Release-1 pipeline: stratum classification, logging,
seed control, SHA-256 hashing, and balanced/stratified subsampling.

Bodies are byte-exact extractions from ``scripts/run_release1_kaggle.py``
(audit recommendation R5); CLI paths, JSON keys, and behavior are unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
from typing import Any

import cv2
import numpy as np
import torch

from src.dataset.domains import DomainClassifier, ManipulationDomain


def get_sample_stratum(rel_path: str, label: int) -> str:
    """Classifies a crop into its source domain and label stratum for balanced sampling."""
    norm = rel_path.replace("\\", "/").lower()
    parts = norm.split("/")
    folder = parts[1] if len(parts) > 1 else parts[0]
    if "celeb" in norm or folder.startswith("id"):
        src = "celeb"
    elif "__" in folder or (len(folder) == 5 and folder.isdigit()) or "dfd" in norm:
        src = "dfd"
    else:
        src = "ffpp"
    return f"{src}_{int(label)}"
def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "release1_pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "missing"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

def subsample_balanced(
    samples: list[Any], n: int, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    fakes = [s for s in samples if float(s[1]) == 1.0]
    reals = [s for s in samples if float(s[1]) == 0.0]
    n_half = n // 2
    n_fakes = min(n_half, len(fakes))
    n_reals = min(n - n_fakes, len(reals))
    selected = rng.sample(fakes, n_fakes) + rng.sample(reals, n_reals)
    rng.shuffle(selected)
    return selected


def subsample_stratified_val(
    samples: list[Any], n_per_source: int = 50, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    ffpp = [s for s in samples if not get_sample_stratum(s[0], int(s[1])).startswith("celeb")]
    celeb = [s for s in samples if get_sample_stratum(s[0], int(s[1])).startswith("celeb")]
    selected = subsample_balanced(ffpp, n_per_source, seed=seed) + subsample_balanced(celeb, n_per_source, seed=seed)
    rng.shuffle(selected)
    return selected


def subsample_stratified_test(
    samples: list[Any], n_per_source: int = 34, seed: int = 42
) -> list[Any]:
    rng = random.Random(seed)
    by_source: dict[str, list[Any]] = {"ffpp": [], "celeb": [], "dfd": []}
    for s in samples:
        p = s[0].replace("\\", "/")
        parts = p.split("/")
        dinfo = DomainClassifier.classify(p)
        if "dfd" in dinfo.domain.value or ("real" in parts[0] and len(parts[1]) == 5 and parts[1].isdigit()):
            by_source["dfd"].append(s)
        elif dinfo.domain == ManipulationDomain.CELEB_DF or ("real" in parts[0] and "id" in parts[1]):
            by_source["celeb"].append(s)
        else:
            by_source["ffpp"].append(s)

    selected = []
    for src_samples in by_source.values():
        if src_samples:
            selected.extend(subsample_balanced(src_samples, n_per_source, seed=seed))
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Stage 1: Dataset Integrity & Leakage Verification
# ---------------------------------------------------------------------------

