import logging
import os
import random
import re
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_identities(
    filename: str, metadata_map: Optional[dict[str, tuple[str, str]]] = None
) -> tuple[str, str]:
    """Extract actor/source identity IDs from video or crop filenames."""
    if metadata_map and filename in metadata_map:
        return metadata_map[filename]

    parts = os.path.normpath(filename).replace("\\", "/").split("/")
    target = parts[-2] if len(parts) > 1 else parts[-1]
    # Strip frame-counter suffixes (e.g. "_f0001", "_frame012") before parsing identities
    clean_base = re.sub(r"_(?:f|frame)\d+", "", target, flags=re.IGNORECASE).split(".")[0]

    # Priority 1: explicit id{N}_id{M} pattern (most reliable)
    match_alpha = re.search(r"(id\d+)_(id\d+)", clean_base)
    if match_alpha:
        return match_alpha.group(1), match_alpha.group(2)

    # Priority 2: numeric pair pattern — but ONLY when neither number looks like a
    # zero-padded frame counter (≥4 digits). Actor IDs in FF++ / Celeb-DF are ≤3 digits.
    match_num = re.search(r"(\d+)_(\d+)", clean_base)
    if match_num:
        g1, g2 = match_num.group(1), match_num.group(2)
        if len(g1) <= 3 and len(g2) <= 3:  # noqa: PLR2004 — actor IDs are ≤3 digits
            return g1, g2
        # Fall through if either looks like a frame counter
        logger.debug(
            "Skipping numeric pair '%s_%s' in '%s': one or both numbers look like frame counters (≥4 digits).",
            g1, g2, clean_base,
        )

    # Priority 3: single id pattern (Celeb-real videos like "id0_0000.mp4")
    match_single_id = re.search(r"(id\d+)", clean_base)
    if match_single_id:
        id_str = match_single_id.group(1)
        return id_str, id_str

    # Priority 4: single numeric id (real-face videos like "000.mp4" or "00000.mp4")
    match_single = re.search(r"(\d+)", clean_base)
    if match_single:
        id_str = match_single.group(1)
        return id_str, id_str

    logger.warning(
        "Failed to isolate actor identity pairs via regex for '%s' (parsed base: '%s'). "
        "Fallback identity used; potential dataset split leakage risk.",
        filename,
        clean_base,
    )
    return clean_base, clean_base


def dedupe_split(split_list: list[Any]) -> list[Any]:
    """Remove duplicate sample path entries from dataset split lists."""
    seen, deduped = set(), []
    for entry in split_list:
        path = entry[0] if isinstance(entry, (list, tuple)) else entry
        if path not in seen:
            seen.add(path)
            deduped.append(entry)
    return deduped


def perform_graph_split(
    samples: Any,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Partition dataset into train/val/test splits using networkx.Graph connected component identity logic."""
    val_ratio = kwargs.get("val_size", val_ratio)
    test_ratio = kwargs.get("test_size", test_ratio)

    if not samples:
        return [], [], []

    is_string_list = isinstance(samples[0], str)
    if is_string_list:
        normalized_samples = [(s, 0) for s in samples]
    else:
        normalized_samples = list(samples)

    parsed_samples = []
    video_map: dict[str, list[tuple[str, int]]] = {}

    for item in normalized_samples:
        path, label = item[0], item[1]
        first_path = path[0] if isinstance(path, (list, tuple)) else path
        id1, id2 = extract_identities(first_path)
        parsed_samples.append((path, label, id1, id2))
        video_map.setdefault(id1, []).append((path, label))

    G = nx.Graph()
    for _, _, id1, id2 in parsed_samples:
        G.add_node(id1)
        G.add_node(id2)
        G.add_edge(id1, id2)

    components = [sorted(list(c)) for c in nx.connected_components(G)]

    # Stratify components by dominant label to prevent class skew
    comp_real_stats = []
    comp_fake_stats = []

    for comp in components:
        comp_set = set(comp)
        comp_samples = [s for s in parsed_samples if s[2] in comp_set or s[3] in comp_set]
        n_real = sum(1 for s in comp_samples if s[1] == 0)
        n_fake = sum(1 for s in comp_samples if s[1] == 1)
        total_comp = len(comp_samples)

        if n_fake > 0:
            comp_fake_stats.append((comp, total_comp, n_fake, n_real))
        else:
            comp_real_stats.append((comp, total_comp, n_fake, n_real))

    def partition_component_list(comp_list: list[tuple[Any, int, int, int]]) -> tuple[set[str], set[str], set[str]]:
        rng = random.Random(seed)
        # Deterministically shuffle before greedy packing
        shuffled = list(comp_list)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: x[1], reverse=True)

        tot_samples = sum(x[1] for x in shuffled)
        tgt_val = max(1, int(tot_samples * val_ratio))
        tgt_test = max(1, int(tot_samples * test_ratio))

        val_c, test_c, train_c = set(), set(), set()
        c_val, c_test = 0, 0

        for comp, n_s, _, _ in shuffled:
            if c_val + n_s <= tgt_val or (c_val == 0 and len(shuffled) > 2):
                val_c.update(comp)
                c_val += n_s
            elif c_test + n_s <= tgt_test or (c_test == 0 and len(shuffled) > 2):
                test_c.update(comp)
                c_test += n_s
            else:
                train_c.update(comp)

        return train_c, val_c, test_c

    train_fake_c, val_fake_c, test_fake_c = partition_component_list(comp_fake_stats)
    train_real_c, val_real_c, test_real_c = partition_component_list(comp_real_stats)

    train_comps = train_fake_c | train_real_c
    val_comps = val_fake_c | val_real_c
    test_comps = test_fake_c | test_real_c

    train_samples, val_samples, test_samples = [], [], []
    for path, label, id1, id2 in parsed_samples:
        if id1 in val_comps or id2 in val_comps:
            val_samples.append((path, label))
        elif id1 in test_comps or id2 in test_comps:
            test_samples.append((path, label))
        elif id1 in train_comps or id2 in train_comps:
            train_samples.append((path, label))
        else:
            logger.warning(
                "Sample %s with identities (%s, %s) not assigned to any component partition; routing to train.",
                path, id1, id2
            )
            train_samples.append((path, label))

    if is_string_list:
        return (
            [s[0] for s in train_samples],
            [s[0] for s in val_samples],
            [s[0] for s in test_samples],
        )

    return train_samples, val_samples, test_samples


def get_transforms(
    img_size: int = 256, hardened: bool = True
) -> tuple[Optional[Any], Optional[Any]]:
    """Build albumentations train and validation transform pipelines for target resolution [H, W]."""
    if not HAS_ALBUMENTATIONS:
        return None, None

    if hardened:
        # Forensic-safe augmentation pipeline:
        # Protects frequency-domain forensic features from artificial sinc-leakage (no CoarseDropout)
        # and aliasing grids (no Downscale), while maintaining realistic compression and blur robustness.
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=10, p=0.25),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.04, p=0.20),
            A.ImageCompression(quality_range=(65, 95), p=0.30),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.2), p=0.20),
            ToTensorV2(),
        ])
    else:
        # Conservative legacy pipeline
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            A.ImageCompression(quality_range=(85, 100), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.2, 0.6), p=0.15),
            ToTensorV2(),
        ])

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])

    return train_transform, eval_transform


def group_video_sequences(
    samples: list[tuple[str, int]], min_frames: int = 4
) -> list[tuple[list[str], int]]:
    """Groups flat frame crop samples into video sequence lists by parent directory."""
    video_map: dict[str, tuple[list[str], int]] = {}
    for path, label in samples:
        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir not in video_map:
            video_map[parent_dir] = ([], label)
        video_map[parent_dir][0].append(path)

    grouped: list[tuple[list[str], int]] = []
    for paths, label in video_map.values():
        if len(paths) >= min_frames:
            # Sort frames alphabetically to preserve chronological order
            grouped.append((sorted(paths), label))
    return grouped


def load_image_rgb(path: str) -> np.ndarray:
    """Load image from disk and return RGB numpy array of shape [H, W, 3]."""
    if os.path.exists(path):
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.zeros((256, 256, 3), dtype=np.uint8)


class SequenceVideoDataset(Dataset):
    """Dataset for temporal video sequences returning frame batch [T, 3, H, W] in [0, 1], label, and padding mask [T]."""

    def __init__(
        self,
        video_samples: list[tuple[list[str], int]],
        transform: Optional[Any] = None,
        seq_len: int = 8,
        is_train: bool = False,
    ) -> None:
        self.video_samples = video_samples
        self.transform = transform
        if HAS_ALBUMENTATIONS and self.transform is not None:
            if isinstance(self.transform, A.Compose) and not isinstance(
                self.transform, A.ReplayCompose
            ):
                self.transform = A.ReplayCompose(self.transform.transforms)
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.video_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_paths, label = self.video_samples[idx]
        n_frames = len(frame_paths)

        if n_frames >= self.seq_len:
            if self.is_train and n_frames > self.seq_len:
                start_idx = int(np.random.randint(0, n_frames - self.seq_len + 1))
            else:
                start_idx = max(0, (n_frames - self.seq_len) // 2)
            selected_paths = frame_paths[start_idx : start_idx + self.seq_len]
            n_pad = 0
        else:
            selected_paths = frame_paths[:n_frames]
            n_pad = self.seq_len - n_frames

        try:
            img_rgb_list = [load_image_rgb(p) for p in selected_paths]
            if n_pad > 0 and len(img_rgb_list) > 0:
                last_frame = img_rgb_list[-1]
                img_rgb_list.extend([last_frame] * n_pad)
        except (OSError, cv2.error, ValueError, TypeError, RuntimeError) as e:
            logger.debug("Video frame sequence loading failed for %s: %s", frame_paths, e)
            dummy = np.zeros((256, 256, 3), dtype=np.uint8)
            img_rgb_list = [dummy] * self.seq_len

        frames = []
        if self.transform is not None and len(img_rgb_list) > 0:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.ReplayCompose):
                first_res = self.transform(image=img_rgb_list[0])
                first_tensor = first_res["image"].float() / 255.0 if first_res["image"].dtype == torch.uint8 else first_res["image"].float()
                frames.append(first_tensor)
                if "replay" in first_res:
                    replay_saved = first_res["replay"]
                    for img_rgb in img_rgb_list[1:]:
                        aug_res = A.ReplayCompose.replay(replay_saved, image=img_rgb)
                        aug_tensor = aug_res["image"].float() / 255.0 if aug_res["image"].dtype == torch.uint8 else aug_res["image"].float()
                        frames.append(aug_tensor)
            else:
                for img_rgb in img_rgb_list:
                    if hasattr(self.transform, "__call__"):
                        frames.append(self.transform(Image.fromarray(img_rgb)))
        else:
            for img_rgb in img_rgb_list:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                frames.append(img_tensor)


        seq_tensor = torch.stack(frames, dim=0) if frames else torch.zeros(self.seq_len, 3, 256, 256)

        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        if n_pad > 0:
            padding_mask[self.seq_len - n_pad :] = True

        return seq_tensor, torch.tensor(label, dtype=torch.long), padding_mask


