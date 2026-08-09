import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

from src.config import load_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_identities(
    filename: str, metadata_map: Optional[Dict[str, Tuple[str, str]]] = None
) -> Tuple[str, str]:
    """Extract actor/source identity IDs from video or crop filenames."""
    if metadata_map and filename in metadata_map:
        return metadata_map[filename]

    parts = os.path.normpath(filename).replace("\\", "/").split("/")
    target = parts[-2] if len(parts) > 1 else parts[-1]
    clean_base = re.sub(r"_(?:f|frame)\d+", "", target, flags=re.IGNORECASE).split(".")[0]

    match_alpha = re.search(r"(id\d+)_(id\d+)", clean_base)
    if match_alpha:
        return match_alpha.group(1), match_alpha.group(2)

    match_num = re.search(r"(\d+)_(\d+)", clean_base)
    if match_num:
        return match_num.group(1), match_num.group(2)

    match_single = re.search(r"(\d+)", clean_base)
    if match_single:
        id_str = match_single.group(1)
        return id_str, id_str

    return clean_base, clean_base


def extract_video_id(filename: str) -> str:
    """Extract primary video/actor identifier from filename."""
    return extract_identities(filename)[0]


def dedupe_split(split_list: List[Any]) -> List[Any]:
    """Remove duplicate sample path entries from dataset split lists."""
    seen, deduped = set(), []
    for entry in split_list:
        path = entry[0] if isinstance(entry, (list, tuple)) else entry
        if path not in seen:
            seen.add(path)
            deduped.append(entry)
    return deduped


def group_samples_by_video(
    samples: List[Tuple[str, int]]
) -> List[Tuple[List[str], int]]:
    """Group individual frame samples into video sequence lists by identity/video ID."""
    if not samples:
        return []
    video_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in samples:
        path, label = item[0], item[1]
        vid_id = extract_video_id(path) if isinstance(path, str) else extract_video_id(path[0])
        key = (vid_id, label)
        if key not in video_map:
            video_map[key] = {"paths": [], "label": label}
        if isinstance(path, (list, tuple)):
            video_map[key]["paths"].extend(path)
        else:
            video_map[key]["paths"].append(path)

    grouped = []
    for key in sorted(video_map.keys(), key=lambda k: str(k[0])):
        paths = sorted(video_map[key]["paths"])
        grouped.append((paths, video_map[key]["label"]))
    return grouped


def get_dir_hash(dir_path: str) -> str:
    """Compute SHA256 checksum of file paths within a directory for split manifest validation."""
    hasher = hashlib.sha256()
    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            hasher.update(f.encode("utf-8"))
    return hasher.hexdigest()


def perform_graph_split(
    samples: Any,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[Any, Any, Any]:
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
    video_map: Dict[str, List[Tuple[str, int]]] = {}

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
    comp_stats = []
    for comp in components:
        comp_set = set(comp)
        n_samples = sum(
            1 for s in parsed_samples if s[2] in comp_set or s[3] in comp_set
        )
        comp_stats.append((comp, n_samples))

    comp_stats.sort(key=lambda x: x[1], reverse=True)
    total_samples = len(parsed_samples)
    target_val = max(1, int(total_samples * val_ratio))
    target_test = max(1, int(total_samples * test_ratio))

    val_comps, test_comps, train_comps = set(), set(), set()
    curr_val, curr_test = 0, 0

    for comp, n_s in comp_stats:
        if curr_val + n_s <= target_val or (curr_val == 0 and len(comp_stats) > 2):
            val_comps.update(comp)
            curr_val += n_s
        elif curr_test + n_s <= target_test or (curr_test == 0 and len(comp_stats) > 2):
            test_comps.update(comp)
            curr_test += n_s
        else:
            train_comps.update(comp)

    train_samples, val_samples, test_samples = [], [], []
    for path, label, id1, id2 in parsed_samples:
        if id1 in train_comps or id2 in train_comps:
            train_samples.append((path, label))
        elif id1 in val_comps or id2 in val_comps:
            val_samples.append((path, label))
        else:
            test_samples.append((path, label))

    if is_string_list:
        return (
            [s[0] for s in train_samples],
            [s[0] for s in val_samples],
            [s[0] for s in test_samples],
        )

    return train_samples, val_samples, test_samples


def get_transforms(img_size: int = 256) -> Tuple[Optional[Any], Optional[Any]]:
    """Build albumentations train and validation transform pipelines for target resolution [H, W]."""
    if not HAS_ALBUMENTATIONS:
        return None, None

    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.OneOf([
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return train_transform, eval_transform


def load_image_rgb(path: str) -> np.ndarray:
    """Load image from disk and return RGB numpy array of shape [H, W, 3]."""
    if os.path.exists(path):
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.zeros((256, 256, 3), dtype=np.uint8)


class DeepfakeDataset(Dataset):
    """Dataset for single-frame face crops returning normalized tensor [3, H, W] and integer label."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Any] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mean_tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std_tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img_rgb = load_image_rgb(path)

        if self.transform is not None:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
                augmented = self.transform(image=img_rgb)
                return augmented["image"], label
            else:
                img_pil = Image.fromarray(img_rgb)
                return self.transform(img_pil), label

        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        norm_img = (tensor_img - self.mean_tensor) / self.std_tensor
        return norm_img, label


class SequenceVideoDataset(Dataset):
    """Dataset for temporal video sequences returning frame batch [T, 3, H, W], label, and padding mask [T]."""

    def __init__(
        self,
        video_samples: List[Tuple[List[str], int]],
        transform: Optional[Any] = None,
        seq_len: int = 8,
    ) -> None:
        self.video_samples = video_samples
        self.transform = transform
        if HAS_ALBUMENTATIONS and self.transform is not None:
            if isinstance(self.transform, A.Compose) and not isinstance(
                self.transform, A.ReplayCompose
            ):
                self.transform = A.ReplayCompose(self.transform.transforms)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.video_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_paths, label = self.video_samples[idx]
        n_frames = len(frame_paths)
        mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        if n_frames >= self.seq_len:
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
        except Exception as e:
            logging.debug("Video frame sequence loading failed for %s: %s", frame_paths, e)
            dummy = np.zeros((256, 256, 3), dtype=np.uint8)
            img_rgb_list = [dummy] * self.seq_len

        frames = []
        if self.transform is not None and len(img_rgb_list) > 0:
            if HAS_ALBUMENTATIONS and isinstance(self.transform, A.ReplayCompose):
                first_res = self.transform(image=img_rgb_list[0])
                frames.append(first_res["image"])
                if "replay" in first_res:
                    replay_saved = first_res["replay"]
                    for img_rgb in img_rgb_list[1:]:
                        aug_img = A.ReplayCompose.replay(replay_saved, image=img_rgb)["image"]
                        frames.append(aug_img)
            else:
                for img_rgb in img_rgb_list:
                    if hasattr(self.transform, "__call__"):
                        frames.append(self.transform(Image.fromarray(img_rgb)))
        else:
            for img_rgb in img_rgb_list:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor - mean_t) / std_t
                frames.append(img_tensor)

        if n_pad > 0:
            h = frames[0].shape[1] if frames else 256
            w = frames[0].shape[2] if frames else 256
            pad_frame = (torch.zeros(3, h, w) - mean_t) / std_t
            frames.extend([pad_frame] * n_pad)

        seq_tensor = torch.stack(frames, dim=0) if frames else torch.zeros(self.seq_len, 3, 256, 256)

        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        if n_pad > 0:
            padding_mask[self.seq_len - n_pad :] = True

        return seq_tensor, torch.tensor(label, dtype=torch.long), padding_mask


def _worker_init_fn(worker_id: int) -> None:
    """Worker initialization hook disabling OpenCV multithreading per worker process."""
    cv2.setNumThreads(0)


def create_dataloaders(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, DataLoader]:
    """Construct PyTorch DataLoaders for train, val, and test splits with class balancing."""
    if config is None:
        config = load_config()

    prep_cfg = config.get("preprocessing", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    cropped_dir = prep_cfg.get("cropped_frames_dir", paths_cfg.get("cropped_dir", "data/cropped"))
    img_size = prep_cfg.get("img_size", 256)
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)
    seed = train_cfg.get("seed", 42)

    samples = []
    manifest_path = os.path.join(cropped_dir, "splits.json")
    dir_hash = ""
    if os.path.exists(cropped_dir):
        dir_hash = get_dir_hash(cropped_dir)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if manifest.get("hash") == dir_hash:
                train_s = manifest["train"]
                val_s = manifest["val"]
                test_s = manifest["test"]
                train_samples = [(os.path.join(cropped_dir, p), lbl) for p, lbl in train_s]
                val_samples = [(os.path.join(cropped_dir, p), lbl) for p, lbl in val_s]
                test_samples = [(os.path.join(cropped_dir, p), lbl) for p, lbl in test_s]
                samples = train_samples + val_samples + test_samples

    if not samples and os.path.exists(cropped_dir):
        for root, _, files in os.walk(cropped_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, file)
                    label = 0 if "original" in full_path.lower() or "real" in full_path.lower() else 1
                    samples.append((full_path, label))

        if samples:
            train_samples, val_samples, test_samples = perform_graph_split(samples, seed=seed)
            try:
                with open(manifest_path, "w") as f:
                    json.dump({
                        "hash": dir_hash,
                        "train": [(os.path.relpath(p, cropped_dir), lbl) for p, lbl in train_samples],
                        "val": [(os.path.relpath(p, cropped_dir), lbl) for p, lbl in val_samples],
                        "test": [(os.path.relpath(p, cropped_dir), lbl) for p, lbl in test_samples],
                    }, f)
            except Exception as e:
                logging.warning("Failed to write splits manifest to %s: %s", manifest_path, e)

    if not samples:
        train_samples, val_samples, test_samples = [], [], []

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = DeepfakeDataset(train_samples, transform=train_transform)
    val_dataset = DeepfakeDataset(val_samples, transform=eval_transform)
    test_dataset = DeepfakeDataset(test_samples, transform=eval_transform)

    if train_samples:
        train_labels = [s[1] for s in train_samples]
        class_counts = np.maximum(np.bincount(train_labels), 1)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    else:
        sampler = None

    worker_init = _worker_init_fn if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


build_dataloaders = create_dataloaders


def create_sequence_dataloaders(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, DataLoader]:
    """Construct PyTorch DataLoaders for video frame sequence datasets [T, 3, H, W]."""
    if config is None:
        config = load_config()

    prep_cfg = config.get("preprocessing", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    cropped_dir = prep_cfg.get("cropped_frames_dir", paths_cfg.get("cropped_dir", "data/cropped"))
    img_size = prep_cfg.get("img_size", 256)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)
    seed = train_cfg.get("seed", 42)
    seq_len = train_cfg.get("seq_len", 8)

    samples = []
    if os.path.exists(cropped_dir):
        for root, _, files in os.walk(cropped_dir):
            for file in sorted(files):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, file)
                    label = 0 if "original" in full_path.lower() or "real" in full_path.lower() else 1
                    samples.append((full_path, label))

    if not samples:
        train_vids, val_vids, test_vids = [], [], []
    else:
        video_samples = group_samples_by_video(samples)
        train_vids, val_vids, test_vids = perform_graph_split(video_samples, seed=seed)

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = SequenceVideoDataset(train_vids, transform=train_transform, seq_len=seq_len)
    val_dataset = SequenceVideoDataset(val_vids, transform=eval_transform, seq_len=seq_len)
    test_dataset = SequenceVideoDataset(test_vids, transform=eval_transform, seq_len=seq_len)

    if train_vids:
        train_labels = [v[1] for v in train_vids]
        class_counts = np.maximum(np.bincount(train_labels), 1)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    else:
        sampler = None

    worker_init = _worker_init_fn if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
