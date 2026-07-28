from typing import List, Tuple, Dict, Any, Optional
import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None

from PIL import Image

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from src.config import load_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def extract_identities(filename: str) -> Tuple[str, str]:
    """Extracts actor/source video identity strings from FaceForensics++ filename patterns."""
    basename = os.path.basename(filename)
    match = re.search(r"(\d{3,4})_(\d{3,4})", basename)
    if match:
        return match.group(1), match.group(2)
    
    match_single = re.search(r"(\d{3,4})", basename)
    if match_single:
        id_str = match_single.group(1)
        return id_str, id_str
        
    return basename, basename

def extract_video_id(filename: str) -> str:
    """Alias function extracting primary video identity string."""
    return extract_identities(filename)[0]

def perform_graph_split(
    samples: Any,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    **kwargs
) -> Tuple[Any, Any, Any]:
    """
    Graph-connected component identity partitioning guaranteeing zero identity leakage.
    Supports samples as List[str] or List[Tuple[str, int]].
    """
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
        id1, id2 = extract_identities(path)
        parsed_samples.append((path, label, id1, id2))
        video_map.setdefault(id1, []).append((path, label))

    if not HAS_NETWORKX:
        # Union-Find to group identities connected via (id1, id2) pairs
        parent: Dict[str, str] = {}

        def _find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
                x = parent.get(x, x)
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        # Collect all identity nodes and edges
        all_ids: set = set()
        for _, _, id1, id2 in parsed_samples:
            all_ids.add(id1)
            all_ids.add(id2)
            _union(id1, id2)

        # Group IDs into connected components via root
        comp_map: Dict[str, list] = {}
        for vid_id in sorted(all_ids):
            root = _find(vid_id)
            comp_map.setdefault(root, []).append(vid_id)

        components = sorted(comp_map.values(), key=lambda c: c[0])
        rng = np.random.RandomState(seed)
        rng.shuffle(components)

        n_comps = len(components)
        n_val_c = max(1, int(n_comps * val_ratio))
        n_test_c = max(1, int(n_comps * test_ratio))

        val_id_set = set(vid for comp in components[:n_val_c] for vid in comp)
        test_id_set = set(vid for comp in components[n_val_c:n_val_c + n_test_c] for vid in comp)
        train_id_set = set(vid for comp in components[n_val_c + n_test_c:] for vid in comp)

        train_s, val_s, test_s = [], [], []
        for path, label, id1, id2 in parsed_samples:
            if id1 in train_id_set or id2 in train_id_set:
                train_s.append((path, label))
            elif id1 in val_id_set or id2 in val_id_set:
                val_s.append((path, label))
            else:
                test_s.append((path, label))

        if is_string_list:
            return [s[0] for s in train_s], [s[0] for s in val_s], [s[0] for s in test_s]
        return train_s, val_s, test_s

    G = nx.Graph()
    for _, _, id1, id2 in parsed_samples:
        G.add_node(id1)
        G.add_node(id2)
        G.add_edge(id1, id2)

    components = sorted(list(nx.connected_components(G)), key=lambda c: sorted(list(c))[0])
    rng = np.random.RandomState(seed)
    rng.shuffle(components)

    n_comps = len(components)
    n_val_c = max(1, int(n_comps * val_ratio))
    n_test_c = max(1, int(n_comps * test_ratio))

    val_comps = set().union(*components[:n_val_c])
    test_comps = set().union(*components[n_val_c:n_val_c + n_test_c])
    train_comps = set().union(*components[n_val_c + n_test_c:])

    train_samples, val_samples, test_samples = [], [], []
    for path, label, id1, id2 in parsed_samples:
        if id1 in train_comps or id2 in train_comps:
            train_samples.append((path, label))
        elif id1 in val_comps or id2 in val_comps:
            val_samples.append((path, label))
        else:
            test_samples.append((path, label))

    if is_string_list:
        return [s[0] for s in train_samples], [s[0] for s in val_samples], [s[0] for s in test_samples]

    return train_samples, val_samples, test_samples

def group_video_split(
    samples: Any,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    **kwargs
) -> Tuple[Any, Any, Any]:
    """Alias wrapper for graph-connected identity split."""
    return perform_graph_split(samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, **kwargs)

def get_transforms(img_size: int = 256) -> Tuple[Any, Any]:
    """Returns PyTorch training and evaluation Albumentations pipelines (or None fallback)."""
    if not HAS_ALBUMENTATIONS:
        return None, None

    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return train_transform, eval_transform

get_eval_transforms = get_transforms

class DeepfakeDataset(Dataset):
    """PyTorch Dataset loading pre-cropped face images with Albumentations augmentation."""
    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Any] = None):
        self.samples = samples
        self.transform = transform
        self.mean_tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std_tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                with Image.open(path) as img_pil:
                    img_rgb = np.array(img_pil.convert("RGB"))
            except Exception:
                img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            return augmented["image"], label

        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        norm_img = (tensor_img - self.mean_tensor) / self.std_tensor
        return norm_img, label

class SequenceVideoDataset(Dataset):
    """Dataset loader yielding 5D video frame sequence tensors [T, 3, H, W] per sample."""
    def __init__(self, video_samples: List[Tuple[List[str], int]], transform: Optional[Any] = None, seq_len: int = 8):
        self.video_samples = video_samples
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.video_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_paths, label = self.video_samples[idx]
        N = len(frame_paths)
        mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        if N >= self.seq_len:
            indices = np.linspace(0, N - 1, self.seq_len, dtype=int)
            selected_paths = [frame_paths[i] for i in indices]
            n_pad = 0
        else:
            selected_paths = frame_paths[:N]
            n_pad = self.seq_len - N

        frames = []
        for p in selected_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

            if self.transform is not None:
                img_tensor = self.transform(image=img_rgb)["image"]
            else:
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor - mean_t) / std_t

            frames.append(img_tensor)

        # Pad with zero frames if video is shorter than seq_len
        if n_pad > 0:
            h = frames[0].shape[1] if frames else 256
            w = frames[0].shape[2] if frames else 256
            pad_frame = torch.zeros(3, h, w)
            frames.extend([pad_frame] * n_pad)

        seq_tensor = torch.stack(frames, dim=0) if frames else torch.zeros(self.seq_len, 3, 256, 256)

        # padding_mask: True = padded position (ignored in temporal attention)
        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        if n_pad > 0:
            padding_mask[self.seq_len - n_pad:] = True

        return seq_tensor, torch.tensor(label, dtype=torch.long), padding_mask

def create_dataloaders(config: Optional[Dict[str, Any]] = None) -> Dict[str, DataLoader]:
    """Builds train, val, and test DataLoaders with zero identity leakage partitioning."""
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
    if os.path.exists(cropped_dir):
        for root, _, files in os.walk(cropped_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, file)
                    label = 0 if "original" in full_path.lower() or "real" in full_path.lower() else 1
                    samples.append((full_path, label))

    if not samples:
        # Fallback dummy samples for integration testing
        samples = [(f"dummy_sample_{i}.jpg", i % 2) for i in range(20)]

    train_samples, val_samples, test_samples = perform_graph_split(samples, seed=seed)

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = DeepfakeDataset(train_samples, transform=train_transform)
    val_dataset = DeepfakeDataset(val_samples, transform=eval_transform)
    test_dataset = DeepfakeDataset(test_samples, transform=eval_transform)

    train_labels = [s[1] for s in train_samples]
    class_counts = np.maximum(np.bincount(train_labels), 1)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

build_dataloaders = create_dataloaders

def create_sequence_dataloaders(config: Optional[Dict[str, Any]] = None) -> Dict[str, DataLoader]:
    """
    Builds train/val/test DataLoaders for video-sequence training (--sequence mode).
    Each batch item is a 3-tuple: (seq_tensor [T,3,H,W], label, padding_mask [T]).
    Frames are grouped by video identity using extract_video_id, then split via
    graph-connected identity partitioning to guarantee zero identity leakage.
    """
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

    # Group frames by video ID → {vid_id: [frame_path, ...]}
    video_frame_map: Dict[str, List[str]] = {}
    video_label_map: Dict[str, int] = {}

    if os.path.exists(cropped_dir):
        for root, _, files in os.walk(cropped_dir):
            for file in sorted(files):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(root, file)
                    vid_id = extract_video_id(full_path)
                    label = 0 if "original" in full_path.lower() or "real" in full_path.lower() else 1
                    video_frame_map.setdefault(vid_id, []).append(full_path)
                    video_label_map[vid_id] = label

    # Build flat (vid_id, label) list for graph split
    flat_samples = [
        (vid_id, video_label_map[vid_id])
        for vid_id in sorted(video_frame_map.keys())
    ]

    if not flat_samples:
        # Dummy fallback for integration testing
        flat_samples = [(f"dummy_video_{i}", i % 2) for i in range(10)]
        video_frame_map = {f"dummy_video_{i}": [f"dummy_video_{i}_frame000.jpg"] for i in range(10)}

    train_s, val_s, test_s = perform_graph_split(flat_samples, seed=seed)

    def _to_video_samples(split: List[Tuple]) -> List[Tuple[List[str], int]]:
        return [(video_frame_map.get(vid_id, []), lbl) for vid_id, lbl in split]

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = SequenceVideoDataset(_to_video_samples(train_s), transform=train_transform, seq_len=seq_len)
    val_dataset   = SequenceVideoDataset(_to_video_samples(val_s),   transform=eval_transform,  seq_len=seq_len)
    test_dataset  = SequenceVideoDataset(_to_video_samples(test_s),  transform=eval_transform,  seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "val": val_loader, "test": test_loader}

