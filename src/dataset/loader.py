from typing import List, Tuple, Optional, Any
import os
import re
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def extract_video_id(filename: str) -> str:
    """Extracts primary target video identifier from face filename."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    base_no_frame = re.sub(r'_f\d+$', '', basename)
    
    match_pair = re.search(r'(\d+)_\d+', base_no_frame)
    if match_pair:
        return match_pair.group(1)
        
    match_single = re.search(r'(\d+)', base_no_frame)
    if match_single:
        return match_single.group(1)
        
    return base_no_frame

def group_video_split(
    file_list: Any,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple[Any, Any, Any]:
    """Stratified Group-based split guaranteeing zero primary video_id overlap and class-balanced splits."""
    video_map: dict = {}
    video_labels: dict = {}

    for item in file_list:
        if isinstance(item, (tuple, list)):
            filepath, label = item[0], item[1]
        else:
            filepath, label = item, (0 if "original" in item.lower() or "real" in item.lower() else 1)

        vid = extract_video_id(filepath)
        if vid not in video_map:
            video_map[vid] = []
            video_labels[vid] = label
        video_map[vid].append(item)

    real_vids = [v for v, l in video_labels.items() if l == 0]
    fake_vids = [v for v, l in video_labels.items() if l == 1]

    random.seed(seed)
    random.shuffle(real_vids)
    random.shuffle(fake_vids)

    def split_vids(v_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
        num_vids = len(v_list)
        n_test = max(1, int(num_vids * test_size)) if num_vids > 0 else 0
        n_val = max(1, int(num_vids * val_size)) if num_vids > 0 else 0
        return v_list[n_test + n_val:], v_list[n_test:n_test + n_val], v_list[:n_test]

    real_train, real_val, real_test = split_vids(real_vids)
    fake_train, fake_val, fake_test = split_vids(fake_vids)

    train_vids = set(real_train + fake_train)
    val_vids = set(real_val + fake_val)
    test_vids = set(real_test + fake_test)

    train_files = [item for vid in train_vids for item in video_map[vid]]
    val_files = [item for vid in val_vids for item in video_map[vid]]
    test_files = [item for vid in test_vids for item in video_map[vid]]

    return train_files, val_files, test_files

def get_transforms(img_size: int = 224, is_train: bool = True) -> Optional[Any]:
    """Albumentations pipeline with spatial & compression augmentations."""
    if A is None:
        return None
    if is_train:
        try:
            downscale_comp = A.Downscale(scale_range=(0.7, 0.9), p=0.3)
        except Exception:
            downscale_comp = A.Downscale(scale_min=0.7, scale_max=0.9, p=0.3)

        transforms = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
            downscale_comp,
        ]
        if hasattr(A, "JPEGCompression"):
            transforms.append(A.JPEGCompression(quality_lower=50, quality_upper=90, p=0.4))
        elif hasattr(A, "ImageCompression"):
            transforms.append(A.ImageCompression(quality_range=(50, 90), p=0.4))
        transforms.extend([
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
        return A.Compose(transforms)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

class DeepfakeDataset(Dataset):
    """
    High-performance PyTorch Dataset using C++ OpenCV decoding for 3x faster I/O.
    """
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        transform: Optional[Any] = None
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]
        label = self.labels[idx]

        # OpenCV image decoding for high-throughput I/O
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with Image.open(path) as img:
                image = np.array(img.convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        return image_tensor, torch.tensor(label, dtype=torch.float32)

def create_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory helper function returning initialized PyTorch DataLoaders."""
    train_paths, train_labels = zip(*train_samples) if train_samples else ([], [])
    val_paths, val_labels = zip(*val_samples) if val_samples else ([], [])
    test_paths, test_labels = zip(*test_samples) if test_samples else ([], [])

    train_ds = DeepfakeDataset(list(train_paths), list(train_labels), get_transforms(img_size, is_train=True))
    val_ds = DeepfakeDataset(list(val_paths), list(val_labels), get_transforms(img_size, is_train=False))
    test_ds = DeepfakeDataset(list(test_paths), list(test_labels), get_transforms(img_size, is_train=False))

    drop_last = len(train_ds) > batch_size
    if train_labels:
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
