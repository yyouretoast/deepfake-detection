import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_video_id(filename: str) -> str:
    """
    Extracts primary target video identifier from face filename.
    Format examples:
      'Deepfakes_123_456_f0.png' -> video_id '123' (primary target identity)
      'original_000_f2.png' -> video_id '000'
      '045_f12.png' -> video_id '045'
    Guarantees original reference videos and their deepfake pairs end up in the exact same split.
    """
    basename = os.path.splitext(filename)[0]
    # Remove frame suffix like _f0, _f12
    base_no_frame = re.sub(r'_f\d+$', '', basename)
    
    # Check for paired pattern like Deepfakes_123_456 or 123_456
    match_pair = re.search(r'(\d+)_\d+', base_no_frame)
    if match_pair:
        return match_pair.group(1)
        
    # Check for single video ID like 000 or original_000
    match_single = re.search(r'(\d+)', base_no_frame)
    if match_single:
        return match_single.group(1)
        
    return base_no_frame.split('_')[0]

def group_video_split(file_list, test_size=0.15, val_size=0.15, seed=42):
    """
    Group-based split guaranteeing zero primary video_id overlap between train, val, and test sets.
    """
    video_map = {}
    for filepath in file_list:
        vid = extract_video_id(os.path.basename(filepath))
        if vid not in video_map:
            video_map[vid] = []
        video_map[vid].append(filepath)

    unique_vids = list(video_map.keys())
    random.seed(seed)
    random.shuffle(unique_vids)

    num_vids = len(unique_vids)
    num_test = max(1, int(num_vids * test_size))
    num_val = max(1, int(num_vids * val_size))

    test_vids = set(unique_vids[:num_test])
    val_vids = set(unique_vids[num_test:num_test + num_val])
    train_vids = set(unique_vids[num_test + num_val:])

    train_files = [f for vid in train_vids for f in video_map[vid]]
    val_files = [f for vid in val_vids for f in video_map[vid]]
    test_files = [f for vid in test_vids for f in video_map[vid]]

    # Safety assertion: Zero video ID overlap across all splits
    assert len(train_vids.intersection(val_vids)) == 0, "Data leakage between Train and Val splits"
    assert len(train_vids.intersection(test_vids)) == 0, "Data leakage between Train and Test splits"
    assert len(val_vids.intersection(test_vids)) == 0, "Data leakage between Val and Test splits"

    return train_files, val_files, test_files

def get_transforms(img_size=224, is_train=True):
    """Albumentations pipeline with spatial & compression augmentations."""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
            A.ImageCompression(quality_range=(60, 100), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for face images with binary targets (0: Fake, 1: Real).
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        with Image.open(path) as img:
            image = np.array(img.convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        return image_tensor, torch.tensor(label, dtype=torch.float32)
