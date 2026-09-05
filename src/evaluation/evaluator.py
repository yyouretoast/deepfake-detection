"""Inference and evaluation engine with AMP autocasting and optional test-time augmentation (TTA)."""

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates deepfake detectors on PyTorch datasets with mixed-precision and TTA support."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_tta: bool = False,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.use_tta = use_tta
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs batched inference across DataLoader returning flat (logits, targets, valid_flags)."""
        all_logits: list[float] = []
        all_targets: list[float] = []
        all_valid: list[float] = []

        device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if len(batch) == 3:
                images, labels, valid_flags = batch
            else:
                images, labels = batch
                valid_flags = torch.ones_like(labels, dtype=torch.float32)

            images = images.to(self.device)
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = self.model(images)
                if self.use_tta:
                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = self.model(images_flipped)
                    logits = (logits + logits_flipped) * 0.5

            all_logits.extend(logits.detach().cpu().reshape(-1).tolist())
            all_targets.extend(labels.detach().cpu().reshape(-1).tolist())
            all_valid.extend(valid_flags.detach().cpu().reshape(-1).tolist())

        return (
            np.array(all_logits, dtype=np.float32).flatten(),
            np.array(all_targets, dtype=np.float32).flatten(),
            np.array(all_valid, dtype=np.float32).flatten(),
        )

    def evaluate_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience runner wrapping a Dataset in a DataLoader for evaluation."""
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        return self.predict_loader(loader)
