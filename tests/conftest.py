import random
from collections.abc import Callable

import numpy as np
import pytest
import torch

from src.models.hybrid_detector import HybridDeepfakeDetector


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    """Ensure deterministic RNG state across all tests."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture(scope="session")
def _eval_model_cache() -> dict[bool, HybridDeepfakeDetector]:
    """Lazily cache uninitialized eval models across the session to avoid repeated allocations."""
    return {}


@pytest.fixture
def eval_model_factory(
    _eval_model_cache: dict[bool, HybridDeepfakeDetector],
) -> Callable[[bool], HybridDeepfakeDetector]:
    def _factory(use_fft: bool = True) -> HybridDeepfakeDetector:
        if use_fft not in _eval_model_cache:
            model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=use_fft)
            model.eval()
            _eval_model_cache[use_fft] = model
        model = _eval_model_cache[use_fft]
        model.eval()
        model.zero_grad(set_to_none=True)
        return model

    return _factory


@pytest.fixture
def dummy_4d_batch() -> torch.Tensor:
    return torch.randn(2, 3, 256, 256)
