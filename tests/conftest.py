import pytest
import torch
from src.models.hybrid_detector import HybridDeepfakeDetector

@pytest.fixture
def eval_model_factory():
    def _factory(use_fft=True):
        model = HybridDeepfakeDetector(pretrained=False, use_fft_branch=use_fft)
        model.eval()
        return model
    return _factory

@pytest.fixture
def dummy_4d_batch():
    return torch.randn(2, 3, 256, 256)
