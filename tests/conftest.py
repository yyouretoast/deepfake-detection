import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.config import load_config
from src.models.hybrid_detector import build_model

@pytest.fixture(scope="session")
def shared_config():
    return load_config()

@pytest.fixture(scope="session")
def shared_model(shared_config):
    model = build_model(use_fft=True, pretrained=False, config=shared_config)
    model.eval()
    return model

@pytest.fixture(scope="session")
def synthetic_dataloader():
    dummy_x = torch.randn(4, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    return DataLoader(dataset, batch_size=2)
