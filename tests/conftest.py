import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.config import load_config
from src.models.hybrid_detector import build_model

@pytest.fixture(scope="session")
def shared_config():
    return load_config()

@pytest.fixture(scope="module")
def shared_model(shared_config):
    model = build_model(use_fft=True, pretrained=False, config=shared_config)
    model.eval()
    yield model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="session")
def synthetic_dataloader():
    dummy_x = torch.randn(4, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def eval_model_factory():
    models = []
    def _factory(use_fft: bool = True, backbone_name: str = "convnext_tiny"):
        model = build_model(use_fft=use_fft, backbone_name=backbone_name, pretrained=False, device=torch.device("cpu"))
        model.eval()
        models.append(model)
        return model
    yield _factory
    for m in models:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def dummy_4d_batch():
    return torch.randn(2, 3, 256, 256, device=torch.device("cpu"))

@pytest.fixture
def dummy_5d_sequence():
    return torch.randn(1, 8, 3, 256, 256, device=torch.device("cpu"))

@pytest.fixture
def dummy_config():
    return {
        "model": {"backbone": "convnext_tiny", "pretrained": False},
        "training": {"batch_size": 2, "epochs": 1},
        "data": {"target_size": 256, "sequence_length": 8}
    }
