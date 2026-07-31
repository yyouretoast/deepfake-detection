import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluator import (
    calculate_crash_proof_eer,
    calculate_adaptive_ece,
    aggregate_video_predictions,
    tune_temperature_scaling,
    evaluate_full_suite,
    EvaluationResult
)

def test_calculate_crash_proof_eer():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    eer, thresh = calculate_crash_proof_eer(y_true, y_prob)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= thresh <= 1.0

def test_calculate_adaptive_ece():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = calculate_adaptive_ece(y_true, y_prob, n_bins=2)
    assert 0.0 <= ece <= 1.0

def test_aggregate_video_predictions():
    samples = [
        ("video1_f01.jpg", 0),
        ("video1_f02.jpg", 0),
        ("video2_f01.jpg", 1),
        ("video2_f02.jpg", 1),
    ]
    probs = np.array([0.2, 0.4, 0.8, 0.9])
    y_true_vid, y_score_vid = aggregate_video_predictions(samples, probs)
    assert len(y_true_vid) == 2
    assert np.isclose(y_score_vid[0], 0.3)
    assert np.isclose(y_score_vid[1], 0.85)

def test_tune_temperature_scaling():
    logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    T = tune_temperature_scaling(logits, labels)
    assert T > 0.0

def test_evaluate_full_suite():
    import torch.nn as nn
    class DummyModel(nn.Module):
        def forward(self, x, padding_mask=None):
            return torch.zeros(x.size(0))
            
    model = DummyModel()
    dummy_x = torch.randn(4, 3, 256, 256)
    dummy_y = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(dummy_x, dummy_y)
    dataset.samples = [("dummy.jpg", int(y)) for y in dummy_y]
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device('cpu')
    
    result = evaluate_full_suite(model, loader, device, is_sequence=False)
    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.frame_auc <= 1.0
