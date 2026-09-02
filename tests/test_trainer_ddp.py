"""Unit tests for distributed training components, EMA shadow weights, and losses."""

import torch
import torch.nn as nn
from src.evaluation.metrics import compute_roc_auc_safe
from src.training.ema import ExponentialMovingAverage
from src.training.loss import MaskedBCEWithLogits


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTrainingComponents:
    """Tests EMA update and context manager, loss masking, and safe metrics."""

    def test_ema_update_and_context(self) -> None:
        model = DummyModel()
        with torch.no_grad():
            model.fc.weight.fill_(1.0)
            model.fc.bias.fill_(0.0)

        ema = ExponentialMovingAverage(model, decay=0.9)
        assert torch.allclose(ema.shadow["fc.weight"], torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

        with torch.no_grad():
            model.fc.weight.fill_(2.0)
        ema.update(model)

        # Expected shadow: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert torch.allclose(ema.shadow["fc.weight"], torch.tensor([[1.1, 1.1, 1.1, 1.1]]))

        with ema.average_parameters(model):
            assert torch.allclose(model.fc.weight, torch.tensor([[1.1, 1.1, 1.1, 1.1]]))
        # Restored original
        assert torch.allclose(model.fc.weight, torch.tensor([[2.0, 2.0, 2.0, 2.0]]))

    def test_masked_loss_ignores_corrupt_samples(self) -> None:
        criterion = MaskedBCEWithLogits()
        logits = torch.tensor([[10.0], [-10.0]])
        targets = torch.tensor([[0.0], [1.0]])  # Highly penalized targets
        valid_flags = torch.tensor([[0.0], [0.0]])  # Both marked corrupt

        loss = criterion(logits, targets, valid_flags)
        assert loss.item() == 0.0  # Completely masked out

    def test_compute_roc_auc_safe_single_class(self) -> None:
        y_true = [1, 1, 1, 1]
        y_score = [0.8, 0.9, 0.7, 0.6]
        # Should gracefully return 0.5 without throwing ValueError
        auc = compute_roc_auc_safe(y_true, y_score, fallback=0.5)
        assert auc == 0.5
