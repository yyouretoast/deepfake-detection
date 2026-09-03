"""Unit tests for Bi-GRU spatiotemporal consistency head and temporal attention."""

import torch
from src.models.temporal_head import BiGRUTemporalDetector


class TestTemporalHead:
    """Verifies Bi-GRU sequence modeling, shape preservation, and temporal attention normalization."""

    def test_temporal_detector_forward_and_attention(self) -> None:
        model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128)
        seq = torch.randn(4, 8, 512)  # [B=4, T=8 frames, embed=512]
        video_logit, attn_weights = model(seq)

        assert video_logit.shape == (4, 1)
        assert attn_weights.shape == (4, 8)

        # Verify temporal attention weights are non-negative and sum to 1.0 per video
        assert (attn_weights >= 0.0).all()
        assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_temporal_detector_autograd_flow(self) -> None:
        model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=64)
        seq = torch.randn(2, 6, 512, requires_grad=True)
        video_logit, _ = model(seq)

        loss = video_logit.sum()
        loss.backward()

        assert seq.grad is not None and not torch.isnan(seq.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None and not torch.isnan(p.grad).any(), f"Missing grad in {name}"
