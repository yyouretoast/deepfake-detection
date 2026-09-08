"""Unit tests for Bi-GRU spatiotemporal consistency head and temporal attention."""

import torch
from src.models.temporal_head import BiGRUTemporalDetector


class TestTemporalHead:
    """Verifies Bi-GRU sequence modeling, shape preservation, and temporal attention normalization."""

    def test_temporal_detector_forward_and_attention(self) -> None:
        # Test with use_deltas=True (default)
        model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True)
        seq = torch.randn(4, 8, 512)  # [B=4, T=8 frames, embed=512]
        video_logit, attn_weights = model(seq)

        assert video_logit.shape == (4, 1)
        assert attn_weights.shape == (4, 8)
        assert (attn_weights >= 0.0).all()
        assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4), atol=1e-5)

        # Test with use_deltas=False (backward compatibility)
        model_no_deltas = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=False)
        video_logit_nd, attn_weights_nd = model_no_deltas(seq)
        assert video_logit_nd.shape == (4, 1)
        assert attn_weights_nd.shape == (4, 8)

    def test_temporal_detector_autograd_flow(self) -> None:
        model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=64, use_deltas=True)
        seq = torch.randn(2, 6, 512, requires_grad=True)
        video_logit, _ = model(seq)

        loss = video_logit.sum()
        loss.backward()

        assert seq.grad is not None and not torch.isnan(seq.grad).any()
        for name, p in model.named_parameters():
            assert p.grad is not None and not torch.isnan(p.grad).any(), f"Missing grad in {name}"

    def test_sequence_video_dataset_striding(self) -> None:
        from src.dataset.loader import SequenceVideoDataset
        # Dummy video with 20 frames
        dummy_paths = [f"frame_{i:03d}.png" for i in range(20)]
        samples = [(dummy_paths, 1)]

        # Eval dataset with seq_len=8, stride=2 (requires span = (8-1)*2 + 1 = 15 frames)
        ds_eval = SequenceVideoDataset(samples, seq_len=8, stride=2, is_train=False)
        assert ds_eval.stride == 2
        # Center start index should be (20 - 15) // 2 = 2
        # Indices: [2, 4, 6, 8, 10, 12, 14, 16]

        # Train dataset with seq_len=8, stride=2
        ds_train = SequenceVideoDataset(samples, seq_len=8, stride=2, is_train=True)
        assert ds_train.is_train is True

    def test_temporal_detector_dual_path_max_pooling(self) -> None:
        """Verifies dual-path (Attention + Max) pooling forward pass, shapes, and gradients."""
        model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True, use_max_pool=True)
        seq = torch.randn(3, 8, 512, requires_grad=True)
        video_logit, attn_weights = model(seq)

        assert video_logit.shape == (3, 1)
        assert attn_weights.shape == (3, 8)
        assert model.classifier[0].in_features == 128 * 4  # 512 for hidden_dim=128

        loss = video_logit.sum()
        loss.backward()
        assert seq.grad is not None and not torch.isnan(seq.grad).any()

    def test_state_dict_backward_compatibility(self) -> None:
        """Verifies seamless loading across legacy single-path and upgraded dual-path checkpoints."""
        # 1. Single-path checkpoint (classifier in_features = 256)
        legacy_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True, use_max_pool=False)
        legacy_sd = legacy_model.state_dict()

        # Loading legacy into modern default (use_max_pool=True) should auto-adapt
        new_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True, use_max_pool=True)
        new_model.load_state_dict(legacy_sd)
        assert new_model.use_max_pool is False
        assert new_model.classifier[0].in_features == 128 * 2

        # 2. Dual-path checkpoint into legacy-configured model
        dual_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True, use_max_pool=True)
        dual_sd = dual_model.state_dict()

        adapting_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=128, use_deltas=True, use_max_pool=False)
        adapting_model.load_state_dict(dual_sd)
        assert adapting_model.use_max_pool is True
        assert adapting_model.classifier[0].in_features == 128 * 4
