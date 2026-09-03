"""Unit tests for ResSE-Spectral Tower, Squeeze-and-Excitation attention, and auxiliary supervision."""

import torch
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.spectral_tower import ResSESpectralTower, SEBlock, SpectralResBlock


class TestSpectralTower:
    """Verifies ResSE-Spectral Tower forward pass, dimensions, autograd, and auxiliary head."""

    def test_se_block_attention_bounds(self) -> None:
        se = SEBlock(channels=64, reduction=8)
        x = torch.randn(4, 64, 32, 32)
        out = se(x)
        assert out.shape == (4, 64, 32, 32)
        # Verify gradient flow
        out.sum().backward()
        for p in se.parameters():
            assert p.grad is not None and not torch.isnan(p.grad).any()

    def test_spectral_res_block_stride_and_channels(self) -> None:
        block = SpectralResBlock(in_c=48, out_c=96, stride=2)
        x = torch.randn(2, 48, 64, 64)
        out = block(x)
        assert out.shape == (2, 96, 32, 32)

    def test_resse_spectral_tower_forward_and_aux(self) -> None:
        tower = ResSESpectralTower(in_channels=20, embed_dim=512)
        x = torch.randn(2, 20, 256, 256)
        feat, aux_logit = tower(x)
        assert feat.shape == (2, 512)
        assert aux_logit.shape == (2, 1)

        # Multi-task loss backprop check
        loss = feat.sum() + 0.3 * aux_logit.sum()
        loss.backward()
        for name, p in tower.named_parameters():
            assert p.grad is not None and not torch.isnan(p.grad).any(), f"Missing grad in {name}"

    def test_hybrid_detector_with_resse_backbone(self) -> None:
        model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
        assert hasattr(model, "freq_tower")
        x = torch.randn(2, 3, 256, 256)

        # Test standard forward
        out = model(x)
        assert out.shape == (2, 1)

        # Test forward with return_aux=True
        fused_logits, aux_logits = model(x, return_aux=True)
        assert fused_logits.shape == (2, 1)
        assert aux_logits.shape == (2, 1)

        # Test feature extraction
        feats = model.extract_features(x)
        assert feats.shape == (2, 512)
