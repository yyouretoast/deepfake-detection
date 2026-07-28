import torch
from src.models.temporal import TemporalSequenceEncoder
from src.models.hybrid_detector import build_model

def test_temporal_sequence_encoder_shape_and_grad():
    encoder = TemporalSequenceEncoder(embed_dim=1152, max_len=32, num_heads=8, num_layers=2)
    dummy_seq = torch.randn(2, 8, 1152, requires_grad=True)  # [B=2, T=8, D=1152]
    
    out = encoder(dummy_seq)
    assert out.shape == (2, 1152)

    loss = out.sum()
    loss.backward()
    assert dummy_seq.grad is not None
    assert not torch.isnan(dummy_seq.grad).any()

def test_hybrid_detector_forward_sequence_parity():
    model = build_model(use_fft=True, pretrained=False)
    model.eval()

    # Single-frame input [B=2, C=3, H=256, W=256]
    single_frame = torch.randn(2, 3, 256, 256)
    out_single = model(single_frame)
    assert out_single.shape == (2,)

    # Video sequence input [B=2, T=4, C=3, H=256, W=256]
    video_seq = torch.randn(2, 4, 3, 256, 256)
    out_seq = model.forward_sequence(video_seq)
    assert out_seq.shape == (2,)

def test_hybrid_detector_forward_sequence_chunking():
    """Verifies that forward_sequence handles long sequences (T=18) via mini-batches of size 8."""
    model = build_model(use_fft=True, pretrained=False)
    model.eval()

    # Video sequence with T=18 (spans 3 chunks: 8 + 8 + 2)
    video_seq = torch.randn(1, 18, 3, 256, 256)
    with torch.inference_mode():
        out_seq = model.forward_sequence(video_seq)
    assert out_seq.shape == (1,)

def test_hybrid_detector_forward_sequence_padding_mask():
    model = build_model(use_fft=True, pretrained=False)
    model.eval()

    video_seq = torch.randn(2, 4, 3, 256, 256)
    padding_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    with torch.no_grad():
        out_padded = model.forward_sequence(video_seq, padding_mask=padding_mask)
    assert out_padded.shape == (2,)


