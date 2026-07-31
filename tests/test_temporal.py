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

def test_temporal_sequence_encoder_interpolation():
    encoder = TemporalSequenceEncoder(embed_dim=1152, max_len=32, num_heads=8, num_layers=2)
    dummy_seq_long = torch.randn(2, 40, 1152, requires_grad=True)  # [B=2, T=40 > 32, D=1152]
    
    out = encoder(dummy_seq_long)
    assert out.shape == (2, 1152)
    
    loss = out.sum()
    loss.backward()
    assert dummy_seq_long.grad is not None
    assert not torch.isnan(dummy_seq_long.grad).any()

def test_hybrid_detector_forward_sequence_parity(eval_model_factory, dummy_4d_batch, dummy_5d_sequence):
    model = eval_model_factory(use_fft=True)

    # Single-frame input
    out_single = model(dummy_4d_batch)
    assert out_single.shape == (dummy_4d_batch.shape[0],)

    # Video sequence input
    out_seq = model.forward_sequence(dummy_5d_sequence)
    assert out_seq.shape == (dummy_5d_sequence.shape[0],)

def test_hybrid_detector_forward_sequence_chunking(eval_model_factory):
    """Verifies that forward_sequence handles long sequences (T=18) via mini-batches of size 8."""
    model = eval_model_factory(use_fft=True)

    # Video sequence with T=18 (spans 3 chunks: 8 + 8 + 2)
    video_seq = torch.randn(1, 18, 3, 256, 256)
    with torch.inference_mode():
        out_seq = model.forward_sequence(video_seq)
    assert out_seq.shape == (1,)

def test_hybrid_detector_forward_sequence_padding_mask(eval_model_factory, dummy_5d_sequence):
    model = eval_model_factory(use_fft=True)

    B, T, C, H, W = dummy_5d_sequence.shape
    padding_mask = torch.zeros((B, T), dtype=torch.bool)
    if T > 2:
        padding_mask[:, -1] = True  # Mask out the last frame

    with torch.no_grad():
        out_padded = model.forward_sequence(dummy_5d_sequence, padding_mask=padding_mask)
    assert out_padded.shape == (B,)


