import torch


def test_hybrid_detector_dual_stream_forward(eval_model_factory, dummy_4d_batch):
    model = eval_model_factory(use_fft=True)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size, 1]), f"Expected logits shape torch.Size([{b_size}, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size, 1]), f"Expected probs shape torch.Size([{b_size}, 1]), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"


def test_hybrid_detector_full_res_512_forward(eval_model_factory):
    """Verifies 512x512 FFT Frequency Extraction and Spatial Backbone pass."""
    model = eval_model_factory(use_fft=True)

    dummy_512 = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(dummy_512)
        probs = torch.sigmoid(logits)

    assert logits.shape == torch.Size([2, 1]), f"Expected logits shape torch.Size([2, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([2, 1]), f"Expected probs shape torch.Size([2, 1]), got {probs.shape}"


def test_hybrid_detector_spatial_only_forward(eval_model_factory, dummy_4d_batch):
    model = eval_model_factory(use_fft=False)

    with torch.no_grad():
        logits = model(dummy_4d_batch)
        probs = torch.sigmoid(logits)

    b_size = dummy_4d_batch.shape[0]
    assert logits.shape == torch.Size([b_size, 1]), f"Expected logits shape torch.Size([{b_size}, 1]), got {logits.shape}"
    assert probs.shape == torch.Size([b_size, 1]), f"Expected probs shape torch.Size([{b_size}, 1]), got {probs.shape}"


def test_use_fft_false_actually_disables_fft(eval_model_factory):
    """Regression test: config must not override explicit use_fft_branch=False."""
    model = eval_model_factory(use_fft=False)
    assert model.use_fft_branch is False, "use_fft_branch=False was overridden"
    assert not hasattr(model, "srm"), "srm module should not exist when use_fft_branch=False"

    first_linear = model.classifier[0]
    assert first_linear.in_features == 512, f"Classifier input dim {first_linear.in_features} != 512"


def test_corrupt_zero_input_nan_safety(eval_model_factory):
    """Verifies that all-zero corrupt image tensors return finite logits without NaN or Inf values."""
    model = eval_model_factory(use_fft=True)
    zero_batch = torch.zeros(2, 3, 256, 256)

    logits = model(zero_batch)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values under all-zero corrupt input"


def test_5d_sequence_forward(eval_model_factory):
    """Verifies 5D video sequence forward pass [B, T, 3, H, W] -> [B, 1]."""
    model = eval_model_factory(use_fft=True)
    seq_batch = torch.randn(2, 5, 3, 256, 256)

    with torch.no_grad():
        logits = model.forward_sequence(seq_batch)

    assert logits.shape == torch.Size([2, 1]), f"Expected sequence logits shape torch.Size([2, 1]), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Sequence logits contain NaN or Inf"
