import pytest
from scripts.train_loto_experiment import matches_holdout_domain, filter_loto_split_strict

def test_matches_holdout_domain_pair_ranges():
    """Verifies that matches_holdout_domain correctly identifies sub-generator pair ranges and boundary cases."""
    # NeuralTextures (Pairs 600-799)
    assert matches_holdout_domain("fake/600_605/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/799_800/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/600_605/frame_001.webp", "nt") is True
    
    # Boundary checks: 599 is FaceSwap, 800 is out-of-range
    assert matches_holdout_domain("fake/599_600/frame_001.webp", "neuraltextures") is False
    assert matches_holdout_domain("fake/800_805/frame_001.webp", "neuraltextures") is False

    # Deepfakes (Pairs 0-199)
    assert matches_holdout_domain("fake/000_003/frame_001.webp", "deepfakes") is True
    assert matches_holdout_domain("fake/199_200/frame_001.webp", "deepfakes") is True
    assert matches_holdout_domain("fake/200_205/frame_001.webp", "deepfakes") is False

    # Face2Face (Pairs 200-399)
    assert matches_holdout_domain("fake/205_210/frame_001.webp", "face2face") is True

    # FaceSwap (Pairs 400-599)
    assert matches_holdout_domain("fake/405_410/frame_001.webp", "faceswap") is True

    # Celeb-DF v2 synthesis
    assert matches_holdout_domain("fake/id0_id16_0000/frame_002.webp", "celeb") is True
    assert matches_holdout_domain("fake/01_02__meeting_serious/frame_002.webp", "celeb") is True

def test_filter_loto_split_strict_retains_reals():
    """Asserts that filter_loto_split_strict filters FAKE samples matching holdout while retaining 100% REAL samples."""
    samples = [
        ("fake/600_605/frame_001.webp", 1.0),
        ("fake/600_605/frame_002.webp", 1.0),
        ("real/000/frame_001.webp", 0.0),
        ("fake/000_003/frame_001.webp", 1.0),
        ("real/600/frame_001.webp", 0.0),  # Real face, should NOT be filtered out even if path contains 600
    ]

    retained, held_out_fakes = filter_loto_split_strict(samples, "neuraltextures")

    # 2 fake NeuralTextures samples filtered into held_out_fakes
    assert len(held_out_fakes) == 2
    assert all(s[1] == 1.0 for s in held_out_fakes)

    # Retained contains all 2 reals + 1 non-NeuralTextures fake
    assert len(retained) == 3
    retained_reals = [s for s in retained if s[1] == 0.0]
    assert len(retained_reals) == 2
