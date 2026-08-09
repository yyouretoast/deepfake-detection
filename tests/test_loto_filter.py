"""Unit tests for Leave-One-Type-Out (LOTO) cross-generator filtering."""

from scripts.train_loto_experiment import filter_loto_split_strict, matches_holdout_domain


def test_matches_holdout_domain_pair_ranges() -> None:
    assert matches_holdout_domain("fake/600_605/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/799_800/frame_001.webp", "neuraltextures") is True
    assert matches_holdout_domain("fake/600_605/frame_001.webp", "nt") is True

    assert matches_holdout_domain("fake/599_600/frame_001.webp", "neuraltextures") is False
    assert matches_holdout_domain("fake/800_805/frame_001.webp", "neuraltextures") is False

    assert matches_holdout_domain("fake/000_003/frame_001.webp", "deepfakes") is True
    assert matches_holdout_domain("fake/199_200/frame_001.webp", "deepfakes") is True
    assert matches_holdout_domain("fake/200_205/frame_001.webp", "deepfakes") is False

    assert matches_holdout_domain("fake/205_210/frame_001.webp", "face2face") is True
    assert matches_holdout_domain("fake/405_410/frame_001.webp", "faceswap") is True

    assert matches_holdout_domain("fake/id0_id16_0000/frame_002.webp", "celeb") is True
    assert matches_holdout_domain("fake/01_02__meeting_serious/frame_002.webp", "celeb") is True


def test_filter_loto_split_strict_retains_reals() -> None:
    samples = [
        ("fake/600_605/frame_001.webp", 1.0),
        ("fake/600_605/frame_002.webp", 1.0),
        ("real/000/frame_001.webp", 0.0),
        ("fake/000_003/frame_001.webp", 1.0),
        ("real/600/frame_001.webp", 0.0),
    ]

    retained, held_out_fakes = filter_loto_split_strict(samples, "neuraltextures")

    assert len(held_out_fakes) == 2
    assert all(s[1] == 1.0 for s in held_out_fakes)

    assert len(retained) == 3
    retained_reals = [s for s in retained if s[1] == 0.0]
    assert len(retained_reals) == 2


def test_matches_holdout_domain_generic_fallback() -> None:
    assert matches_holdout_domain("fake/sora_clip_001/frame_001.webp", "sora") is True
    assert matches_holdout_domain("fake/midjourney_002/frame_001.webp", "midjourney") is True
    assert matches_holdout_domain("fake/diff_001/frame_001.webp", "flux") is False
