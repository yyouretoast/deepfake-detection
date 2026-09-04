"""Unit tests for canonical DomainClassifier and manipulation taxonomy."""

import pytest
from src.dataset.domains import DomainClassifier, ManipulationDomain


class TestDomainClassifier:
    """Tests pair regex extraction, manipulation domain categorization, and LOTO holdout matching."""

    @pytest.mark.parametrize(
        ("path", "expected_domain", "expected_pair"),
        [
            ("data/real/001/frame_01.png", ManipulationDomain.REAL, None),
            ("data/fake/050_060/frame_01.png", ManipulationDomain.DEEPFAKES, 50),
            ("data/fake/199_200/frame_01.png", ManipulationDomain.FACE2FACE, 199),
            ("data/fake/200_201/frame_01.png", ManipulationDomain.FACE2FACE, 200),
            ("data/fake/399_400/frame_01.png", ManipulationDomain.FACE2FACE, 399),
            ("data/fake/400_401/frame_01.png", ManipulationDomain.FACESWAP, 400),
            ("data/fake/599_600/frame_01.png", ManipulationDomain.FACESWAP, 599),
            ("data/fake/600_601/frame_01.png", ManipulationDomain.NEURALTEXTURES, 600),
            ("data/fake/799_800/frame_01.png", ManipulationDomain.NEURALTEXTURES, 799),
            ("data/fake/id0_id1/frame_01.png", ManipulationDomain.CELEB_DF, None),
            ("data/fake/id16_0000.png", ManipulationDomain.CELEB_DF, None),
            ("data/fake/celeb_fake_01.png", ManipulationDomain.CELEB_DF, None),
            ("data/valid/050_060/frame_01.png", ManipulationDomain.DEEPFAKES, 50),
            ("dataset/video_id/050_060/frame_01.png", ManipulationDomain.DEEPFAKES, 50),
            ("grid/fake/050_060/frame_01.png", ManipulationDomain.DEEPFAKES, 50),
            # Raw FaceForensics++ benchmark paths
            ("original_sequences/youtube/c23/videos/001.mp4", ManipulationDomain.REAL, None),
            ("original_sequences/actors/c23/videos/050.mp4", ManipulationDomain.REAL, None),
            ("./real/001/frame_01.png", ManipulationDomain.REAL, None),
            ("manipulated_sequences/Face2Face/c23/videos/000_003.mp4", ManipulationDomain.FACE2FACE, 0),
            ("manipulated_sequences/Face2Face/c23/frames/000_003/f01.png", ManipulationDomain.FACE2FACE, 0),
            ("manipulated_sequences/FaceSwap/c23/videos/050_060.mp4", ManipulationDomain.FACESWAP, 50),
            ("manipulated_sequences/NeuralTextures/c23/videos/050_060.mp4", ManipulationDomain.NEURALTEXTURES, 50),
            ("manipulated_sequences/Deepfakes/c23/videos/250_260.mp4", ManipulationDomain.DEEPFAKES, 250),
            # Official Celeb-DF v2 benchmark paths
            ("Celeb-real/id0_0000.mp4", ManipulationDomain.REAL, None),
            ("YouTube-real/00001.mp4", ManipulationDomain.REAL, None),
            ("Celeb-synthesis/id0_id16_0000.mp4", ManipulationDomain.CELEB_DF, None),
            # Extracted crops from deepfake_crops_512
            ("deepfake_crops_512/fake/850_860/f0.png", ManipulationDomain.UNKNOWN, 850),
            ("deepfake_crops_512/fake/01_02__meeting_serious__YVGY8LOK/f0.webp", ManipulationDomain.CELEB_DF, None),
        ],
    )
    def test_domain_classification(self, path: str, expected_domain: ManipulationDomain, expected_pair: int | None) -> None:
        info = DomainClassifier.classify(path)
        assert info.domain == expected_domain
        assert info.pair_number == expected_pair

    @pytest.mark.parametrize(
        ("path", "keyword", "expected_match"),
        [
            ("deepfake_crops_512/fake/050_060/f0.png", "deepfakes", True),
            ("deepfake_crops_512/fake/250_260/f0.png", "face2face", True),
            ("deepfake_crops_512/fake/450_460/f0.png", "faceswap", True),
            ("deepfake_crops_512/fake/650_660/f0.png", "neuraltextures", True),
            ("deepfake_crops_512/fake/id0_0000.png", "celeb", True),
            ("deepfake_crops_512/real/050/f0.png", "deepfakes", False),
            ("deepfake_crops_512\\fake\\450_460\\f0.png", "faceswap", True),  # Windows backslashes
        ],
    )
    def test_loto_holdout_matching(self, path: str, keyword: str, expected_match: bool) -> None:
        assert DomainClassifier.matches_holdout(path, keyword) is expected_match
