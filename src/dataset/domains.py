"""Canonical domain classification and identity taxonomy for FaceForensics++ and Celeb-DF."""

from enum import Enum
import re
from typing import NamedTuple, Optional


class ManipulationDomain(str, Enum):
    """Enumeration of deepfake manipulation generator domains."""

    REAL = "real"
    DEEPFAKES = "deepfakes"
    FACE2FACE = "face2face"
    FACESWAP = "faceswap"
    NEURALTEXTURES = "neuraltextures"
    CELEB_DF = "celeb"
    UNKNOWN = "unknown"


class DomainInfo(NamedTuple):
    """Domain metadata for a dataset sample."""

    domain: ManipulationDomain
    display_name: str
    is_fake: bool
    pair_number: Optional[int]


class DomainClassifier:
    """Canonical domain classifier and holdout matcher for FF++ and Celeb-DF."""

    PAIR_REGEX = re.compile(r"(?:^|[\\/])(\d{3})_\d{3}(?:[\\/\.]|$)")
    CELEB_REGEX = re.compile(r"(?:celeb|(?:^|[\\/_\-])id\d+|__)", re.IGNORECASE)

    # Boundary-aware generator method regexes matching official benchmark folder names
    DEEPFAKES_METHOD_REGEX = re.compile(r"(?:^|[\\/])(?:manipulated_sequences[\\/])?deepfakes(?:[\\/]|$)", re.IGNORECASE)
    FACE2FACE_METHOD_REGEX = re.compile(r"(?:^|[\\/])(?:manipulated_sequences[\\/])?face2face(?:[\\/]|$)", re.IGNORECASE)
    FACESWAP_METHOD_REGEX = re.compile(r"(?:^|[\\/])(?:manipulated_sequences[\\/])?faceswap(?:[\\/]|$)", re.IGNORECASE)
    NEURALTEXTURES_METHOD_REGEX = re.compile(r"(?:^|[\\/])(?:manipulated_sequences[\\/])?neuraltextures(?:[\\/]|$)", re.IGNORECASE)

    @classmethod
    def extract_pair_number(cls, path: str) -> Optional[int]:
        """Extract the 3-digit source actor pair number from a sample path if present."""
        norm_path = path.replace("\\", "/").lower()
        if norm_path.startswith("./"):
            norm_path = norm_path[2:]
        match = cls.PAIR_REGEX.search(norm_path)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def classify(cls, path: str) -> DomainInfo:
        """Classify a relative or absolute sample path into its manipulation domain and metadata."""
        norm_path = path.replace("\\", "/").lower()
        if norm_path.startswith("./"):
            norm_path = norm_path[2:]

        # Tier 1: Authentic / Real Faces
        is_explicit_real = (
            "/real/" in norm_path
            or norm_path.startswith("real/")
            or "original_sequences" in norm_path
            or "celeb-real" in norm_path
            or "youtube-real" in norm_path
        )
        is_explicit_fake = (
            "/fake/" in norm_path
            or norm_path.startswith("fake/")
            or "manipulated_sequences" in norm_path
            or "celeb-synthesis" in norm_path
        )
        if is_explicit_real and not is_explicit_fake:
            return DomainInfo(
                domain=ManipulationDomain.REAL,
                display_name="Original Real Faces",
                is_fake=False,
                pair_number=None,
            )

        # Tier 2: Celeb-DF v2 Synthesis
        if cls.CELEB_REGEX.search(norm_path):
            return DomainInfo(
                domain=ManipulationDomain.CELEB_DF,
                display_name="Celeb-DF v2 Synthesis",
                is_fake=True,
                pair_number=None,
            )

        pair_num = cls.extract_pair_number(norm_path)

        # Tier 3: Explicit FF++ Method Directories (Official FF++ Structure)
        if cls.FACE2FACE_METHOD_REGEX.search(norm_path) or "face2face" in norm_path:
            return DomainInfo(
                domain=ManipulationDomain.FACE2FACE,
                display_name="FF++ Face2Face",
                is_fake=True,
                pair_number=pair_num,
            )
        if cls.FACESWAP_METHOD_REGEX.search(norm_path) or "faceswap" in norm_path:
            return DomainInfo(
                domain=ManipulationDomain.FACESWAP,
                display_name="FF++ FaceSwap",
                is_fake=True,
                pair_number=pair_num,
            )
        if cls.NEURALTEXTURES_METHOD_REGEX.search(norm_path) or "neuraltextures" in norm_path:
            return DomainInfo(
                domain=ManipulationDomain.NEURALTEXTURES,
                display_name="FF++ NeuralTextures",
                is_fake=True,
                pair_number=pair_num,
            )
        if cls.DEEPFAKES_METHOD_REGEX.search(norm_path):
            return DomainInfo(
                domain=ManipulationDomain.DEEPFAKES,
                display_name="FF++ Deepfakes",
                is_fake=True,
                pair_number=pair_num,
            )

        # Tier 4: Pair-Range Fallback (For Flattened Extracted Crops Without Method Names)
        if pair_num is not None:
            if 0 <= pair_num <= 99:
                return DomainInfo(
                    domain=ManipulationDomain.DEEPFAKES,
                    display_name="FF++ Deepfakes (Pairs 0-99)",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 100 <= pair_num <= 399:
                return DomainInfo(
                    domain=ManipulationDomain.FACE2FACE,
                    display_name="FF++ Face2Face",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 400 <= pair_num <= 599:
                return DomainInfo(
                    domain=ManipulationDomain.FACESWAP,
                    display_name="FF++ FaceSwap",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 600 <= pair_num <= 799:
                return DomainInfo(
                    domain=ManipulationDomain.NEURALTEXTURES,
                    display_name="FF++ NeuralTextures",
                    is_fake=True,
                    pair_number=pair_num,
                )
            return DomainInfo(
                domain=ManipulationDomain.UNKNOWN,
                display_name=f"FF++ Manipulation (Pair {pair_num:03d})",
                is_fake=True,
                pair_number=pair_num,
            )

        # Tier 5: Generic Fallback
        return DomainInfo(ManipulationDomain.UNKNOWN, "FF++ Deepfakes / Mixed", True, None)

    @classmethod
    def matches_holdout(cls, path: str, holdout_keyword: str) -> bool:
        """
        Check if sample path belongs to the specified LOTO holdout generator domain.
        Strictly delegates to classify() as the single canonical source of truth.
        """
        norm_path = path.replace("\\", "/").lower()
        kw = holdout_keyword.lower().strip()

        info = cls.classify(norm_path)
        if not info.is_fake:
            return False

        keyword_to_domain = {
            "deepfakes": ManipulationDomain.DEEPFAKES,
            "df": ManipulationDomain.DEEPFAKES,
            "face2face": ManipulationDomain.FACE2FACE,
            "f2f": ManipulationDomain.FACE2FACE,
            "faceswap": ManipulationDomain.FACESWAP,
            "fs": ManipulationDomain.FACESWAP,
            "neuraltextures": ManipulationDomain.NEURALTEXTURES,
            "nt": ManipulationDomain.NEURALTEXTURES,
            "celeb": ManipulationDomain.CELEB_DF,
            "celeb-df": ManipulationDomain.CELEB_DF,
            "celeb_df": ManipulationDomain.CELEB_DF,
            "celebdf": ManipulationDomain.CELEB_DF,
        }

        if kw in keyword_to_domain:
            return info.domain == keyword_to_domain[kw]

        # Generic keyword fallback for synthetic / custom domains (e.g. sora, midjourney)
        return kw in norm_path
