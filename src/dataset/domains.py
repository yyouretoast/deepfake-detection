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

    PAIR_REGEX = re.compile(r"(?:^|/)(\d{3})_\d{3}(?:/|$)")
    CELEB_INDICATORS = ("id", "__", "celeb")

    @classmethod
    def extract_pair_number(cls, path: str) -> Optional[int]:
        """Extract the 3-digit source actor pair number from a sample path if present."""
        norm_path = path.replace("\\", "/").lower()
        match = cls.PAIR_REGEX.search(norm_path)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def classify(cls, path: str) -> DomainInfo:
        """Classify a relative or absolute sample path into its manipulation domain and metadata."""
        norm_path = path.replace("\\", "/").lower()

        if "real" in norm_path and "fake" not in norm_path:
            return DomainInfo(
                domain=ManipulationDomain.REAL,
                display_name="Original Real Faces",
                is_fake=False,
                pair_number=None,
            )

        if any(ind in norm_path for ind in cls.CELEB_INDICATORS):
            return DomainInfo(
                domain=ManipulationDomain.CELEB_DF,
                display_name="Celeb-DF v2 Synthesis",
                is_fake=True,
                pair_number=None,
            )

        pair_num = cls.extract_pair_number(norm_path)
        if pair_num is not None:
            if 0 <= pair_num <= 99:
                return DomainInfo(
                    domain=ManipulationDomain.DEEPFAKES,
                    display_name="FF++ Deepfakes (Pairs 0-99)",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 100 <= pair_num <= 199 or 200 <= pair_num <= 399:
                return DomainInfo(
                    domain=ManipulationDomain.FACE2FACE,
                    display_name="FF++ Face2Face",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 200 <= pair_num <= 299 or 400 <= pair_num <= 599:
                return DomainInfo(
                    domain=ManipulationDomain.FACESWAP,
                    display_name="FF++ FaceSwap",
                    is_fake=True,
                    pair_number=pair_num,
                )
            elif 300 <= pair_num <= 399 or 600 <= pair_num <= 799:
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

        if "deepfake" in norm_path or "df" in norm_path:
            return DomainInfo(ManipulationDomain.DEEPFAKES, "FF++ Deepfakes", True, None)
        if "face2face" in norm_path or "f2f" in norm_path:
            return DomainInfo(ManipulationDomain.FACE2FACE, "FF++ Face2Face", True, None)
        if "faceswap" in norm_path or "fs" in norm_path:
            return DomainInfo(ManipulationDomain.FACESWAP, "FF++ FaceSwap", True, None)
        if "neuraltextures" in norm_path or "nt" in norm_path:
            return DomainInfo(ManipulationDomain.NEURALTEXTURES, "FF++ NeuralTextures", True, None)

        return DomainInfo(ManipulationDomain.UNKNOWN, "FF++ Deepfakes / Mixed", True, None)

    @classmethod
    def matches_holdout(cls, path: str, holdout_keyword: str) -> bool:
        """
        Check if sample path belongs to the specified LOTO holdout generator domain.
        Strictly path-invariant across flat, nested, and Windows/Linux paths.
        """
        norm_path = path.replace("\\", "/").lower()
        kw = holdout_keyword.lower().strip()

        sub_path = norm_path
        if "/fake/" in sub_path:
            sub_path = sub_path.split("/fake/", 1)[1]
        elif sub_path.startswith("fake/"):
            sub_path = sub_path[5:]
        elif "/real/" in sub_path:
            sub_path = sub_path.split("/real/", 1)[1]
        elif sub_path.startswith("real/"):
            sub_path = sub_path[5:]

        if "real" in sub_path and "fake" not in sub_path:
            return False

        if kw in ("celeb", "celeb-df", "celeb_df", "celebdf"):
            return "id" in sub_path or "__" in sub_path or "celeb" in sub_path

        pair_num = cls.extract_pair_number(sub_path)
        if pair_num is not None:
            if kw in ("deepfakes", "df"):
                return 0 <= pair_num <= 99
            elif kw in ("face2face", "f2f"):
                return 100 <= pair_num <= 199 or 200 <= pair_num <= 399
            elif kw in ("faceswap", "fs"):
                return 200 <= pair_num <= 299 or 400 <= pair_num <= 599
            elif kw in ("neuraltextures", "nt"):
                return 300 <= pair_num <= 399 or 600 <= pair_num <= 799

        return kw in sub_path
