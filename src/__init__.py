from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector

try:
    from src.dataset.loader import (
        DeepfakeDataset,
        group_samples_by_video,
        extract_video_id,
        extract_identities,
        perform_graph_split,
        create_dataloaders,
    )
    from src.dataset.preprocess import DynamicFaceCropper

    # Backward-compatible alias so any existing call-sites using the old name still work
    group_video_split = group_samples_by_video  # noqa: E501
except ImportError as exc:
    raise ImportError(
        f"Failed to import core dataset modules: {exc}. "
        "Ensure all dependencies are installed (`pip install -r requirements.txt`)."
    ) from exc

__all__ = [
    "load_config",
    "HybridDeepfakeDetector",
    "DeepfakeDataset",
    "group_samples_by_video",
    "group_video_split",  # backward-compat alias
    "extract_video_id",
    "extract_identities",
    "perform_graph_split",
    "create_dataloaders",
    "DynamicFaceCropper",
]

