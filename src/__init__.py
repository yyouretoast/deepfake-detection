from src.config import load_config
from src.models.hybrid_detector import HybridDeepfakeDetector

try:
    from src.dataset.datasets import DeepfakeDataset
    from src.dataset.loader import (
        extract_identities,
        perform_graph_split,
    )
    from src.dataset.preprocess import DynamicFaceCropper
except ImportError as exc:
    raise ImportError(
        f"Failed to import core dataset modules: {exc}. "
        "Ensure all dependencies are installed (`pip install -r requirements.txt`)."
    ) from exc

__all__ = [
    "load_config",
    "HybridDeepfakeDetector",
    "DeepfakeDataset",
    "extract_identities",
    "perform_graph_split",
    "DynamicFaceCropper",
]

