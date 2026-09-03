"""Dataset loading, domain classification, and preprocessing public API."""

from src.dataset.datasets import (
    DeepfakeDataset,
    FaceCropDataset,
)
from src.dataset.domains import DomainClassifier, DomainInfo, ManipulationDomain
from src.dataset.loader import (
    SequenceVideoDataset,
    dedupe_split,
    extract_identities,
    get_transforms,
    group_video_sequences,
    perform_graph_split,
)
from src.dataset.preprocess import DynamicFaceCropper
from src.dataset.resolver import (
    DatasetResolver,
    find_dataset_root,
    find_weights_path,
    resolve_splits_path,
)

__all__ = [
    "DeepfakeDataset",
    "FaceCropDataset",
    "DomainClassifier",
    "DomainInfo",
    "ManipulationDomain",
    "DatasetResolver",
    "find_dataset_root",
    "resolve_splits_path",
    "find_weights_path",
    "extract_identities",
    "perform_graph_split",
    "dedupe_split",
    "get_transforms",
    "DynamicFaceCropper",
    "SequenceVideoDataset",
    "group_video_sequences",
]
