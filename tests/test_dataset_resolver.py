"""Unit tests for DatasetResolver priority order and path discovery."""

import json
import os
from src.dataset.resolver import DatasetResolver


class TestDatasetResolver:
    """Tests resolution priority order for dataset roots, manifests, and weights."""

    def test_custom_splits_priority(self, tmp_path) -> None:
        custom_splits = tmp_path / "custom_splits.json"
        with open(custom_splits, "w") as f:
            json.dump({"train": [], "val": [], "test": []}, f)

        resolved = DatasetResolver.resolve_splits_path(custom_splits=str(custom_splits))
        assert os.path.abspath(resolved) == os.path.abspath(str(custom_splits))

    def test_valid_dataset_root_detection(self, tmp_path) -> None:
        fake_dir = tmp_path / "fake"
        fake_dir.mkdir()
        splits = tmp_path / "splits.json"
        with open(splits, "w") as f:
            json.dump({"train": []}, f)

        assert DatasetResolver.is_valid_dataset_root(str(tmp_path)) is True

    def test_invalid_dataset_root_rejection(self, tmp_path) -> None:
        # Missing splits.json
        assert DatasetResolver.is_valid_dataset_root(str(tmp_path)) is False
