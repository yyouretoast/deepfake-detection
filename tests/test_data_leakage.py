import pytest
from src.dataset.loader import perform_graph_split, extract_identities

def test_extract_identities_parsing():
    """Verifies that extract_identities correctly parses video IDs and actor identities."""
    id1, id2 = extract_identities("000_003.mp4")
    assert id1 == "000"
    assert id2 == "003"

    id1_real, id2_real = extract_identities("000.mp4")
    assert id1_real == "000"
    assert id2_real == "000"

def test_perform_graph_split_zero_identity_leakage():
    """Asserts 0 identity overlap between Train, Validation, and Test sets in LOTO graph partitioning."""
    samples = [
        ("000_001.mp4", 1),
        ("000_001_2.mp4", 1),
        ("000.mp4", 0),
        ("002_003.mp4", 1),
        ("002.mp4", 0),
        ("004_005.mp4", 1),
        ("004.mp4", 0),
        ("006_007.mp4", 1),
        ("006.mp4", 0),
        ("008_009.mp4", 1),
    ]

    train_split, val_split, test_split = perform_graph_split(
        samples, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    def get_identities(split):
        ids = set()
        for item in split:
            path = item[0]
            id1, id2 = extract_identities(path)
            ids.add(id1)
            ids.add(id2)
        return ids

    train_ids = get_identities(train_split)
    val_ids = get_identities(val_split)
    test_ids = get_identities(test_split)

    assert len(train_ids.intersection(val_ids)) == 0, f"Identity leakage between Train and Val: {train_ids.intersection(val_ids)}"
    assert len(train_ids.intersection(test_ids)) == 0, f"Identity leakage between Train and Test: {train_ids.intersection(test_ids)}"
    assert len(val_ids.intersection(test_ids)) == 0, f"Identity leakage between Val and Test: {val_ids.intersection(test_ids)}"

def test_perform_graph_split_nested_kaggle_paths():
    """Asserts graph partitioning handles complex nested Kaggle subfolder directory structures."""
    nested_samples = [
        ("/kaggle/input/datasets/ff-c23/manipulated_sequences/Deepfakes/c23/videos/000_003/frame_001.webp", 1),
        ("/kaggle/input/datasets/ff-c23/original_sequences/youtube/c23/videos/000/frame_001.webp", 0),
        ("/kaggle/input/datasets/ff-c23/manipulated_sequences/FaceSwap/c23/videos/004_007/frame_001.webp", 1),
        ("/kaggle/input/datasets/ff-c23/original_sequences/youtube/c23/videos/004/frame_001.webp", 0),
        ("/kaggle/input/datasets/ff-c23/manipulated_sequences/Face2Face/c23/videos/008_011/frame_001.webp", 1),
        ("/kaggle/input/datasets/ff-c23/original_sequences/youtube/c23/videos/008/frame_001.webp", 0),
    ]

    train_split, val_split, test_split = perform_graph_split(
        nested_samples, val_ratio=0.3, test_ratio=0.3, seed=42
    )

    def get_identities(split):
        ids = set()
        for item in split:
            path = item[0]
            id1, id2 = extract_identities(path)
            ids.add(id1)
            ids.add(id2)
        return ids

    train_ids = get_identities(train_split)
    val_ids = get_identities(val_split)
    test_ids = get_identities(test_split)

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
