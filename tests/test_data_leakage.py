import pytest
from src.dataset.loader import extract_video_id, group_video_split

def test_extract_video_id():
    assert extract_video_id("original_000_f0.png") == "000"
    assert extract_video_id("Deepfakes_001_002_f3.png") == "001"
    assert extract_video_id("045_f12.png") == "045"

def test_group_video_split_zero_leakage():
    # Generate synthetic video filenames across 10 distinct video IDs
    sample_files = []
    for vid in range(10):
        for frame_idx in range(5):
            sample_files.append(f"Deepfakes_{vid:03d}_000_f{frame_idx}.png")

    train_files, val_files, test_files = group_video_split(sample_files, test_size=0.2, val_size=0.2, seed=42)

    train_vids = set(extract_video_id(f) for f in train_files)
    val_vids = set(extract_video_id(f) for f in val_files)
    test_vids = set(extract_video_id(f) for f in test_files)

    # Strict Zero Data Leakage Assertions
    assert len(train_vids.intersection(val_vids)) == 0, "Data leakage detected between Train and Val splits!"
    assert len(train_vids.intersection(test_vids)) == 0, "Data leakage detected between Train and Test splits!"
    assert len(val_vids.intersection(test_vids)) == 0, "Data leakage detected between Val and Test splits!"
    assert len(train_files) + len(val_files) + len(test_files) == len(sample_files)
