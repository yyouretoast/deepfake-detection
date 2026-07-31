import pytest
from src.dataset.loader import extract_video_id, perform_graph_split, group_samples_by_video

def test_extract_video_id():
    assert extract_video_id("original_000_f0.png") == "000"
    assert extract_video_id("Deepfakes_001_002_f3.png") == "001"
    assert extract_video_id("045_f12.png") == "045"
    assert extract_video_id("id0_id16_0000.mp4") == "id0"
    assert extract_video_id("aagfhxzi.mp4") == "aagfhxzi"
    assert extract_video_id("000_003.mp4") == "000"

def test_group_samples_by_video():
    flat_samples = [
        ("Deepfakes_001_002_f0.png", 1),
        ("Deepfakes_001_002_f1.png", 1),
        ("original_000_f0.png", 0),
        ("original_000_f1.png", 0),
    ]
    grouped = group_samples_by_video(flat_samples)
    assert len(grouped) == 2
    for frame_paths, label in grouped:
        assert isinstance(frame_paths, list)
        assert len(frame_paths) == 2
        assert isinstance(label, int)

def test_perform_graph_split_zero_leakage():
    # Numeric IDs
    sample_files = []
    for vid in range(10):
        for frame_idx in range(5):
            sample_files.append(f"Deepfakes_{vid:03d}_000_f{frame_idx}.png")
            
    # Alphanumeric IDs
    for i in range(5):
        sample_files.append(f"id{i}_id{i+1}_000{i}.png")
        
    # UUIDs
    for uuid in ["aagfhxzi", "bbgfhxzi", "ccgfhxzi"]:
        sample_files.append(f"{uuid}.mp4")

    train_files, val_files, test_files = perform_graph_split(sample_files, test_size=0.2, val_size=0.2, seed=42)

    train_vids = set(extract_video_id(f) for f in train_files)
    val_vids = set(extract_video_id(f) for f in val_files)
    test_vids = set(extract_video_id(f) for f in test_files)

    # Strict Zero Data Leakage Assertions
    assert len(train_vids.intersection(val_vids)) == 0, "Data leakage detected between Train and Val splits!"
    assert len(train_vids.intersection(test_vids)) == 0, "Data leakage detected between Train and Test splits!"
    assert len(val_vids.intersection(test_vids)) == 0, "Data leakage detected between Val and Test splits!"
    assert len(train_files) + len(val_files) + len(test_files) == len(sample_files)
