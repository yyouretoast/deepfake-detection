from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.preprocess import DynamicFaceCropper
from src.dataset.loader import perform_graph_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = "/kaggle/working/deepfake_crops_512"
FACE_SIZE = 512
CROP_SCALE_FACTOR = 1.50
FRAMES_PER_VIDEO = 12
NUM_WORKERS = 8


def process_video_fast(
    video_path: str, cropper: DynamicFaceCropper, num_frames: int = 12
) -> List[np.ndarray]:
    """Extract face crop frames array [num_frames, H, W, 3] from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
        step = max(1, total_frames // num_frames)
        crops = []
        frame_idx = 0
        while cap.isOpened() and len(crops) < num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % step == 0:
                crops.append(cropper.crop_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_idx += 1
        return crops
    finally:
        cap.release()


def process_single_video_worker(
    vid_path: str, cropper: DynamicFaceCropper
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Worker task processing a single video file."""
    path_lower = vid_path.lower().replace("\\", "/")
    fake_keywords = [
        "manipulated_sequences",
        "deepfakes",
        "face2face",
        "faceswap",
        "neuraltextures",
        "celeb-synthesis",
        "spliced",
        "fake",
        "swap",
    ]
    real_keywords = ["original_sequences", "youtube-real", "celeb-real", "real", "original"]

    if any(kw in path_lower for kw in fake_keywords):
        is_fake = 1
    elif any(kw in path_lower for kw in real_keywords):
        is_fake = 0
    else:
        is_fake = 0

    vid_id = os.path.splitext(os.path.basename(vid_path))[0]
    out_sub_dir = os.path.join(OUTPUT_DIR, "fake" if is_fake else "real", vid_id)
    os.makedirs(out_sub_dir, exist_ok=True)

    crops = process_video_fast(vid_path, cropper, num_frames=FRAMES_PER_VIDEO)
    if not crops:
        return [], vid_path

    records = []
    for c_idx, crop in enumerate(crops):
        crop_path = os.path.join(out_sub_dir, f"frame_{c_idx:03d}.webp")
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 90])
        records.append({"path": os.path.relpath(crop_path, OUTPUT_DIR), "video_id": vid_id, "label": is_fake})
    return records, None


def main() -> None:
    """Main execution function for dataset face crop extraction."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cropper = DynamicFaceCropper(target_size=FACE_SIZE, scale_factor=CROP_SCALE_FACTOR)
    kaggle_input = "/kaggle/input"

    video_extensions = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV")
    all_videos_set = set()
    for ext in video_extensions:
        for found_p in glob.glob(f"{kaggle_input}/**/{ext}", recursive=True):
            all_videos_set.add(found_p)

    all_videos = sorted(list(all_videos_set))
    logging.info("Discovered %d video files in %s", len(all_videos), kaggle_input)
    if not all_videos:
        logging.error("No video files found in /kaggle/input! Verify mounted datasets.")
        return

    manifest, corrupted_videos = [], []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_video_worker, vp, cropper): vp for vp in all_videos}
        for future in tqdm(as_completed(futures), total=len(all_videos), desc="Extracting Face Crops"):
            try:
                records, corrupted = future.result()
                if records:
                    manifest.extend(records)
                if corrupted:
                    corrupted_videos.append(corrupted)
            except Exception as e:
                vp = futures[future]
                logging.warning("Skipping corrupted video %s: %s", vp, e)
                corrupted_videos.append(vp)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "corrupted_videos.json"), "w", encoding="utf-8") as f:
        json.dump(corrupted_videos, f, indent=2)

    logging.info(
        "Extracted %d face crops across %d valid videos.",
        len(manifest),
        len(all_videos) - len(corrupted_videos),
    )

    try:
        dataset_samples = [(item["path"], item["label"]) for item in manifest]
        train_s, val_s, test_s = perform_graph_split(dataset_samples, val_ratio=0.10, test_ratio=0.10, seed=42)
        with open(os.path.join(OUTPUT_DIR, "splits.json"), "w", encoding="utf-8") as f:
            json.dump({"train": train_s, "val": val_s, "test": test_s}, f, indent=2)
        logging.info("Successfully generated zero-leakage identity splits.json.")
    except Exception as e:
        logging.warning("Graph split generation warning: %s", e)


if __name__ == "__main__":
    main()
