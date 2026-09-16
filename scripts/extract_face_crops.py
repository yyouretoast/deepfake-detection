import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import logging
import os
import sys
import threading
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dataset.domains import DomainClassifier
from src.dataset.loader import perform_graph_split
from src.dataset.preprocess import DynamicFaceCropper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "/kaggle/input"
DEFAULT_OUTPUT_DIR = "/kaggle/working/deepfake_crops_512"
DEFAULT_FACE_SIZE = 512
DEFAULT_CROP_SCALE_FACTOR = 1.50
DEFAULT_FRAMES_PER_VIDEO = 12
DEFAULT_NUM_WORKERS = 8

_thread_local = threading.local()


def get_thread_local_cropper(face_size: int = DEFAULT_FACE_SIZE, scale_factor: float = DEFAULT_CROP_SCALE_FACTOR) -> DynamicFaceCropper:
    """Retrieve or create a thread-local DynamicFaceCropper instance to ensure thread safety."""
    if not hasattr(_thread_local, "cropper"):
        _thread_local.cropper = DynamicFaceCropper(target_size=face_size, scale_factor=scale_factor)
    return _thread_local.cropper


def process_video_fast(
    video_path: str, cropper: DynamicFaceCropper, num_frames: int = 12
) -> list[np.ndarray]:
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
    vid_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cropper: Optional[DynamicFaceCropper] = None,
    face_size: int = DEFAULT_FACE_SIZE,
    scale_factor: float = DEFAULT_CROP_SCALE_FACTOR,
    num_frames: int = DEFAULT_FRAMES_PER_VIDEO,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Worker task processing a single video file with thread-safe face cropper."""
    if cropper is None:
        cropper = get_thread_local_cropper(face_size=face_size, scale_factor=scale_factor)

    domain_info = DomainClassifier.classify(vid_path)
    is_fake = 1 if domain_info.is_fake else 0

    vid_id = os.path.splitext(os.path.basename(vid_path))[0]
    out_sub_dir = os.path.join(output_dir, "fake" if is_fake else "real", vid_id)
    os.makedirs(out_sub_dir, exist_ok=True)

    crops = process_video_fast(vid_path, cropper, num_frames=num_frames)
    if not crops:
        return [], vid_path

    records = []
    for c_idx, crop in enumerate(crops):
        crop_path = os.path.join(out_sub_dir, f"frame_{c_idx:03d}.webp")
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 90])
        records.append({"path": os.path.relpath(crop_path, output_dir), "video_id": vid_id, "label": is_fake})
    return records, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned face crops from video datasets.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Path to input directory containing videos")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory for crops and manifests")
    parser.add_argument("--face_size", type=int, default=DEFAULT_FACE_SIZE, help="Target face crop resolution (width and height)")
    parser.add_argument("--scale_factor", type=float, default=DEFAULT_CROP_SCALE_FACTOR, help="Bounding box scale factor for context margin")
    parser.add_argument("--frames_per_video", type=int, default=DEFAULT_FRAMES_PER_VIDEO, help="Number of frames to sample per video")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker threads")
    return parser.parse_args()


def main() -> None:
    """Main execution function for dataset face crop extraction."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    video_extensions = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV")
    all_videos_set = set()
    for ext in video_extensions:
        for found_p in glob.glob(f"{args.input_dir}/**/{ext}", recursive=True):
            all_videos_set.add(found_p)

    all_videos = sorted(list(all_videos_set))
    logger.info("Discovered %d video files in %s", len(all_videos), args.input_dir)
    if not all_videos:
        logger.error("No video files found in %s! Verify mounted datasets.", args.input_dir)
        return

    manifest, corrupted_videos = [], []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_video_worker,
                vp,
                args.output_dir,
                None,
                args.face_size,
                args.scale_factor,
                args.frames_per_video,
            ): vp
            for vp in all_videos
        }
        for future in tqdm(as_completed(futures), total=len(all_videos), desc="Extracting Face Crops"):
            try:
                records, corrupted = future.result()
                if records:
                    manifest.extend(records)
                if corrupted:
                    corrupted_videos.append(corrupted)
            except (OSError, cv2.error, ValueError, RuntimeError, TypeError, KeyError) as e:
                vp = futures[future]
                logger.warning("Skipping corrupted video %s: %s", vp, e)
                corrupted_videos.append(vp)

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.output_dir, "corrupted_videos.json"), "w", encoding="utf-8") as f:
        json.dump(corrupted_videos, f, indent=2)

    logger.info(
        "Extracted %d face crops across %d valid videos.",
        len(manifest),
        len(all_videos) - len(corrupted_videos),
    )

    try:
        dataset_samples = [(item["path"], item["label"]) for item in manifest]
        train_s, val_s, test_s = perform_graph_split(dataset_samples, val_ratio=0.10, test_ratio=0.10, seed=42)
        with open(os.path.join(args.output_dir, "splits.json"), "w", encoding="utf-8") as f:
            json.dump({"train": train_s, "val": val_s, "test": test_s}, f, indent=2)
        logger.info("Successfully generated zero-leakage identity splits.json.")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as e:
        logger.warning("Graph split generation warning: %s", e)


if __name__ == "__main__":
    main()
