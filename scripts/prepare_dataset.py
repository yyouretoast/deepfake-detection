import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.dataset.preprocess import DynamicFaceCropper

def process_video(video_path: str, output_dir: str, max_frames: int, img_size: int) -> int:
    try:
        cropper = DynamicFaceCropper(target_size=img_size, scale_factor=1.30)
        saved_paths = cropper.extract_faces_from_video(video_path, output_dir, max_frames=max_frames, target_size=img_size)
        return len(saved_paths)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract face crops from videos.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MP4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted faces")
    parser.add_argument("--max_frames", type=int, default=30, help="Maximum number of frames to extract per video")
    parser.add_argument("--img_size", type=int, default=256, help="Target image size for face crops")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"No videos found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction with {args.num_workers} workers...")

    total_extracted = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_video,
                v_path,
                args.output_dir,
                args.max_frames,
                args.img_size
            ): v_path for v_path in video_files
        }

        for future in tqdm(as_completed(futures), total=len(video_files), desc="Processing Videos"):
            total_extracted += future.result()

    print(f"Extraction complete! Saved {total_extracted} face crops to {args.output_dir}")

if __name__ == "__main__":
    main()
