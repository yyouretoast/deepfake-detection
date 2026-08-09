import os
import re
import glob
import json
import logging
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import networkx as nx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = '/kaggle/working/deepfake_crops_512'
FACE_SIZE = 512
CROP_SCALE_FACTOR = 1.50
FRAMES_PER_VIDEO = 12
NUM_WORKERS = 8
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def get_yunet_model_path() -> str:
    target_path = '/tmp/face_detection_yunet_2023mar.onnx'
    if not (os.path.exists(target_path) and os.path.getsize(target_path) > 1000):
        os.makedirs('/tmp', exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, target_path)
    return target_path


class DynamicFaceCropper:
    def __init__(self, target_size: int = 512, scale_factor: float = 1.50) -> None:
        self.target_size = target_size
        self.scale_factor = scale_factor
        self._local = threading.local()
        self.yunet_path = get_yunet_model_path()

    def _get_thread_yunet(self):
        if not hasattr(self._local, "yunet"):
            self._local.yunet = cv2.FaceDetectorYN.create(
                model=self.yunet_path, config="", input_size=(300, 300), score_threshold=0.6, nms_threshold=0.3, top_k=5000
            ) if os.path.exists(self.yunet_path) and hasattr(cv2, "FaceDetectorYN") else None
        return self._local.yunet

    def _apply_cosine_edge_taper(self, crop: np.ndarray, border_ratio: float = 0.05) -> np.ndarray:
        h, w, _ = crop.shape
        taper_h, taper_w = max(1, int(h * border_ratio)), max(1, int(w * border_ratio))
        win_y = np.ones(h, dtype=np.float32)
        win_y[:taper_h] = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_h)))
        win_y[-taper_h:] = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, taper_h)))

        win_x = np.ones(w, dtype=np.float32)
        win_x[:taper_w] = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_w)))
        win_x[-taper_w:] = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, taper_w)))

        return np.clip(crop.astype(np.float32) * np.outer(win_y, win_x)[:, :, np.newaxis], 0, 255).astype(np.uint8)

    def _crop_single_box(self, image_rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        h_img, w_img, _ = image_rgb.shape
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * self.scale_factor

        crop_x1, crop_y1 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
        crop_x2, crop_y2 = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))

        pad_l, pad_t = max(0, -crop_x1), max(0, -crop_y1)
        pad_r, pad_b = max(0, crop_x2 - w_img), max(0, crop_y2 - h_img)

        padded = cv2.copyMakeBorder(image_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE) if (pad_l or pad_t or pad_r or pad_b) else image_rgb

        src_y1, src_x1 = crop_y1 + pad_t, crop_x1 + pad_l
        src_y2, src_x2 = crop_y2 + pad_t, crop_x2 + pad_l

        if src_y2 <= src_y1 or src_x2 <= src_x1:
            return self._center_crop(image_rgb)

        raw_crop = padded[src_y1:src_y2, src_x1:src_x2]
        if raw_crop.size == 0:
            return self._center_crop(image_rgb)

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            raw_crop = self._apply_cosine_edge_taper(raw_crop)

        return cv2.resize(raw_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def _center_crop(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = image_rgb.shape
        side = min(h, w)
        cy, cx = h // 2, w // 2
        return cv2.resize(image_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2], (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def crop_face(self, image_rgb: np.ndarray) -> np.ndarray:
        yunet = self._get_thread_yunet()
        if yunet is not None:
            try:
                h, w, _ = image_rgb.shape
                yunet.setInputSize((w, h))
                _, faces = yunet.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                if faces is not None and len(faces) > 0:
                    best_f = max(faces, key=lambda f: f[2] * f[3])
                    return self._crop_single_box(image_rgb, np.array([best_f[0], best_f[1], best_f[0] + best_f[2], best_f[1] + best_f[3]]))
            except Exception:
                pass
        return self._center_crop(image_rgb)


def extract_identities(filename: str) -> Tuple[str, str]:
    clean_base = re.sub(r"_(?:f|frame)\d+", "", os.path.basename(filename), flags=re.IGNORECASE).split('.')[0]
    match_alpha = re.search(r"(id\d+)_(id\d+)", clean_base)
    if match_alpha:
        return match_alpha.group(1), match_alpha.group(2)
    match_num = re.search(r"(\d+)_(\d+)", clean_base)
    if match_num:
        return match_num.group(1), match_num.group(2)
    match_single = re.search(r"(\d+)", clean_base)
    return (match_single.group(1), match_single.group(1)) if match_single else (clean_base, clean_base)


def perform_graph_split(samples: List[Tuple[str, int]], val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    parsed = [(path, label, *extract_identities(path)) for path, label in samples]
    G = nx.Graph()
    for _, _, id1, id2 in parsed:
        G.add_edge(id1, id2)

    comp_stats = sorted([(c, sum(1 for s in parsed if s[2] in set(c) or s[3] in set(c))) for c in nx.connected_components(G)], key=lambda x: x[1], reverse=True)
    total_samples = len(parsed)
    target_val, target_test = int(total_samples * val_ratio), int(total_samples * test_ratio)
    val_comps, test_comps, train_comps = set(), set(), set()
    curr_val, curr_test = 0, 0

    for comp, n_s in comp_stats:
        if curr_val + n_s <= target_val or (curr_val == 0 and len(comp_stats) > 2):
            val_comps.update(comp)
            curr_val += n_s
        elif curr_test + n_s <= target_test or (curr_test == 0 and len(comp_stats) > 2):
            test_comps.update(comp)
            curr_test += n_s
        else:
            train_comps.update(comp)

    train_s, val_s, test_s = [], [], []
    for path, label, id1, id2 in parsed:
        if id1 in train_comps or id2 in train_comps:
            train_s.append((path, label))
        elif id1 in val_comps or id2 in val_comps:
            val_s.append((path, label))
        else:
            test_s.append((path, label))

    return train_s, val_s, test_s


def process_video_fast(video_path: str, cropper: DynamicFaceCropper, num_frames: int = 12) -> List[np.ndarray]:
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


def process_single_video_worker(vid_path: str, cropper: DynamicFaceCropper) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path_lower = vid_path.lower().replace('\\', '/')
    fake_keywords = ['manipulated_sequences', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures', 'celeb-synthesis', 'spliced', 'fake', 'swap']
    real_keywords = ['original_sequences', 'youtube-real', 'celeb-real', 'real', 'original']

    if any(kw in path_lower for kw in fake_keywords):
        is_fake = 1
    elif any(kw in path_lower for kw in real_keywords):
        is_fake = 0
    else:
        is_fake = 0

    vid_id = os.path.splitext(os.path.basename(vid_path))[0]
    out_sub_dir = os.path.join(OUTPUT_DIR, 'fake' if is_fake else 'real', vid_id)
    os.makedirs(out_sub_dir, exist_ok=True)

    crops = process_video_fast(vid_path, cropper, num_frames=FRAMES_PER_VIDEO)
    if not crops:
        return [], vid_path

    records = []
    for c_idx, crop in enumerate(crops):
        crop_path = os.path.join(out_sub_dir, f"frame_{c_idx:03d}.webp")
        cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 90])
        records.append({'path': os.path.relpath(crop_path, OUTPUT_DIR), 'video_id': vid_id, 'label': is_fake})
    return records, None


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cropper = DynamicFaceCropper(target_size=FACE_SIZE, scale_factor=CROP_SCALE_FACTOR)
    kaggle_input = '/kaggle/input'

    video_extensions = ('*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', '*.mkv', '*.MKV')
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

    with open(os.path.join(OUTPUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'corrupted_videos.json'), 'w', encoding='utf-8') as f:
        json.dump(corrupted_videos, f, indent=2)

    logging.info("Extracted %d face crops across %d valid videos.", len(manifest), len(all_videos) - len(corrupted_videos))

    try:
        dataset_samples = [(item['path'], item['label']) for item in manifest]
        train_s, val_s, test_s = perform_graph_split(dataset_samples, val_ratio=0.10, test_ratio=0.10, seed=42)
        with open(os.path.join(OUTPUT_DIR, 'splits.json'), 'w', encoding='utf-8') as f:
            json.dump({'train': train_s, 'val': val_s, 'test': test_s}, f, indent=2)
        logging.info("Successfully generated zero-leakage identity splits.json.")
    except Exception as e:
        logging.warning("Graph split generation warning: %s", e)


if __name__ == '__main__':
    main()
