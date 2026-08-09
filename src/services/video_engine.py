"""
Video inference and model loading service for Deepfake Detector Engine.
Decouples video reading, keyframe seeking, face cropping, and batch inference from the UI layer.
"""

import gc
import os
import json
import logging
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

from src.dataset.preprocess import DynamicFaceCropper, preprocess_tensors_batch
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config
from src.utils.checkpoint import clean_state_dict, DEFAULT_THRESHOLD
from src.utils.temporal_aggregation import aggregate_video_predictions

CONFIG = load_config()
APP_CFG = CONFIG.get("app", {})
IMG_SIZE: int = CONFIG.get("preprocessing", {}).get("img_size", 512)
FRAMES_TO_SAMPLE: int = APP_CFG.get("frames_to_sample", 10)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_prediction_engine() -> Tuple[torch.nn.Module, DynamicFaceCropper, bool, float, float]:
    """
    Decoupled model loader function.
    Returns: (pytorch_model, cropper, has_pytorch_weights, classification_threshold, temperature)
    """
    candidate_paths = [
        "models/dual_stream_calibrated.pth",
        "weights/dual_stream_calibrated.pth",
        "dual_stream_calibrated.pth",
        "models/dual_stream_best.pth",
        "weights/dual_stream_best.pth",
        "dual_stream_best.pth"
    ]
    weights_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            weights_path = p
            break

    if weights_path is None:
        try:
            from huggingface_hub import hf_hub_download
            logging.info("No local checkpoint found. Attempting download from HuggingFace Hub...")
            weights_path = hf_hub_download(
                repo_id="yyouretoast/deepfake-detector",
                filename="dual_stream_calibrated.pth"
            )
            logging.info("Downloaded weights from HuggingFace Hub to %s", weights_path)
        except Exception as e:
            logging.warning("Could not download weights from HuggingFace Hub: %s", e)

    opt_threshold = DEFAULT_THRESHOLD
    temperature = 1.0

    sidecar_paths = ["models/dual_stream_detector.json", "dual_stream_detector.json"]
    for sp in sidecar_paths:
        if os.path.exists(sp):
            try:
                with open(sp, "r") as f:
                    meta = json.load(f)
                    opt_threshold = float(meta.get("optimal_threshold", DEFAULT_THRESHOLD))
                    temperature = float(meta.get("temperature", 1.0))
                    break
            except Exception as e:
                logging.warning("Could not load sidecar metadata: %s", e)

    has_weights = weights_path is not None and os.path.exists(weights_path)
    state_dict = None
    if has_weights:
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            opt_threshold = float(checkpoint.get("optimal_threshold", opt_threshold))
            temperature = float(checkpoint.get("temperature", temperature))
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = clean_state_dict(state_dict)

    backbone_name = CONFIG.get("model", {}).get("backbone", "convnext_small")
    pytorch_model = HybridDeepfakeDetector(
        backbone_name=backbone_name, pretrained=False, use_fft_branch=True, config=CONFIG
    )

    if state_dict is not None:
        incompatible_keys = pytorch_model.load_state_dict(state_dict, strict=False)
        missing_critical = [
            k for k in incompatible_keys.missing_keys
            if any(prefix in k for prefix in ["spatial_backbone", "freq_conv", "gate_fc", "classifier"])
        ]
        if missing_critical:
            raise RuntimeError(f"Critical model weights missing from loaded checkpoint: {missing_critical[:5]}")

    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    scale_factor: float = CONFIG.get("preprocessing", {}).get("scale_factor", 1.50)
    cropper = DynamicFaceCropper(scale_factor=scale_factor, target_size=IMG_SIZE, device=DEVICE)

    return pytorch_model, cropper, has_weights, opt_threshold, temperature


def process_video_frames(
    video_path: str,
    pytorch_model: Optional[torch.nn.Module] = None,
    cropper: Optional[DynamicFaceCropper] = None,
    classification_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    has_pytorch_weights: Optional[bool] = None,
    aggregation_method: str = "soft_max",
) -> Optional[Dict[str, Any]]:
    """
    Video inference engine with OpenCV keyframe seeking, AMP autocast, and temporal aggregation.
    """
    if not video_path or not os.path.exists(video_path) or not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    if pytorch_model is None or cropper is None or classification_threshold is None or temperature is None:
        p_model, c_crop, h_weights, threshold, temp = load_prediction_engine()
        pytorch_model = pytorch_model or p_model
        cropper = cropper or c_crop
        classification_threshold = classification_threshold if classification_threshold is not None else threshold
        temperature = temperature if temperature is not None else temp
        if has_pytorch_weights is None:
            has_pytorch_weights = h_weights

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            return None

        if total >= FRAMES_TO_SAMPLE:
            start_frame = max(0, (total - FRAMES_TO_SAMPLE) // 2)
            frame_indices = list(range(start_frame, start_frame + FRAMES_TO_SAMPLE))
        else:
            frame_indices = list(range(total))

        all_faces: List[np.ndarray] = []
        detected_frame_indices: List[int] = []
        detected_timestamps: List[float] = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = cropper.crop_face(rgb)
                if face is not None:
                    all_faces.append(face)
                    detected_frame_indices.append(idx)
                    detected_timestamps.append(float(idx) / fps)
    finally:
        cap.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_faces:
        return None

    numpy_batch, torch_batch = preprocess_tensors_batch(all_faces, device=DEVICE)
    sequence_tensor = torch_batch.unsqueeze(0)

    unwrapped = pytorch_model.module if isinstance(pytorch_model, torch.nn.DataParallel) else pytorch_model

    with torch.inference_mode():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            seq_logits = unwrapped.forward_sequence(sequence_tensor)
            video_prob = float(torch.sigmoid(seq_logits.float() / temperature).mean().item())

    BATCH_SIZE = CONFIG.get("training", {}).get("batch_size", 16)
    all_probs = []

    for i in range(0, len(all_faces), BATCH_SIZE):
        batch_faces = all_faces[i : i + BATCH_SIZE]
        _, sub_torch = preprocess_tensors_batch(batch_faces, device=DEVICE)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                p1 = torch.sigmoid(pytorch_model(sub_torch).float() / temperature)
                p2 = torch.sigmoid(pytorch_model(torch.flip(sub_torch, dims=[-1])).float() / temperature)
                batch_probs = ((p1 + p2) / 2.0).cpu().numpy().tolist()

        all_probs.extend(batch_probs)

    _agg = aggregate_video_predictions(
        scores=all_probs,
        method=aggregation_method,
        threshold=classification_threshold,
    )
    raw_video_prob = (video_prob + _agg["video_score"]) / 2.0

    zipped_data = list(zip(all_faces, all_probs))
    zipped_data.sort(key=lambda x: x[1], reverse=True)

    top_4 = zipped_data[:4]
    sample_faces = [item[0] for item in top_4]
    sample_probs = [item[1] for item in top_4]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "raw_video_prob": raw_video_prob,
        "sample_faces": sample_faces,
        "sample_probs": sample_probs,
        "all_probs": all_probs,
        "all_faces": all_faces,
        "frame_indices": detected_frame_indices,
        "timestamps": detected_timestamps,
    }
