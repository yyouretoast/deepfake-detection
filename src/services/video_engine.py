"""Video inference and model loading service for Deepfake Detector Engine."""

import hashlib
import json
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.config import load_config
from src.dataset.preprocess import DynamicFaceCropper, preprocess_tensors_batch
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import DEFAULT_THRESHOLD, classify_three_zone, clean_state_dict
from src.utils.temporal_aggregation import aggregate_video_predictions

logger = logging.getLogger(__name__)

CONFIG: dict[str, Any] = load_config()
APP_CFG: dict[str, Any] = CONFIG.get("app", {})
IMG_SIZE: int = CONFIG.get("preprocessing", {}).get("img_size", 512)
FRAMES_TO_SAMPLE: int = APP_CFG.get("frames_to_sample", 10)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Expected SHA-256 hash of published checkpoint weights (can be overridden via ENV)
EXPECTED_WEIGHTS_SHA256: Optional[str] = os.getenv(
    "EXPECTED_WEIGHTS_SHA256",
    os.getenv("PUBLISHED_WEIGHTS_SHA256", None)
)


class PredictionEngine(tuple):
    """5-tuple holding prediction components with attribute access for Bayesian thresholds."""

    def __new__(
        cls,
        model: torch.nn.Module,
        cropper: DynamicFaceCropper,
        has_weights: bool,
        threshold: float,
        temperature: float,
        tau_real: float = 0.40,
        tau_fake: float = 0.60,
    ) -> "PredictionEngine":
        instance = super().__new__(cls, (model, cropper, has_weights, threshold, temperature))
        instance.model = model
        instance.cropper = cropper
        instance.has_weights = has_weights
        instance.threshold = threshold
        instance.temperature = temperature
        instance.tau_real = tau_real
        instance.tau_fake = tau_fake
        return instance


def load_prediction_engine(
    weights_path: Optional[str] = None,
) -> PredictionEngine:
    """
    Load prediction engine model weights, sidecar metadata, and face cropper.

    Returns:
        PredictionEngine 5-tuple: (pytorch_model, cropper, has_pytorch_weights, classification_threshold, temperature).
    """
    if weights_path is None:
        candidate_paths = [
            "models/dual_stream_calibrated.pth",
            "weights/dual_stream_calibrated.pth",
            "dual_stream_calibrated.pth",
            "models/dual_stream_best.pth",
            "weights/dual_stream_best.pth",
            "dual_stream_best.pth",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                weights_path = p
                break

    if weights_path is None:
        try:
            from huggingface_hub import hf_hub_download

            logger.info("No local checkpoint found. Attempting download from HuggingFace Hub...")
            weights_path = hf_hub_download(
                repo_id="yyouretoast/deepfake-detector",
                filename="dual_stream_calibrated.pth",
            )
            logger.info("Downloaded weights from HuggingFace Hub to %s", weights_path)
            if weights_path and EXPECTED_WEIGHTS_SHA256:
                sha256 = hashlib.sha256()
                with open(weights_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                computed_hash = sha256.hexdigest()
                if computed_hash.lower() != EXPECTED_WEIGHTS_SHA256.lower():
                    raise ValueError(
                        f"Checksum mismatch for downloaded weights: expected {EXPECTED_WEIGHTS_SHA256}, got {computed_hash}"
                    )
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            logger.warning("Could not download weights from HuggingFace Hub: %s", e)

    opt_threshold = DEFAULT_THRESHOLD
    temperature = 1.0
    tau_real = 0.40
    tau_fake = 0.60

    sidecar_paths = ["models/dual_stream_detector.json", "dual_stream_detector.json"]
    for sp in sidecar_paths:
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    opt_threshold = float(meta.get("optimal_threshold", DEFAULT_THRESHOLD))
                    temperature = float(meta.get("temperature", 1.0))
                    tau_real = float(meta.get("tau_real", 0.40))
                    tau_fake = float(meta.get("tau_fake", 0.60))
                    break
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning("Could not load sidecar metadata: %s", e)

    has_weights = weights_path is not None and os.path.exists(weights_path)
    state_dict: Optional[dict[str, Any]] = None
    if has_weights:
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            opt_threshold = float(checkpoint.get("optimal_threshold", opt_threshold))
            temperature = float(checkpoint.get("temperature", temperature))
            tau_real = float(checkpoint.get("tau_real", tau_real))
            tau_fake = float(checkpoint.get("tau_fake", tau_fake))
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
        freq_prefix = "freq_tower" if getattr(pytorch_model, "frequency_backbone", "legacy") == "resse" else "freq_conv"
        missing_critical = [
            k
            for k in incompatible_keys.missing_keys
            if any(prefix in k for prefix in ["spatial_backbone", freq_prefix, "gate_fc", "classifier"])
        ]
        if missing_critical:
            raise RuntimeError(f"Critical model weights missing from loaded checkpoint: {missing_critical[:5]}")

    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    scale_factor: float = CONFIG.get("preprocessing", {}).get("scale_factor", 1.50)
    cropper = DynamicFaceCropper(scale_factor=scale_factor, target_size=IMG_SIZE, device=DEVICE)

    return PredictionEngine(
        pytorch_model, cropper, has_weights, opt_threshold, temperature, tau_real, tau_fake
    )


def load_temporal_engine(
    weights_path: Optional[str] = None,
) -> Optional[torch.nn.Module]:
    """Loads optional Bi-GRU spatiotemporal consistency head if checkpoint exists."""
    if weights_path is None:
        candidate_paths = [
            "models/temporal_head_best.pth",
            "weights/temporal_head_best.pth",
            "temporal_head_best.pth",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                weights_path = p
                break

    if weights_path and os.path.exists(weights_path):
        try:
            ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            hidden_dim = int(ckpt.get("hidden_dim", 256))
            temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=hidden_dim).to(DEVICE)
            temporal_model.load_state_dict(clean_state_dict(state_dict), strict=False)
            temporal_model.eval()
            logger.info("Loaded Bi-GRU temporal model from %s", weights_path)
            return temporal_model
        except Exception as e:
            logger.warning("Failed to load temporal model from %s: %s", weights_path, e)
    return None


def process_video_frames(
    video_path: str,
    pytorch_model: Optional[torch.nn.Module] = None,
    cropper: Optional[DynamicFaceCropper] = None,
    classification_threshold: Optional[float] = None,
    temperature: Optional[float] = None,
    has_pytorch_weights: Optional[bool] = None,
    aggregation_method: str = "soft_max",
    num_frames: Optional[int] = None,
    temporal_model: Optional[torch.nn.Module] = None,
    tau_real: Optional[float] = None,
    tau_fake: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """
    Video inference engine with OpenCV keyframe seeking, AMP autocast, and temporal aggregation.
    """
    if not video_path or not os.path.exists(video_path) or not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    engine_tau_real = 0.40
    engine_tau_fake = 0.60
    if pytorch_model is None or cropper is None or classification_threshold is None or temperature is None:
        engine = load_prediction_engine()
        p_model, c_crop, h_weights, threshold, temp = engine
        pytorch_model = pytorch_model or p_model
        cropper = cropper or c_crop
        classification_threshold = (
            classification_threshold if classification_threshold is not None else threshold
        )
        temperature = temperature if temperature is not None else temp
        if has_pytorch_weights is None:
            has_pytorch_weights = h_weights
        engine_tau_real = getattr(engine, "tau_real", 0.40)
        engine_tau_fake = getattr(engine, "tau_fake", 0.60)

    effective_tau_real = tau_real if tau_real is not None else engine_tau_real
    effective_tau_fake = tau_fake if tau_fake is not None else engine_tau_fake

    if temporal_model is None:
        temporal_model = load_temporal_engine()

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            return None

        target_frames = num_frames if (num_frames is not None and num_frames > 0) else FRAMES_TO_SAMPLE
        if total >= target_frames:
            step = total / float(target_frames)
            frame_indices = [int(i * step) for i in range(target_frames)]
        else:
            frame_indices = list(range(total))

        all_faces: list[np.ndarray] = []
        detected_frame_indices: list[int] = []
        detected_timestamps: list[float] = []

        # Sequential reading is significantly faster and more accurate than random seeking
        # via cap.set(CAP_PROP_POS_FRAMES). Random seeking on compressed formats (H.264)
        # forces decoding from the nearest I-frame, which is slow and can be inaccurate
        # for Variable Frame Rate videos or containers with bad headers.
        target_set = set(frame_indices)
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_frame in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = cropper.crop_face(rgb, fallback_on_empty=False)
                if face is not None:
                    all_faces.append(face)
                    detected_frame_indices.append(current_frame)
                    detected_timestamps.append(float(current_frame) / fps)
                target_set.discard(current_frame)
                if not target_set:
                    break  # All target frames collected — stop early
            current_frame += 1

    finally:
        cap.release()

    if not all_faces:
        logger.warning("No face detections found in %s; falling back to center crops", video_path)
        cap = cv2.VideoCapture(video_path)
        try:
            target_set = set(frame_indices)
            current_frame = 0
            while cap.isOpened() and target_set:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if current_frame in target_set:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    c = cropper._center_crop(rgb, target_size=cropper.target_size)
                    all_faces.append(c)
                    detected_frame_indices.append(current_frame)
                    detected_timestamps.append(float(current_frame) / fps)
                    target_set.discard(current_frame)
                current_frame += 1
        finally:
            cap.release()

    if not all_faces:
        return None

    from src.utils.interpretability import MODEL_INFERENCE_LOCK

    batch_size = CONFIG.get("training", {}).get("batch_size", 16)
    all_probs: list[float] = []
    unflipped_probs: list[float] = []
    all_embeddings: list[torch.Tensor] = []

    for i in range(0, len(all_faces), batch_size):
        batch_faces = all_faces[i : i + batch_size]
        _, sub_torch = preprocess_tensors_batch(batch_faces, device=DEVICE)

        with MODEL_INFERENCE_LOCK:
            with torch.inference_mode():
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    p1 = torch.sigmoid(pytorch_model(sub_torch).float() / temperature).view(-1)
                    p2 = torch.sigmoid(pytorch_model(torch.flip(sub_torch, dims=[-1])).float() / temperature).view(-1)
                    p_avg = (p1 + p2) / 2.0
                    unflipped_probs.extend([float(val) for val in p1.cpu().numpy().tolist()])
                    if temporal_model is not None and hasattr(pytorch_model, "extract_features"):
                        emb = pytorch_model.extract_features(sub_torch)
                        all_embeddings.append(emb)
                all_probs.extend([float(val) for val in p_avg.cpu().numpy().tolist()])

    frame_attention = None
    if temporal_model is not None and all_embeddings:
        with MODEL_INFERENCE_LOCK:
            with torch.inference_mode():
                seq_tensor = torch.cat(all_embeddings, dim=0).unsqueeze(0)  # [1, T, 512]
                v_logit, v_attn = temporal_model(seq_tensor)
                raw_video_prob = float(torch.sigmoid(v_logit.float() / temperature).item())
                frame_attention = [float(w) for w in v_attn.squeeze(0).cpu().tolist()]
    else:
        _agg = aggregate_video_predictions(
            scores=all_probs,
            method=aggregation_method,
            threshold=classification_threshold,
        )
        raw_video_prob = float(_agg["video_score"])

    three_zone = classify_three_zone(
        raw_video_prob, tau_real=effective_tau_real, tau_fake=effective_tau_fake
    )

    zipped_data = list(zip(all_faces, all_probs))
    zipped_data.sort(key=lambda x: x[1], reverse=True)

    top_4 = zipped_data[:4]
    sample_faces = [item[0] for item in top_4]
    sample_probs = [item[1] for item in top_4]

    return {
        "raw_video_prob": raw_video_prob,
        "sample_faces": sample_faces,
        "sample_probs": sample_probs,
        "all_probs": all_probs,
        "all_faces": all_faces,
        "frame_indices": detected_frame_indices,
        "timestamps": detected_timestamps,
        "three_zone": three_zone,
        "temporal_attention": frame_attention,
        "tau_real": effective_tau_real,
        "tau_fake": effective_tau_fake,
    }

