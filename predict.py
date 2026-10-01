"""Standalone Turnkey Forensic Inference CLI for Deepfake Detection.

Supports single-frame facial crops, raw images, and full video sequences.
Evaluates using the Dual-Stream Spatial-Frequency model with Platt calibration
and operational Bayesian 3-zone triage.

Usage Examples:
    # Single image evaluation
    python predict.py --input path/to/face.jpg

    # Video evaluation with temporal consistency
    python predict.py --input path/to/video.mp4 --device cuda

    # Batch output to JSON
    python predict.py --input path/to/media.mp4 --json_output result.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any

import cv2
import numpy as np
import torch

from src.dataset.preprocess import DynamicFaceCropper
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.models.temporal_head import BiGRUTemporalDetector
from src.utils.checkpoint import clean_state_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict_cli")


def find_default_weights() -> tuple[str | None, str | None]:
    """Finds best available dual-stream and temporal checkpoints."""
    dual_candidates = [
        "models/dual_stream_calibrated.pth",
        "results/release1_run/dual_stream_calibrated.pth",
        "results/release1_run/dual_stream_best.pth",
        "weights/dual_stream_calibrated.pth",
        "weights/dual_stream_best.pth",
        "dual_stream_calibrated.pth",
        "dual_stream_best.pth",
    ]
    dual_path = next((p for p in dual_candidates if os.path.exists(p)), None)

    temporal_candidates = [
        "models/temporal_head_best.pth",
        "results/release1_run/temporal_head_best.pth",
        "weights/temporal_head_best.pth",
        "temporal_head_best.pth",
    ]
    temporal_path = next((p for p in temporal_candidates if os.path.exists(p)), None)
    return dual_path, temporal_path


def load_detector(
    weights_path: str, device: torch.device
) -> tuple[HybridDeepfakeDetector, dict[str, float]]:
    logger.info("Loading dual-stream model from: %s", weights_path)
    model = HybridDeepfakeDetector(pretrained=False, frequency_backbone="resse")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    calib_params = {
        "platt_scale_a": 0.2783,
        "platt_bias_b": 0.4089,
        "optimal_threshold": 0.2600,
        "temperature": 3.5931,
    }

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        for k in calib_params:
            if k in ckpt:
                calib_params[k] = float(ckpt[k])
    else:
        state_dict = ckpt

    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model, calib_params


def classify_zone(prob: float, tau_star: float = 0.26) -> dict[str, Any]:
    """Classifies posterior probability into Bayesian 3-zone operational triage."""
    if prob < 0.40:
        zone = "Zone 1: Authentic Clearance"
        action = "Cleared (Low synthetic probability)"
        risk = "Low"
    elif prob <= 0.60:
        zone = "Zone 2: Human Forensic Review"
        action = "Flagged for manual expert review (Ambiguous posterior)"
        risk = "Moderate (High forensic entropy)"
    else:
        zone = "Zone 3: Confirmed Synthetic"
        action = "Quarantined / Flagged Synthetic"
        risk = "Critical (High synthetic certainty)"

    binary_decision = "SYNTHETIC" if prob >= tau_star else "AUTHENTIC"
    return {
        "binary_decision": binary_decision,
        "zone": zone,
        "action": action,
        "risk_level": risk,
    }


def predict_image(
    image_path: str,
    model: HybridDeepfakeDetector,
    cropper: DynamicFaceCropper,
    calib: dict[str, float],
    device: torch.device,
    is_crop: bool = False,
) -> dict[str, Any]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect & crop face using YuNet/MTCNN unless already a pre-cropped facial patch
    h, w = img_rgb.shape[:2]
    already_cropped = is_crop or (h == w and (h == 256 or h == 512))
    if already_cropped:
        crop_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        face_detected = True
    else:
        crop_rgb = cropper.crop_face(img_rgb, target_size=256)
        if crop_rgb is None:
            crop_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            face_detected = False
        else:
            face_detected = True

    crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    crop_tensor = (crop_tensor - mean) / std
    batch = crop_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits, aux_logits = model(batch, return_aux=True)
        raw_logit = float(logits[0].item())
        aux_logit = float(aux_logits[0].item())

    # Platt calibration
    a = calib["platt_scale_a"]
    b = calib["platt_bias_b"]
    calibrated_prob = float(1.0 / (1.0 + np.exp(-(a * raw_logit + b))))
    triage = classify_zone(calibrated_prob, calib["optimal_threshold"])

    return {
        "input_path": image_path,
        "media_type": "image",
        "face_detected": face_detected,
        "raw_logit": round(raw_logit, 4),
        "aux_spectral_logit": round(aux_logit, 4),
        "calibrated_posterior_probability": round(calibrated_prob, 4),
        "triage": triage,
    }


def predict_video(
    video_path: str,
    model: HybridDeepfakeDetector,
    temporal_model: BiGRUTemporalDetector | None,
    cropper: DynamicFaceCropper,
    calib: dict[str, float],
    device: torch.device,
    max_frames: int = 32,
    frame_stride: int = 2,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames_rgb: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if len(frames_rgb) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames_rgb:
        raise ValueError(f"No frames could be extracted from video: {video_path}")

    crops_rgb: list[np.ndarray] = []
    for f in frames_rgb:
        c = cropper.crop_face(f, target_size=256)
        if c is None:
            c = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
        crops_rgb.append(c)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensors = [torch.from_numpy(c).permute(2, 0, 1).float() / 255.0 for c in crops_rgb]
    batch = torch.stack(tensors, dim=0)  # (T, 3, 256, 256)
    batch = (batch - mean) / std
    batch = batch.to(device)

    with torch.inference_mode():
        logits, _ = model(batch, return_aux=True)
        raw_logits = logits.view(-1).cpu().numpy()

    a = calib["platt_scale_a"]
    b = calib["platt_bias_b"]
    frame_probs = 1.0 / (1.0 + np.exp(-(a * raw_logits + b)))

    mean_prob = float(np.mean(frame_probs))
    max_prob = float(np.max(frame_probs))

    # Temporal head inference if available
    temporal_prob = None
    if temporal_model is not None and len(crops_rgb) >= 4:
        with torch.inference_mode():
            feats = model.extract_features(batch)  # (T, 512)
            seq_feats = feats.unsqueeze(0)  # (1, T, 512)
            seq_logits, _ = temporal_model(seq_feats)
            temporal_logit = float(seq_logits[0].item())
            temporal_prob = float(1.0 / (1.0 + np.exp(-(a * temporal_logit + b))))

    effective_prob = temporal_prob if temporal_prob is not None else mean_prob
    triage = classify_zone(effective_prob, calib["optimal_threshold"])

    return {
        "input_path": video_path,
        "media_type": "video",
        "total_video_frames": total_frames,
        "analyzed_frames": len(crops_rgb),
        "fps": round(fps, 2) if fps else 30.0,
        "frame_probabilities": [round(float(p), 4) for p in frame_probs],
        "mean_frame_probability": round(mean_prob, 4),
        "max_frame_probability": round(max_prob, 4),
        "temporal_bigru_probability": round(temporal_prob, 4) if temporal_prob else None,
        "calibrated_posterior_probability": round(effective_prob, 4),
        "triage": triage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Turnkey Forensic Deepfake Inference CLI")
    parser.add_argument("input_pos", nargs="?", default=None, help="Path to input image or video (positional)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input image or video")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to dual-stream weights checkpoint")
    parser.add_argument("--temporal_weights", type=str, default=None, help="Path to temporal head checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true", help="Output JSON directly to stdout")
    parser.add_argument("--json_output", "-o", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--is_crop", action="store_true", help="Input image is already a pre-cropped facial patch; bypass full-frame face detection.")
    args = parser.parse_args()

    input_target = args.input or args.input_pos
    if not input_target:
        parser.error("The input path is required (provide as positional argument or via --input/-i).")

    device = torch.device(args.device)
    dual_ckpt, temp_ckpt = find_default_weights()
    weights_path = args.weights or dual_ckpt
    temporal_path = args.temporal_weights or temp_ckpt

    if not weights_path or not os.path.exists(weights_path):
        logger.error("No valid checkpoint found! Please specify --weights.")
        sys.exit(1)

    model, calib = load_detector(weights_path, device)
    cropper = DynamicFaceCropper(target_size=256, device=device)

    temporal_model = None
    if temporal_path and os.path.exists(temporal_path):
        try:
            temporal_model = BiGRUTemporalDetector(embed_dim=512, hidden_dim=256, use_deltas=True, use_max_pool=True)
            t_state = torch.load(temporal_path, map_location=device, weights_only=False)
            temporal_model.load_state_dict(clean_state_dict(t_state), strict=False)
            temporal_model.to(device)
            temporal_model.eval()
            logger.info("Loaded Bi-GRU temporal head from: %s", temporal_path)
        except Exception as e:
            logger.warning("Could not load temporal head: %s", e)

    inp = input_target.lower()
    is_video = inp.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    if is_video:
        result = predict_video(input_target, model, temporal_model, cropper, calib, device)
    else:
        result = predict_image(input_target, model, cropper, calib, device, is_crop=args.is_crop)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Print clean CLI report
        triage = result["triage"]
        print("\n" + "=" * 60)
        print("       FORENSIC MEDIA VERIFICATION REPORT")
        print("=" * 60)
        print(f" Target Media:    {result['input_path']}")
        print(f" Media Type:      {result['media_type'].upper()}")
        print(f" Posterior Prob:  {result['calibrated_posterior_probability']:.4f}")
        print(f" Decision:        {triage['binary_decision']}")
        print(f" Bayesian Zone:   {triage['zone']}")
        print(f" Triage Action:   {triage['action']}")
        print(f" Forensic Risk:   {triage['risk_level']}")
        print("=" * 60 + "\n")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved forensic analysis JSON -> %s", args.json_output)


if __name__ == "__main__":
    main()
