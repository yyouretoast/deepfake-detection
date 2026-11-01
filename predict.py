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
from src.models.hybrid_detector import (
    HybridDeepfakeDetector,
    assert_critical_weights_loaded,
)
from src.models.temporal_head import BiGRUTemporalDetector
from src.services.video_engine import process_video_frames
from src.utils.calibration_release1 import (
    EFFECTIVE_TEMPERATURE,
    OPTIMAL_THRESHOLD,
    PLATT_BIAS_B,
    PLATT_SCALE_A,
    TAU_FAKE,
    TAU_REAL,
    TEMPORAL_OPTIMAL_THRESHOLD,
)
from src.utils.checkpoint import classify_three_zone, clean_state_dict

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
        "platt_scale_a": PLATT_SCALE_A,
        "platt_bias_b": PLATT_BIAS_B,
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "temperature": EFFECTIVE_TEMPERATURE,
    }

    # Traceability: record whether the calibration constants actually came from the
    # checkpoint or fell back to the hardcoded Release-1 defaults, and surface this
    # in every report so verdicts can be tied to the run that produced them.
    calibration_source = "hardcoded_release1_defaults"
    ckpt_source_keys = ["temperature", "optimal_threshold"]

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        calib_dict = ckpt.get("calibration", ckpt) if isinstance(ckpt.get("calibration"), dict) else ckpt
        if isinstance(calib_dict, dict):
            calibration_source = "checkpoint" if all(
                str(k) in calib_dict or str(k) in ckpt for k in ckpt_source_keys
            ) else "partial_checkpoint"
        for k in calib_params:
            if k in calib_dict:
                calib_params[k] = float(calib_dict[k])
            elif k in ckpt:
                calib_params[k] = float(ckpt[k])
        for k in ("tau_real", "tau_fake"):
            if k in calib_dict:
                calib_params[k] = float(calib_dict[k])
        if "effective_temperature" in calib_dict:
            calib_params["temperature"] = float(calib_dict["effective_temperature"])
    else:
        state_dict = ckpt

    if calibration_source != "checkpoint":
        logger.warning(
            "Calibration constants (%s) were loaded from %s, not from %s. "
            "Verdict thresholds may not match the weights' training run.",
            ", ".join(calib_params),
            calibration_source,
            weights_path,
        )
    calib_params["calibration_source"] = calibration_source
    calib_params["weights_source"] = weights_path

    incompatible_keys = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    if len(state_dict) == 0:
        # Placeholder weights (e.g. CLI smoke tests using {"model_state_dict": {}}): there
        # are no parameters to be silently dropped, so keep the legacy structural-report
        # behavior but say loudly that the backbones are randomly initialized.
        logger.warning(
            "Checkpoint %s contains no model weights; running with randomly initialized "
            "backbones (placeholder-weights behavior).",
            weights_path,
        )
    else:
        assert_critical_weights_loaded(model, incompatible_keys)
    model.to(device)
    model.eval()
    return model, calib_params


def classify_zone(prob: float, tau_star: float = OPTIMAL_THRESHOLD) -> dict[str, Any]:
    """Classifies posterior probability into the Bayesian 3-zone operational triage.

    Thin adapter over ``src.utils.checkpoint.classify_three_zone`` so the CLI and the
    serving engine share a single triage policy; the legacy Zone 1/2/3 wording is kept
    for CLI/JSON report compatibility.
    """
    zone_result = classify_three_zone(prob, tau_real=TAU_REAL, tau_fake=TAU_FAKE)
    zone_key = zone_result["zone"]
    if zone_key == "high_confidence_real":
        zone = "Zone 1: Authentic Clearance"
        action = "Cleared (Low synthetic probability)"
        risk = "Low"
    elif zone_key == "ambiguity_zone":
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
        "calibration": {
            "source": calib.get("calibration_source", "unknown"),
            "weights": calib.get("weights_source", "unknown"),
            "platt_scale_a": calib["platt_scale_a"],
            "platt_bias_b": calib["platt_bias_b"],
            "temperature": calib["temperature"],
            "optimal_threshold": calib["optimal_threshold"],
        },
    }


def predict_video(
    video_path: str,
    model: HybridDeepfakeDetector,
    temporal_model: BiGRUTemporalDetector | None,
    cropper: DynamicFaceCropper,
    calib: dict[str, float],
    device: torch.device,
    max_frames: int = 8,
) -> dict[str, Any]:
    """Video verdict JSON, delegating sampling/inference to the serving engine.

    Unifies the CLI with ``src.services.video_engine.process_video_frames`` so the CLI and
    the Streamlit app share one video pipeline: uniform keyframe sampling, horizontal-flip
    TTA (both Platt-calibrated), and BiGRU temporal scoring when a temporal head is present.
    Mapping keeps the legacy JSON schema (names and rounding) unchanged:

    - ``frame_probabilities`` / ``mean`` / ``max``  <- engine per-frame calibrated probs
    - ``temporal_bigru_probability``                <- engine sigmoid(video logit) if temporal ran
    - ``calibrated_posterior_probability``          <- temporal video prob if temporal ran,
      else mean frame probability (legacy aggregation policy: mean)
    """
    result = process_video_frames(
        video_path,
        pytorch_model=model,
        cropper=cropper,
        aggregation_method="mean",
        num_frames=max_frames,
        temporal_model=temporal_model,
        platt_a=calib["platt_scale_a"],
        platt_b=calib["platt_bias_b"],
        device=device,
    )
    if result is None:
        raise ValueError(f"Could not open video or no faces detected: {video_path}")

    frame_probs: list[float] = result["all_probs"]
    mean_prob = float(np.mean(frame_probs))
    max_prob = float(np.max(frame_probs))

    # Temporal-head verdict if the engine ran it (attention list is non-None only then).
    temporal_prob: float | None = None
    if result.get("temporal_attention") is not None:
        temporal_prob = float(result["raw_video_prob"])

    effective_prob = temporal_prob if temporal_prob is not None else mean_prob
    if temporal_prob is not None:
        if "temporal_optimal_threshold" in calib:
            effective_threshold = float(calib["temporal_optimal_threshold"])
        else:
            effective_threshold = TEMPORAL_OPTIMAL_THRESHOLD
            logger.warning(
                "temporal_optimal_threshold missing from the calibration sidecar; using "
                f"legacy fallback {TEMPORAL_OPTIMAL_THRESHOLD} (provenance unverified, "
                "see AUDIT_REPORT.md section 3.5)."
            )
    else:
        effective_threshold = calib["optimal_threshold"]
    triage = classify_zone(effective_prob, effective_threshold)

    return {
        "input_path": video_path,
        "media_type": "video",
        "total_video_frames": int(result["total_frames"]),
        "analyzed_frames": len(result["all_faces"]),
        "fps": round(float(result["fps"]), 2) if result["fps"] else 30.0,
        "frame_probabilities": [round(float(p), 4) for p in frame_probs],
        "mean_frame_probability": round(mean_prob, 4),
        "max_frame_probability": round(max_prob, 4),
        "temporal_bigru_probability": round(temporal_prob, 4) if temporal_prob is not None else None,
        "calibrated_posterior_probability": round(effective_prob, 4),
        "triage": triage,
        "calibration": {
            "source": calib.get("calibration_source", "unknown"),
            "weights": calib.get("weights_source", "unknown"),
            "platt_scale_a": calib["platt_scale_a"],
            "platt_bias_b": calib["platt_bias_b"],
            "temperature": calib["temperature"],
            "optimal_threshold": calib["optimal_threshold"],
        },
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
        print(f" 3-Zone Triage:   {triage['zone']}")
        print(f" Triage Action:   {triage['action']}")
        print(f" Forensic Risk:   {triage['risk_level']}")
        print("-" * 60)
        print(
            f" Calibration:     {calib.get('calibration_source', 'unknown')}"
            f" | weights: {calib.get('weights_source', 'unknown')}"
        )
        print(
            f"                  a={calib.get('platt_scale_a', float('nan')):.4f}"
            f" b={calib.get('platt_bias_b', float('nan')):.4f}"
            f" T={calib.get('temperature', float('nan')):.4f}"
            f" tau*={calib.get('optimal_threshold', float('nan')):.4f}"
        )
        print("=" * 60 + "\n")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved forensic analysis JSON -> %s", args.json_output)


if __name__ == "__main__":
    main()
