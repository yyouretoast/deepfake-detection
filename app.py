import gc
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hashlib
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import shutil

from src.dataset.preprocess import DynamicFaceCropper, preprocess_tensors_batch
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.config import load_config
from src.utils.checkpoint import clean_state_dict, normalize_confidence, DEFAULT_THRESHOLD
from src.utils.temporal_aggregation import aggregate_video_predictions

CONFIG = load_config()
APP_CFG = CONFIG.get("app", {})
IMG_SIZE: int = CONFIG.get("preprocessing", {}).get("img_size", 512)
FRAMES_TO_SAMPLE: int = APP_CFG.get("frames_to_sample", 10)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Grad-CAM Heatmap Engine for ConvNeXt Backbone
# ---------------------------------------------------------------------------

class ConvNeXtGradCAM:
    def __init__(self, model: HybridDeepfakeDetector):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        target_layer = self.model.spatial_backbone[-1]
        self.forward_handle = target_layer.register_forward_hook(self._save_feature_maps)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            logits = self.model(input_tensor)
            scalar_logit = logits.squeeze()
            scalar_logit.backward()

        if self.feature_maps is None or self.gradients is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i]

        cam = F.relu(cam).detach().cpu().numpy()
        denom = cam.max() - cam.min()
        if denom > 1e-6:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    def remove_hooks(self):
        try:
            self.forward_handle.remove()
            self.backward_handle.remove()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_prediction_engine() -> Tuple[torch.nn.Module, DynamicFaceCropper, bool, float, float]:
    """
    Decoupled cached model loader using Streamlit cache_resource.
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


# ---------------------------------------------------------------------------
# Interpretability Diagnostics Generator
# ---------------------------------------------------------------------------

def generate_face_diagnostics(
    model: HybridDeepfakeDetector,
    face_rgb: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Generates 4-panel interpretability representations:
      (a) Original RGB Face Crop
      (b) SRM High-Pass Spatial Noise Residual Map
      (c) Centered 2D Real FFT Log-Magnitude Spectrum
      (d) Grad-CAM Spatial ConvNeXt Attention Overlay Heatmap
    """
    img_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        srm_out = model.srm(img_tensor)                           # [1, 9, H, W]
        bayar_out = model.bayar(img_tensor)                       # [1, 1, H, W]
        noise_combined = torch.cat([srm_out, bayar_out], dim=1)   # [1, 10, H, W]
        freq_maps = model.fft(noise_combined)                     # [1, 20, H, W]

    # Panel B: SRM High-Pass Residual Noise Map
    srm_map = srm_out[0].abs().mean(dim=0).cpu().numpy()
    srm_norm = (srm_map - srm_map.min()) / max(srm_map.max() - srm_map.min(), 1e-6)
    srm_uint8 = (srm_norm * 255.0).astype(np.uint8)
    srm_colored = cv2.applyColorMap(srm_uint8, cv2.COLORMAP_VIRIDIS)
    srm_rgb = cv2.cvtColor(srm_colored, cv2.COLOR_BGR2RGB)

    # Panel C: Centered 2D Real FFT Log-Magnitude Spectrum
    mag_maps = freq_maps[0, :10].cpu().numpy()
    mean_mag = np.mean(mag_maps, axis=0)
    fft_centered = np.fft.fftshift(mean_mag)
    fft_norm = (fft_centered - fft_centered.min()) / max(fft_centered.max() - fft_centered.min(), 1e-6)
    fft_uint8 = (fft_norm * 255.0).astype(np.uint8)
    fft_colored = cv2.applyColorMap(fft_uint8, cv2.COLORMAP_MAGMA)
    fft_rgb = cv2.cvtColor(fft_colored, cv2.COLOR_BGR2RGB)

    # Panel D: Grad-CAM Heatmap Overlay
    grad_cam = ConvNeXtGradCAM(model)
    cam_map = grad_cam.generate_heatmap(img_tensor.clone())
    grad_cam.remove_hooks()

    cam_uint8 = (cam_map * 255.0).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(face_rgb, 0.6, cam_rgb, 0.4, 0)

    return {
        "original": face_rgb,
        "srm_residual": srm_rgb,
        "fft_spectrum": fft_rgb,
        "gradcam_overlay": cam_overlay,
    }


# ---------------------------------------------------------------------------
# Video Inference Engine
# ---------------------------------------------------------------------------

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
        if total <= 0:
            return None

        if total >= FRAMES_TO_SAMPLE:
            start_frame = max(0, (total - FRAMES_TO_SAMPLE) // 2)
            frame_indices = list(range(start_frame, start_frame + FRAMES_TO_SAMPLE))
        else:
            frame_indices = list(range(total))
        frames_rgb: List[np.ndarray] = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(rgb)
    finally:
        cap.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not frames_rgb:
        return None

    faces_per_frame = [cropper.crop_face(f) for f in frames_rgb]
    all_faces = [f for f in faces_per_frame if f is not None]
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
    }


# ---------------------------------------------------------------------------
# Streamlit UI Rendering
# ---------------------------------------------------------------------------

def render_ui() -> None:
    """Renders the Streamlit frontend layout and handles session state."""
    st.set_page_config(
        page_title="Dual-Stream Deepfake Detector",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main {
            background-color: #0b0f19;
        }
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .header-box {
            text-align: center;
            padding: 28px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(147, 51, 234, 0.15));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
        }
        .card-workflow {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }
        .result-card-fake {
            text-align: center;
            padding: 24px;
            border-radius: 16px;
            background: rgba(220, 38, 38, 0.12);
            border: 2px solid #ef4444;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(239, 68, 68, 0.2);
        }
        .result-card-real {
            text-align: center;
            padding: 24px;
            border-radius: 16px;
            background: rgba(34, 197, 94, 0.12);
            border: 2px solid #22c55e;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(34, 197, 94, 0.2);
        }
        .sidebar-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .badge-tag {
            display: inline-block;
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        /* Mobile Responsive Media Queries */
        @media (max-width: 640px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 12px !important;
            }
            .header-box {
                padding: 16px !important;
            }
            .header-box h1 {
                font-size: 22px !important;
            }
            .result-card-fake, .result-card-real {
                padding: 16px !important;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="header-box">
            <h1 style='color: #60a5fa; margin-bottom: 4px; font-size: 32px;'>Dual-Stream Deepfake Detector</h1>
            <p style='color: #94a3b8; font-size: 14px; margin-bottom: 0;'>
                Spatial ConvNeXt-Small + SRM/Bayar-Stamm 2D Real FFT Spectral Gated Fusion
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        pytorch_model, cropper, has_pytorch_weights, default_threshold, default_temperature = (
            load_prediction_engine()
        )
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        st.stop()

    # Sidebar Interactive Controls & Metadata Panel
    st.sidebar.markdown("### Control Panel & Settings")
    
    threshold_slider = st.sidebar.slider(
        "Decision Threshold (T*)",
        min_value=0.01,
        max_value=0.99,
        value=float(default_threshold),
        step=0.01,
        help="Adjust classification decision threshold for sensitivity tuning."
    )

    aggregation_select = st.sidebar.selectbox(
        "Temporal Aggregation",
        options=["soft_max", "top_k", "ema", "mean"],
        index=0,
        help="Frame-level score pooling method across video sequences."
    )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("### System Metadata")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <p style="margin:0; font-size:12px; color:#94a3b8;">Primary Backbone</p>
            <p style="margin:0 0 6px 0; font-weight:600; color:#f1f5f9;">ConvNeXt-Small</p>
            <p style="margin:0; font-size:12px; color:#94a3b8;">Frequency Stream</p>
            <p style="margin:0 0 6px 0; font-weight:600; color:#f1f5f9;">SRM + Bayar + 2D FFT</p>
            <p style="margin:0; font-size:12px; color:#94a3b8;">Inference Device</p>
            <p style="margin:0; font-weight:600; color:#38bdf8;">{str(DEVICE).upper()}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None

    # Landing Page Visual Workflow Cards (When no video uploaded)
    if st.session_state.analysis_results is None:
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown(
                """
                <div class="card-workflow">
                    <h4 style="color:#60a5fa; margin-top:0;">1. Video Keyframe Seeking</h4>
                    <p style="color:#94a3b8; font-size:12px; margin:0;">
                        OpenCV hardware-accelerated keyframe extraction with temporal sampling.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with col_w2:
            st.markdown(
                """
                <div class="card-workflow">
                    <h4 style="color:#c084fc; margin-top:0;">2. YuNet Face Alignment</h4>
                    <p style="color:#94a3b8; font-size:12px; margin:0;">
                        5-point landmark similarity transformation with 1.50x scale expansion.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        with col_w3:
            st.markdown(
                """
                <div class="card-workflow">
                    <h4 style="color:#34d399; margin-top:0;">3. Spectral Gated Fusion</h4>
                    <p style="color:#94a3b8; font-size:12px; margin:0;">
                        Spatial features + SRM/Bayar 2D FFT noise magnitude/phase gating.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Upload Video for Analysis")
    uploaded_file = st.file_uploader("Select MP4, AVI, or MOV video file (Max 50MB)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size exceeds 50MB limit.")
            st.stop()

        _hasher = hashlib.md5()
        _hasher.update(uploaded_file.name.encode())
        _hasher.update(str(uploaded_file.size).encode())
        uploaded_file.seek(0)
        _hasher.update(uploaded_file.read(65536))
        uploaded_file.seek(0)
        file_id = _hasher.hexdigest()

        if st.session_state.last_file_id != file_id or st.session_state.analysis_results is None:
            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    shutil.copyfileobj(uploaded_file, tmp)
                    tmp_path = tmp.name

                with st.spinner("Running Dual-Stream inference & spectral analysis..."):
                    res = process_video_frames(
                        video_path=tmp_path,
                        pytorch_model=pytorch_model,
                        cropper=cropper,
                        classification_threshold=threshold_slider,
                        temperature=default_temperature,
                        has_pytorch_weights=has_pytorch_weights,
                        aggregation_method=aggregation_select,
                    )
                    st.session_state.analysis_results = res
                    st.session_state.last_file_id = file_id
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass

        res = st.session_state.analysis_results

        if res is None:
            st.error("No clear face detections were found in the uploaded video.")
        else:
            # Dynamic threshold recalculation without re-running model forward passes
            raw_video_prob = res["raw_video_prob"]
            all_probs = res["all_probs"]
            sample_faces = res["sample_faces"]
            sample_probs = res["sample_probs"]

            final_label = "Fake" if raw_video_prob > threshold_slider else "Real"
            final_conf = normalize_confidence(raw_video_prob, threshold_slider)

            fake_faces_count = sum(1 for p in all_probs if p > threshold_slider)
            real_faces_count = len(all_probs) - fake_faces_count

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
            
            # Side-by-side layout: Input Video Player + Detection Result Container
            col_video, col_results = st.columns([1, 1])

            with col_video:
                st.markdown("#### Input Video Stream")
                uploaded_file.seek(0)
                st.video(uploaded_file)

            with col_results:
                st.markdown("#### Detection Result")
                is_fake = final_label == "Fake"
                card_class = "result-card-fake" if is_fake else "result-card-real"
                color = "#ef4444" if is_fake else "#22c55e"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <h2 style="color: {color}; margin: 0;">DETECTED: {final_label.upper()}</h2>
                        <h4 style="color: {color}; margin-top: 4px;">Confidence: {final_conf:.1f}%</h4>
                        <p style="color: #94a3b8; font-size: 12px; margin: 4px 0 0 0;">
                            Probability Score: {raw_video_prob:.4f} (Threshold: {threshold_slider:.2f})
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Analyzed Faces", len(all_probs))
                col_m2.metric("Real Faces", real_faces_count)
                col_m3.metric("Fake Faces", fake_faces_count)

            # Face Crop Inspection Grid
            if sample_faces:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### Extracted Face Crop Predictions")
                cols = st.columns(len(sample_faces))
                for col, face_img, prob in zip(cols, sample_faces, sample_probs):
                    label = "Fake" if prob > threshold_slider else "Real"
                    conf = normalize_confidence(prob, threshold_slider)
                    with col:
                        st.image(face_img, use_container_width=True)
                        c_color = "#22c55e" if label == "Real" else "#ef4444"
                        st.markdown(
                            f"<p style='text-align:center; color:{c_color}; font-size: 12px; margin-top:4px;'><b>{label}</b><br>{conf:.1f}%</p>",
                            unsafe_allow_html=True,
                        )

            # On-Demand Selectable Face Diagnostics (Fast Baseline + Selectable Target Crop)
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander("🔬 View 4-Panel Interpretability Diagnostics (On-Demand SRM + FFT + Grad-CAM)", expanded=False):
                if sample_faces:
                    face_options = [f"Face Crop #{i+1} (Prob: {prob:.4f})" for i, prob in enumerate(sample_probs)]
                    selected_idx = st.selectbox("Select Face Crop to Inspect", options=list(range(len(face_options))), format_func=lambda i: face_options[i])
                    
                    if st.button("Generate Interpretability Maps"):
                        unwrapped = pytorch_model.module if isinstance(pytorch_model, torch.nn.DataParallel) else pytorch_model
                        selected_face = sample_faces[selected_idx]

                        with st.spinner("Computing Grad-CAM attention & spectral noise maps..."):
                            diag = generate_face_diagnostics(unwrapped, selected_face, temperature=default_temperature)

                        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                        with d_col1:
                            st.image(diag["original"], caption="(a) RGB Face Crop", use_container_width=True)
                        with d_col2:
                            st.image(diag["srm_residual"], caption="(b) SRM Noise Residual", use_container_width=True)
                        with d_col3:
                            st.image(diag["fft_spectrum"], caption="(c) 2D FFT Magnitude", use_container_width=True)
                        with d_col4:
                            st.image(diag["gradcam_overlay"], caption="(d) Grad-CAM Attention", use_container_width=True)

    # Clickable Footer
    st.markdown("<hr style='margin: 30px 0 15px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #64748b; font-size: 12px;'>
            <p style='margin-bottom: 6px;'>
                Dual-Stream Deepfake Detector • PyTorch 2.x • ConvNeXt-Small + SRM/Bayar 2D FFT Gated Fusion
            </p>
            <p>
                <a href='https://github.com/yyouretoast/deepfake-detection' target='_blank' style='color: #60a5fa; text-decoration: none; margin-right: 12px;'>📦 GitHub Repository</a>
                <a href='https://huggingface.co/spaces/yyouretoast/deepfake-detector' target='_blank' style='color: #60a5fa; text-decoration: none;'>🤗 Hugging Face Space</a>
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_ui()
