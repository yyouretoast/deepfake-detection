import gc
from typing import List, Tuple, Dict, Any, Optional
import os
import tempfile
import cv2
import numpy as np
import torch
import streamlit as st

from src.dataset.preprocess import DynamicFaceCropper
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.explainability.gradcam import PyTorchGradCAM
from src.config import load_config

try:
    from src.models.onnx_exporter import ONNXDeepfakePredictor, HAS_ONNX
except ImportError:
    HAS_ONNX = False

st.set_page_config(
    page_title="Deepfake Detector (PyTorch + ConvNeXt + ONNX)",
    page_icon="🎭",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .header-box {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, rgba(26, 115, 232, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 25px;
    }
    .result-card-fake {
        text-align: center;
        padding: 25px;
        border-radius: 16px;
        background: rgba(229, 57, 53, 0.15);
        border: 2px solid #ef4444;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);
    }
    .result-card-real {
        text-align: center;
        padding: 25px;
        border-radius: 16px;
        background: rgba(67, 160, 71, 0.15);
        border: 2px solid #22c55e;
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-box">
        <h1 style='color: #60a5fa; margin-bottom: 5px;'>Deepfake Detection App</h1>
        <p style='color: #94a3b8; font-size: 14px;'>
            PyTorch 2.x / ONNX Runtime • ConvNeXt-Base + 2D FFT Frequency Stream • GroupKFold Verified
        </p>
    </div>
""", unsafe_allow_html=True)

CONFIG = load_config()
APP_CFG = CONFIG.get("app", {})
IMG_SIZE: int = CONFIG.get("preprocessing", {}).get("img_size", 512)
FRAMES_TO_SAMPLE: int = APP_CFG.get("frames_to_sample", 10)
DEFAULT_THRESHOLD: float = APP_CFG.get("classification_threshold", 0.5)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models() -> Tuple[torch.nn.Module, Optional[Any], bool, float]:
    onnx_path = "models/deepfake_convnext_v2.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = "deepfake_convnext_v2.onnx"

    onnx_predictor: Optional[Any] = None
    if HAS_ONNX and os.path.exists(onnx_path):
        try:
            onnx_predictor = ONNXDeepfakePredictor(onnx_path)
            st.toast("ONNX Runtime initialized.")
        except Exception:
            onnx_predictor = None

    backbone_name = CONFIG.get("model", {}).get("backbone", "convnext_base")
    pytorch_model = HybridDeepfakeDetector(backbone_name=backbone_name, pretrained=False, use_fft_branch=True, config=CONFIG)
    
    weights_path = "models/deepfake_convnext_v2.pt"
    if not os.path.exists(weights_path):
        weights_path = "deepfake_convnext_v2.pth"

    opt_threshold = DEFAULT_THRESHOLD
    has_weights = os.path.exists(weights_path)
    
    if has_weights:
        # weights_only=False required: checkpoint contains config dict alongside state_dict.
        # Ensure this file is loaded only from a trusted source.
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            opt_thresh_val = checkpoint.get("optimal_threshold", None)
            if opt_thresh_val is not None:
                opt_threshold = float(opt_thresh_val)
        else:
            state_dict = checkpoint
        pytorch_model.load_state_dict(state_dict)

    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    return pytorch_model, onnx_predictor, has_weights, opt_threshold

try:
    pytorch_model, onnx_predictor, has_pytorch_weights, CLASSIFICATION_THRESHOLD = load_models()
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=IMG_SIZE, device=DEVICE)
    if not has_pytorch_weights:
        st.error("🚨 **Critical Error: Trained model weights not found!** (`models/deepfake_convnext_v2.pt`). Cannot run inference with a randomly initialized model.")
        st.stop()
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()

def preprocess_tensors_batch(faces_rgb_list: List[np.ndarray]) -> Tuple[np.ndarray, torch.Tensor]:
    """Returns normalized numpy batch [B, 3, 256, 256] and PyTorch tensor batch."""
    batch_arr = np.stack(faces_rgb_list)
    batch_nchw = batch_arr.transpose(0, 3, 1, 2)
    
    tensor = torch.from_numpy(batch_nchw).float().to(DEVICE) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    
    norm_nchw = tensor.cpu().numpy()
    return norm_nchw, tensor

def predict_video_sequence(video_path: str, enable_gradcam: bool = False) -> Optional[Dict[str, Any]]:
    """
    Video inference engine with direct OpenCV frame seeking, AMP autocast, and batched Grad-CAM.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None

        frame_indices = np.linspace(0, total - 1, min(FRAMES_TO_SAMPLE, total), dtype=int)
        frames_rgb: List[np.ndarray] = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
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

    BATCH_SIZE = 16
    all_probs = []
    all_tensors = []
    
    for i in range(0, len(all_faces), BATCH_SIZE):
        batch_faces = all_faces[i:i+BATCH_SIZE]
        numpy_batch, torch_batch = preprocess_tensors_batch(batch_faces)
        
        batch_probs = None
        if onnx_predictor is not None:
            try:
                batch_probs = onnx_predictor.predict_batch(numpy_batch).tolist()
            except Exception:
                batch_probs = None

        if batch_probs is None:
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    p1 = torch.sigmoid(pytorch_model(torch_batch).float())
                    p2 = torch.sigmoid(pytorch_model(torch.flip(torch_batch, dims=[-1])).float())
                    batch_probs = ((p1 + p2) / 2.0).cpu().numpy().tolist()

        all_probs.extend(batch_probs)
        all_tensors.append(torch_batch)
        
    all_tensors_concat = torch.cat(all_tensors, dim=0)

    zipped_data = list(zip(all_faces, all_probs, all_tensors_concat))
    zipped_data.sort(key=lambda x: x[1], reverse=True)
    
    top_4 = zipped_data[:4]
    sample_faces = [item[0] for item in top_4]
    sample_probs = [item[1] for item in top_4]
    sample_tensors = torch.stack([item[2] for item in top_4])
    
    can_render_gradcam = enable_gradcam and has_pytorch_weights
    sample_heatmaps = []

    if can_render_gradcam:
        try:
            with PyTorchGradCAM(pytorch_model) as gradcam_engine:
                sample_heatmaps = gradcam_engine.generate_heatmaps_batch(sample_tensors)
        except Exception:
            sample_heatmaps = [None] * len(sample_faces)

    def normalize_confidence(prob: float, threshold: float) -> float:
        if prob > threshold:
            return 50.0 + 50.0 * ((prob - threshold) / (1.0 - threshold)) if threshold < 1.0 else 100.0
        else:
            return 50.0 + 50.0 * ((threshold - prob) / threshold) if threshold > 0.0 else 100.0

    sample_outputs = []
    for idx, (face, prob) in enumerate(zip(sample_faces, sample_probs)):
        label = "Fake" if prob > CLASSIFICATION_THRESHOLD else "Real"
        conf = normalize_confidence(prob, CLASSIFICATION_THRESHOLD)
        
        heatmap = sample_heatmaps[idx] if idx < len(sample_heatmaps) else None
        overlay_img = PyTorchGradCAM.overlay_heatmap(face, heatmap) if heatmap is not None else face
        sample_outputs.append((overlay_img, label, conf, prob))

    avg_prob = float(np.mean(all_probs))
    final_label = "Fake" if avg_prob > CLASSIFICATION_THRESHOLD else "Real"
    final_conf = normalize_confidence(avg_prob, CLASSIFICATION_THRESHOLD)
    
    fake_faces_count = sum(1 for p in all_probs if p > CLASSIFICATION_THRESHOLD)
    real_faces_count = len(all_probs) - fake_faces_count

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "final_label": final_label,
        "final_conf": final_conf,
        "real_frames": real_faces_count,
        "fake_frames": fake_faces_count,
        "sample_outputs": sample_outputs,
        "frame_preds": all_probs
    }

st.markdown("### Upload Video File for Analysis")
uploaded_file = st.file_uploader(
    "Upload MP4, AVI, or MOV video file (Max 50MB)",
    type=["mp4", "avi", "mov"]
)

enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM Explainability Heatmaps", value=True)
if enable_gradcam and not has_pytorch_weights:
    st.sidebar.warning("Trained PyTorch weights missing (`models/deepfake_convnext_v2.pt`). Heatmaps disabled.")

st.sidebar.markdown(f"""
---
### System Configuration
- **Engine**: {'ONNX Runtime' if onnx_predictor else 'PyTorch Native'}
- **Model**: {CONFIG.get('model', {}).get('backbone', 'ConvNeXt-Base').title()} + 2D FFT Frequency Stream
- **Padding**: Relative 1.30x Scale Expansion
- **Threshold (T*)**: {CLASSIFICATION_THRESHOLD:.4f}
""")

if uploaded_file:
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("File size exceeds 50MB limit.")
        st.stop()

    content = uploaded_file.read()
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        st.video(content)

        with st.spinner("Running model inference..."):
            res = predict_video_sequence(tmp_path, enable_gradcam=enable_gradcam)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass

    if res is None:
        st.error("No clear face detections were found in the uploaded video.")
    else:
        st.markdown("<hr>", unsafe_allow_html=True)
        is_fake = res["final_label"] == "Fake"
        card_class = "result-card-fake" if is_fake else "result-card-real"
        color = "#ef4444" if is_fake else "#22c55e"

        st.markdown(f"""
            <div class="{card_class}">
                <h1 style="color: {color}; margin: 0;">DETECTED: {res['final_label'].upper()}</h1>
                <h3 style="color: {color}; margin-top: 5px;">Confidence: {res['final_conf']:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Analyzed Faces", res["real_frames"] + res["fake_frames"])
        col2.metric("Real Faces", res["real_frames"])
        col3.metric("Fake Faces", res["fake_frames"])

        st.markdown("### Overall Confidence")
        st.progress(min(int(res["final_conf"]), 100))

        if res["sample_outputs"]:
            st.markdown("### Sample Face Crop Predictions & Grad-CAM Analysis")
            cols = st.columns(len(res["sample_outputs"]))
            for col, (face_img, label, conf, prob) in zip(cols, res["sample_outputs"]):
                with col:
                    st.image(face_img, width=150)
                    c_color = "#22c55e" if label == "Real" else "#ef4444"
                    st.markdown(
                        f"<p style='text-align:center; color:{c_color}; font-size: 13px;'><b>{label}</b><br>{conf:.1f}%</p>",
                        unsafe_allow_html=True
                    )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center; color: #64748b; font-size: 12px;'>
            PyTorch 2.x & ONNX Runtime • Dual-Stream ConvNeXt + 2D FFT Spectrum Extractor • FaceForensics++ Verified
            </p>
        """, unsafe_allow_html=True)
