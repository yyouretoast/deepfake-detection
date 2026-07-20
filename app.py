from typing import List, Tuple, Dict, Any, Optional
import os
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st

from src.dataset.preprocess import DynamicFaceCropper
from src.models.hybrid_detector import HybridDeepfakeDetector
from src.explainability.gradcam import PyTorchGradCAM, overlay_cam

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
        <h1 style='color: #60a5fa; margin-bottom: 5px;'>Deepfake Detection Engine</h1>
        <p style='color: #94a3b8; font-size: 14px;'>
            PyTorch 2.x / ONNX Runtime • ConvNeXt-Small + 2D FFT Frequency Stream • GroupKFold Verified
        </p>
    </div>
""", unsafe_allow_html=True)

IMG_SIZE: int = 224
FRAMES_TO_SAMPLE: int = 10
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models() -> Tuple[torch.nn.Module, Optional[Any], bool]:
    onnx_path = "deepfake_convnext_v2.onnx"
    onnx_predictor: Optional[Any] = None
    if HAS_ONNX and os.path.exists(onnx_path):
        try:
            onnx_predictor = ONNXDeepfakePredictor(onnx_path)
            st.toast("ONNX Runtime Acceleration Engine initialized successfully.")
        except Exception:
            onnx_predictor = None

    pytorch_model = HybridDeepfakeDetector(backbone_name="convnext_small", pretrained=False, use_fft_branch=True)
    weights_path = "deepfake_convnext_v2.pth"
    has_weights = os.path.exists(weights_path)
    if has_weights:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        pytorch_model.load_state_dict(state_dict)
    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    return pytorch_model, onnx_predictor, has_weights

try:
    pytorch_model, onnx_predictor, has_pytorch_weights = load_models()
    cropper = DynamicFaceCropper(scale_factor=1.30, target_size=IMG_SIZE, device=DEVICE)
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()

def preprocess_tensors_batch(faces_rgb_list: List[np.ndarray]) -> Tuple[np.ndarray, torch.Tensor]:
    """Returns normalized numpy batch [B, 3, 224, 224] and PyTorch tensor batch."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    
    batch_arr = np.stack(faces_rgb_list).astype(np.float32) / 255.0
    norm = (batch_arr - mean) / std
    norm_nchw = norm.transpose(0, 3, 1, 2).astype(np.float32)
    tensor = torch.from_numpy(norm_nchw).float().to(DEVICE)
    return norm_nchw, tensor

def predict_video_sequence(video_path: str, enable_gradcam: bool = False) -> Optional[Dict[str, Any]]:
    """
    Video inference engine with PyTorch AMP autocast and batched Grad-CAM.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None

        step = max(total // FRAMES_TO_SAMPLE, 1)
        frames_rgb: List[np.ndarray] = []

        curr_frame = 0
        target_frames = set(i * step for i in range(FRAMES_TO_SAMPLE))

        while cap.isOpened() and len(frames_rgb) < FRAMES_TO_SAMPLE and curr_frame <= max(target_frames):
            if curr_frame in target_frames:
                ret, frame = cap.read()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_rgb.append(rgb)
            else:
                cap.grab()
            curr_frame += 1
    finally:
        cap.release()

    if not frames_rgb:
        return None

    faces = cropper.crop_faces_batched(frames_rgb)
    if not faces:
        return None

    numpy_batch, torch_batch = preprocess_tensors_batch(faces)

    # PyTorch AMP Autocast / ONNX Inference
    if onnx_predictor is not None:
        probs = onnx_predictor.predict_batch(numpy_batch).tolist()
    else:
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                logits = pytorch_model(torch_batch)
                probs = torch.sigmoid(logits.float()).cpu().numpy().tolist()

    if isinstance(probs, float):
        probs = [probs]

    frame_preds = probs
    sample_outputs = []
    
    can_render_gradcam = enable_gradcam and has_pytorch_weights

    # Batched Grad-CAM Generation for sample frames
    sample_faces = faces[:4]
    sample_probs = probs[:4]
    sample_heatmaps = []

    if can_render_gradcam:
        try:
            with PyTorchGradCAM(pytorch_model) as gradcam_engine:
                sample_tensors = torch_batch[:len(sample_faces)]
                target_classes = [1 if p > 0.5 else 0 for p in sample_probs]
                sample_heatmaps = gradcam_engine.generate_heatmaps_batch(sample_tensors, target_classes=target_classes)
        except Exception:
            sample_heatmaps = [None] * len(sample_faces)

    for idx, (face, prob) in enumerate(zip(sample_faces, sample_probs)):
        label = "Real" if prob > 0.5 else "Fake"
        conf = prob * 100 if prob > 0.5 else (1 - prob) * 100
        
        heatmap = sample_heatmaps[idx] if idx < len(sample_heatmaps) else None
        overlay_img = overlay_cam(face, heatmap) if heatmap is not None else face
        sample_outputs.append((overlay_img, label, conf, prob))

    avg_prob = float(np.mean(frame_preds))
    final_label = "Real" if avg_prob > 0.5 else "Fake"
    final_conf = avg_prob * 100 if avg_prob > 0.5 else (1 - avg_prob) * 100
    fake_frames = sum(1 for p in frame_preds if p <= 0.5)
    real_frames = len(frame_preds) - fake_frames

    return {
        "final_label": final_label,
        "final_conf": final_conf,
        "real_frames": real_frames,
        "fake_frames": fake_frames,
        "sample_outputs": sample_outputs,
        "frame_preds": frame_preds
    }

st.markdown("### Upload Video File for Analysis")
uploaded_file = st.file_uploader(
    "Upload MP4, AVI, or MOV video file (Max 50MB)",
    type=["mp4", "avi", "mov"]
)

enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM Explainability Heatmaps", value=True)
if enable_gradcam and not has_pytorch_weights:
    st.sidebar.warning("Trained PyTorch weights missing (`deepfake_convnext_v2.pth`). Heatmaps disabled.")

st.sidebar.markdown(f"""
---
### System Configuration
- **Engine**: {'ONNX Runtime (Accelerated)' if onnx_predictor else 'PyTorch Native (AMP Autocast)'}
- **Model**: ConvNeXt-Small + 2D FFT Frequency Stream
- **Padding**: Relative 1.30x Scale Expansion
- **Validation**: Video-ID GroupKFold Split
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

        with st.spinner("Analyzing video frames with Spatial & Frequency Neural Engine..."):
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
        col1.metric("Analyzed Frames", res["real_frames"] + res["fake_frames"])
        col2.metric("Real Frames", res["real_frames"])
        col3.metric("Fake Frames", res["fake_frames"])

        st.markdown("### Frame-by-Frame Confidence Timeline")
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
