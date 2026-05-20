import streamlit as st
import cv2, numpy as np
from retinaface import RetinaFace
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download
import tempfile, os

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="centered"
)

st.markdown("""
    <h1 style='text-align: center; color: #1a73e8;'>🎭 Deepfake Detection System</h1>
    <p style='text-align: center; color: gray;'>EfficientNetB0 + RetinaFace | Trained on FaceForensics++</p>
    <hr>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    with st.spinner("Loading model..."):
        model_path = hf_hub_download(
            repo_id="yyouretoast/deepfake-detection",
            filename="deepfake_final_SUBMIT.keras"
        )
        return load_model(model_path)

try:
    model = load_detector()
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

IMG_SIZE = 224
PADDING = 20
FRAMES_TO_SAMPLE = 8

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return None, None, None, None, None

    step = max(total // FRAMES_TO_SAMPLE, 1)
    frame_preds = []
    sample_frames = []

    for i in range(FRAMES_TO_SAMPLE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = RetinaFace.detect_faces(rgb)
        except:
            continue

        if not isinstance(faces, dict) or len(faces) == 0:
            continue

        best_face = max(faces.keys(), key=lambda k: (
            faces[k]["facial_area"][2] - faces[k]["facial_area"][0]) *
            (faces[k]["facial_area"][3] - faces[k]["facial_area"][1])
        )
        x1, y1, x2, y2 = faces[best_face]["facial_area"]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(rgb.shape[1], x2 + PADDING)
        y2 = min(rgb.shape[0], y2 + PADDING)
        face = rgb[y1:y2, x1:x2]

        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            continue

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        img_array = preprocess_input(
            np.expand_dims(face_resized.astype("float32"), axis=0)
        )
        pred = model.predict(img_array, verbose=0)[0][0]
        frame_preds.append(pred)

        if len(sample_frames) < 4:
            label = "Real" if pred > 0.5 else "Fake"
            conf = pred * 100 if pred > 0.5 else (1 - pred) * 100
            sample_frames.append((face_resized, label, conf))

    cap.release()

    if not frame_preds:
        return None, None, None, None, None

    avg_pred = np.mean(frame_preds)
    final_label = "Real" if avg_pred > 0.5 else "Fake"
    final_conf = avg_pred * 100 if avg_pred > 0.5 else (1 - avg_pred) * 100
    fake_frames = sum(1 for p in frame_preds if p <= 0.5)
    real_frames = len(frame_preds) - fake_frames

    return final_label, final_conf, real_frames, fake_frames, sample_frames

# Upload
st.markdown("### 📤 Upload a Video")
uploaded = st.file_uploader(
    "Choose a video file (max 50MB)",
    type=["mp4", "avi", "mov"]
)

if uploaded:
    if uploaded.size > 50 * 1024 * 1024:
        st.error("❌ File too large — please upload a video under 50MB")
        st.stop()

    content = uploaded.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    st.video(content)

    with st.spinner("🔍 Analyzing video..."):
        final_label, final_conf, real_frames, fake_frames, sample_frames = predict_video(tmp_path)

    os.unlink(tmp_path)

    if final_label is None:
        st.error("❌ No faces detected in video")
    else:
        st.markdown("<hr>", unsafe_allow_html=True)

        color = "#e53935" if final_label == "Fake" else "#43a047"
        icon = "🚨" if final_label == "Fake" else "✅"
        st.markdown(f"""
            <div style="text-align: center; padding: 30px; border-radius: 15px; background: {color}20; border: 2px solid {color};">
                <h1 style="color: {color};">{icon} {final_label.upper()}</h1>
                <h3 style="color: {color};">Confidence: {final_conf:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Frames Analyzed", real_frames + fake_frames)
        col2.metric("Real Frames", real_frames)
        col3.metric("Fake Frames", fake_frames)

        st.markdown("### 📊 Confidence Score")
        st.progress(min(int(final_conf), 100))

        if sample_frames:
            st.markdown("### 🖼️ Sample Frame Predictions")
            cols = st.columns(len(sample_frames))
            for col, (face_img, label, conf) in zip(cols, sample_frames):
                with col:
                    st.image(face_img, width=150)
                    c = "green" if label == "Real" else "red"
                    st.markdown(
                        f"<p style='text-align:center; color:{c};'><b>{label}</b><br>{conf:.1f}%</p>",
                        unsafe_allow_html=True
                    )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <p style='text-align: center; color: gray; font-size: 12px;'>
            Powered by EfficientNetB0 + RetinaFace | Trained on FaceForensics++ | 90.4% Accuracy
            </p>
        """, unsafe_allow_html=True)
