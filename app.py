"""
Streamlit Web Interface for Dual-Stream Deepfake Detector Engine.
Decoupled frontend orchestrator rendering dark glassmorphism UI, video player,
temporal anomaly timeline, interactive frame scrubbing, and 4-panel diagnostics.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hashlib
import tempfile
import shutil
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import streamlit as st

from src.services.video_engine import load_prediction_engine, process_video_frames, DEVICE
from src.utils.interpretability import generate_face_diagnostics
from src.utils.visualization import render_temporal_anomaly_timeline
from src.utils.checkpoint import clean_state_dict, normalize_confidence  # noqa: F401
from src.dataset.preprocess import preprocess_tensors_batch  # noqa: F401


# ---------------------------------------------------------------------------
# Streamlit UI Rendering
# ---------------------------------------------------------------------------

def render_ui() -> None:
    """Renders the Streamlit frontend layout and handles user interactions."""
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
        @st.cache_resource
        def cached_model_loader():
            return load_prediction_engine()

        pytorch_model, cropper, has_pytorch_weights, default_threshold, default_temperature = (
            cached_model_loader()
        )
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        st.stop()

    # Sidebar Controls & System Panel
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

    # Landing Workflow Cards
    if st.session_state.analysis_results is None:
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.markdown(
                """
                <div class="card-workflow">
                    <h4 style="color:#60a5fa; margin-top:0;">1. Keyframe Seeking</h4>
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
            raw_video_prob = res["raw_video_prob"]
            all_probs = res["all_probs"]
            sample_faces = res["sample_faces"]
            sample_probs = res["sample_probs"]

            final_label = "Fake" if raw_video_prob > threshold_slider else "Real"
            final_conf = normalize_confidence(raw_video_prob, threshold_slider)

            fake_faces_count = sum(1 for p in all_probs if p > threshold_slider)
            real_faces_count = len(all_probs) - fake_faces_count

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            # Side-by-side Video Player + Detection Metrics
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

            # Temporal Video Anomaly Timeline & Interactive Frame Scrubbing
            timestamps = res.get("timestamps", [])
            frame_indices = res.get("frame_indices", [])
            all_faces = res.get("all_faces", [])

            if timestamps and all_probs:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### 📈 Temporal Video Anomaly Timeline")

                fig = render_temporal_anomaly_timeline(timestamps, all_probs, threshold_slider)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.markdown("#### 🔍 Interactive Timestamp / Frame Scrubbing")
                scrub_options = [
                    f"Timestamp {t:.2f}s — Frame #{f_idx} (Prob: {p:.4f} - {'FAKE' if p > threshold_slider else 'REAL'})"
                    for t, f_idx, p in zip(timestamps, frame_indices, all_probs)
                ]
                scrub_idx = st.selectbox(
                    "Select Timestamp to Inspect & Analyze",
                    options=list(range(len(scrub_options))),
                    format_func=lambda idx: scrub_options[idx],
                    key="timeline_scrub_select"
                )

                if scrub_idx < len(all_faces):
                    s_face = all_faces[scrub_idx]
                    s_prob = all_probs[scrub_idx]
                    s_time = timestamps[scrub_idx]
                    s_frame = frame_indices[scrub_idx]

                    scrub_col1, scrub_col2 = st.columns([1, 2])
                    with scrub_col1:
                        st.image(s_face, caption=f"Scrubbed Face Crop at {s_time:.2f}s (Frame #{s_frame})", use_container_width=True)

                    with scrub_col2:
                        s_label = "Fake" if s_prob > threshold_slider else "Real"
                        s_color = "#ef4444" if s_label == "Fake" else "#22c55e"
                        s_conf = normalize_confidence(s_prob, threshold_slider)
                        st.markdown(
                            f"""
                            <div style="background: rgba(30, 41, 59, 0.5); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                                <h4 style="margin:0; color:{s_color};">Status: {s_label.upper()} ({s_conf:.1f}% Confidence)</h4>
                                <p style="color:#94a3b8; font-size:13px; margin: 6px 0 0 0;">
                                    Timestamp: <b>{s_time:.2f}s</b> | Frame Index: <b>#{s_frame}</b> | Raw Probability: <b>{s_prob:.4f}</b>
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔬 Generate 4-Panel Diagnostics for Selected Timestamp", key="btn_scrub_diag"):
                            unwrapped = pytorch_model.module if isinstance(pytorch_model, torch.nn.DataParallel) else pytorch_model
                            with st.spinner(f"Computing Grad-CAM attention & spectral noise maps for timestamp {s_time:.2f}s..."):
                                diag = generate_face_diagnostics(unwrapped, s_face, device=DEVICE, temperature=default_temperature)

                            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                            with d_col1:
                                st.image(diag["original"], caption="(a) RGB Face Crop", use_container_width=True)
                            with d_col2:
                                st.image(diag["srm_residual"], caption="(b) SRM Noise Residual", use_container_width=True)
                            with d_col3:
                                st.image(diag["fft_spectrum"], caption="(c) 2D FFT Magnitude", use_container_width=True)
                            with d_col4:
                                st.image(diag["gradcam_overlay"], caption="(d) Grad-CAM Attention", use_container_width=True)

            # Top Anomaly Face Crop Grid
            if sample_faces:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### Top Anomaly Face Crops")
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

            # On-Demand Selectable Diagnostics
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander("🔬 View 4-Panel Interpretability Diagnostics (Selectable Target Crop)", expanded=False):
                if sample_faces:
                    face_options = [f"Face Crop #{i+1} (Prob: {prob:.4f})" for i, prob in enumerate(sample_probs)]
                    selected_idx = st.selectbox("Select Face Crop to Inspect", options=list(range(len(face_options))), format_func=lambda i: face_options[i])

                    if st.button("Generate Interpretability Maps", key="btn_crop_diag"):
                        unwrapped = pytorch_model.module if isinstance(pytorch_model, torch.nn.DataParallel) else pytorch_model
                        selected_face = sample_faces[selected_idx]

                        with st.spinner("Computing Grad-CAM attention & spectral noise maps..."):
                            diag = generate_face_diagnostics(unwrapped, selected_face, device=DEVICE, temperature=default_temperature)

                        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                        with d_col1:
                            st.image(diag["original"], caption="(a) RGB Face Crop", use_container_width=True)
                        with d_col2:
                            st.image(diag["srm_residual"], caption="(b) SRM Noise Residual", use_container_width=True)
                        with d_col3:
                            st.image(diag["fft_spectrum"], caption="(c) 2D FFT Magnitude", use_container_width=True)
                        with d_col4:
                            st.image(diag["gradcam_overlay"], caption="(d) Grad-CAM Attention", use_container_width=True)

    # Footer
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
