"""Streamlit Web Interface for Dual-Stream Deepfake Detector Engine."""

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.services.ui_components import render_diagnostic_quad
from src.services.video_engine import (
    DEVICE,
    load_prediction_engine,
    load_temporal_engine,
    process_single_image,
    process_video_frames,
)
from src.utils.checkpoint import normalize_confidence
from src.utils.interpretability import generate_face_diagnostics
from src.utils.visualization import render_temporal_anomaly_timeline

logger = logging.getLogger(__name__)

__all__ = [
    "render_ui",
    "safe_remove_file",
]


def safe_remove_file(file_path: str, max_retries: int = 3, delay: float = 0.5) -> None:
    """Safely attempt to remove a file with retries for file locks on Windows."""
    if not file_path or not os.path.exists(file_path):
        return
    for attempt in range(max_retries):
        try:
            os.unlink(file_path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
        except OSError:
            return


@st.cache_resource
def _cached_model_loader() -> tuple[Any, Any, bool, float, float]:
    """Cache and warm up the primary dual-stream prediction engine."""
    engine = load_prediction_engine()
    model, _cropper, _has_weights, _threshold, _temp = engine
    try:
        dummy_tensor = torch.zeros(1, 3, 256, 256, device=DEVICE)
        with torch.inference_mode():
            model(dummy_tensor)
    except Exception as e:
        logger.debug("Model warm-up pass skipped: %s", e)
    return engine


@st.cache_resource
def _cached_temporal_loader() -> torch.nn.Module | None:
    """Cache the optional Bi-GRU spatiotemporal consistency head."""
    return load_temporal_engine()


def render_ui() -> None:
    """Render the Streamlit frontend layout and handle user interactions."""
    st.set_page_config(
        page_title="Dual-Stream Deepfake Forensics",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Crisp, professional scientific dark theme without decorative AI slop
    st.markdown(
        """
        <style>
        .main {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .header-box {
            padding: 22px 28px;
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .telemetry-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px 18px;
            text-align: center;
        }
        .result-card-fake {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: rgba(239, 68, 68, 0.08);
            border: 1.5px solid #ef4444;
        }
        .result-card-real {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: rgba(34, 197, 94, 0.08);
            border: 1.5px solid #22c55e;
        }
        .result-card-ambiguous {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: rgba(245, 158, 11, 0.08);
            border: 1.5px solid #f59e0b;
        }
        .sidebar-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 20px !important;
            font-weight: 700;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    try:
        engine = _cached_model_loader()
        pytorch_model, cropper, has_pytorch_weights, default_threshold, default_temperature = engine
        tau_real = getattr(engine, "tau_real", 0.40)
        tau_fake = getattr(engine, "tau_fake", 0.60)
        temporal_model = _cached_temporal_loader()
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        st.stop()

    temporal_status = "Bi-GRU Head: Active" if temporal_model is not None else "Bi-GRU Head: Inactive"
    temporal_color = "#34d399" if temporal_model is not None else "#94a3b8"

    st.markdown(
        f"""
        <div class="header-box">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 style='color: #f8fafc; margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.3px;'>
                        Dual-Stream Deepfake Forensics Dashboard
                    </h1>
                    <p style='color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;'>
                        Spatial ConvNeXt-Small + SRM/Bayar 2D Real FFT Spectral Gated Residual Fusion
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span style="background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        DEVICE: {str(DEVICE).upper()}
                    </span>
                    <span style="background: #1e293b; color: {temporal_color}; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                        {temporal_status}
                    </span>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.markdown("### Analysis Parameters")

    threshold_slider = st.sidebar.slider(
        "Decision Threshold (τ*)",
        min_value=0.01,
        max_value=0.99,
        value=float(default_threshold),
        step=0.01,
        help="Operating decision threshold calibrated via validation PR-curve (Youden's J).",
    )

    n_frames_slider = st.sidebar.slider(
        "Sampled Keyframes (N)",
        min_value=4,
        max_value=20,
        value=10,
        step=2,
        help="Number of uniformly sampled keyframes across video temporal span.",
    )

    aggregation_select = st.sidebar.selectbox(
        "Temporal Aggregation Policy",
        options=["soft_max", "top_k", "ema", "mean"],
        index=0,
        help="Sequence pooling method when Bi-GRU temporal sequence model is inactive.",
    )

    st.sidebar.markdown("<hr style='margin: 16px 0; border-color: #1e293b;'>", unsafe_allow_html=True)
    st.sidebar.markdown("### Architecture Telemetry")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <p style="margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px;">Spatial Stream</p>
            <p style="margin:0 0 8px 0; font-weight:600; color:#f1f5f9; font-size:13px;">ConvNeXt-Small (512-d)</p>
            <p style="margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px;">Frequency Stream</p>
            <p style="margin:0 0 8px 0; font-weight:600; color:#f1f5f9; font-size:13px;">SRM + Bayar + 2D Real FFT</p>
            <p style="margin:0; font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px;">Calibration & Bayesian Bands</p>
            <p style="margin:0; font-weight:600; color:#38bdf8; font-size:13px;">T*={default_temperature:.2f} | τ_real={tau_real:.2f} | τ_fake={tau_fake:.2f}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("ℹ️ Spectral SNR Gating Mechanics", expanded=False):
        st.markdown(
            """
            <div style="font-size:12px; color:#94a3b8; line-height: 1.5;">
                <b>Noise Attenuation (γ)</b>: Frequency noise residuals degrade severely under heavy compression or Gaussian blur.
                Our SNR-adaptive gate dynamically down-weights the spectral stream (γ &rarr; 0) and relies on spatial ConvNeXt features,
                preventing performance degradation cliffs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Intake Mode Selection
    intake_mode = st.radio(
        "Select Forensic Intake Mode",
        options=["📷 Single Photo / Frame", "🎬 Video Sequence"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin: 12px 0 20px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Mode 1: Single Photo Intake
    # ---------------------------------------------------------
    if intake_mode == "📷 Single Photo / Frame":
        st.markdown("### Upload Still Image for Forensic Analysis")
        uploaded_img = st.file_uploader(
            "Select PNG, JPG, JPEG, or WEBP image file (Max 20MB)",
            type=["png", "jpg", "jpeg", "webp"],
            key="single_img_uploader",
        )

        if uploaded_img:
            file_size_mb = uploaded_img.size / (1024 * 1024)
            if uploaded_img.size > 20 * 1024 * 1024:
                st.error(f"Image size ({file_size_mb:.1f} MB) exceeds 20 MB limit.")
                st.stop()

            img_bytes = uploaded_img.getvalue()
            img_hash = hashlib.md5(img_bytes).hexdigest()

            if (
                "img_results" not in st.session_state
                or st.session_state.get("last_img_id") != img_hash
            ):
                with st.spinner("Extracting face, computing dual-stream features, and running SNR gating..."):
                    img_res = process_single_image(
                        image_input=img_bytes,
                        pytorch_model=pytorch_model,
                        cropper=cropper,
                        classification_threshold=threshold_slider,
                        temperature=default_temperature,
                        tau_real=tau_real,
                        tau_fake=tau_fake,
                    )
                    st.session_state.img_results = img_res
                    st.session_state.last_img_id = img_hash

            res = st.session_state.img_results

            if res is None:
                st.error("No facial region could be detected in the uploaded image.")
            else:
                prob = float(res["prob"])
                three_zone = res.get("three_zone", {})
                is_ambiguous = three_zone.get("is_inconclusive", False)
                final_label = "Fake" if prob > threshold_slider else "Real"
                final_conf = normalize_confidence(prob, threshold_slider)

                col_preview, col_verdict = st.columns([1, 1])

                with col_preview:
                    st.markdown("#### Input & Aligned Crop")
                    c_orig, c_crop = st.columns(2)
                    with c_orig:
                        st.image(uploaded_img, caption="Source Image", use_container_width=True)
                    with c_crop:
                        st.image(res["face_crop"], caption="YuNet Aligned Crop (512x512)", use_container_width=True)

                with col_verdict:
                    st.markdown("#### Forensic Verdict")
                    if is_ambiguous:
                        card_class = "result-card-ambiguous"
                        color = "#f59e0b"
                        card_title = "INCONCLUSIVE / AMBIGUITY ZONE"
                    elif final_label == "Fake":
                        card_class = "result-card-fake"
                        color = "#ef4444"
                        card_title = "DETECTED: DEEPFAKE / SYNTHETIC"
                    else:
                        card_class = "result-card-real"
                        color = "#22c55e"
                        card_title = "DETECTED: AUTHENTIC / REAL"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <h2 style="color: {color}; margin: 0; font-size: 22px; font-weight: 700;">{card_title}</h2>
                            <h3 style="color: {color}; margin: 6px 0 0 0; font-size: 28px; font-weight: 800;">{final_conf:.1f}% Confidence</h3>
                            <p style="color: #94a3b8; font-size: 12px; margin: 6px 0 0 0;">
                                Calibrated Prob: <b>{prob:.4f}</b> | Raw Logit: <b>{res['raw_logit']:.3f}</b> | Threshold τ*: <b>{threshold_slider:.2f}</b>
                            </p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if is_ambiguous:
                        st.warning(
                            f"⚠️ **Ambiguity Warning**: Calibrated probability ({prob:.4f}) falls between "
                            f"τ_real ({tau_real:.2f}) and τ_fake ({tau_fake:.2f}). Expert manual review required."
                        )
                    else:
                        zone_label = three_zone.get("verdict", final_label)
                        st.caption(
                            f"🛡️ Bayesian High-Certainty Band: **{zone_label}** "
                            f"(τ_real={tau_real:.2f}, τ_fake={tau_fake:.2f})"
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Dual-Stream Telemetry")
                    col_t1, col_t2, col_t3 = st.columns(3)
                    spectral_pct = res["spectral_gate"] * 100.0
                    spatial_pct = res["spatial_gate"] * 100.0
                    col_t1.metric("Spectral Gate (g)", f"{spectral_pct:.1f}%", f"{spatial_pct:.1f}% Spatial")
                    col_t2.metric("SNR Attenuator (γ)", f"{res['snr_attenuator']:.2f}")
                    col_t3.metric("Laplacian Var (σ²)", f"{res['laplacian_var']:.1f}")

                # 4-Panel Interpretability Quad
                st.markdown("<hr style='margin: 24px 0 16px 0; border-color: #1e293b;'>", unsafe_allow_html=True)
                render_diagnostic_quad(res["diagnostics"], title_prefix="Interpretability Diagnostic Quad")

                # Expandable guide
                with st.expander("📖 How to Read the 4-Panel Diagnostic Quad", expanded=False):
                    st.markdown(
                        """
                        - **Panel A (RGB Face Crop)**: Normalized facial bounding box cropped via YuNet 5-point landmark similarity transform.
                        - **Panel B (SRM Residual)**: Spatial Rich Model high-pass convolution highlighting suppression of camera sensor PRNU noise.
                        - **Panel C (2D FFT Magnitude Spectrum)**: Log-magnitude 2D Fourier transform centered DC component. Synthetic generators (GANs/Diffusion) exhibit distinct regular periodic grid artifacts or high-frequency azimuthal decay (Durall et al., Frank et al.).
                        - **Panel D (Grad-CAM Overlay)**: Gradient-weighted class activation map exposing the spatial regions of ConvNeXt driving the classification decision.
                        """
                    )

                # Report Download
                report_data = {
                    "image_filename": uploaded_img.name,
                    "file_md5_hash": img_hash,
                    "verdict": final_label.upper(),
                    "calibrated_confidence_pct": round(float(final_conf), 2),
                    "calibrated_probability": round(float(prob), 6),
                    "raw_logit": round(float(res["raw_logit"]), 6),
                    "decision_threshold_tau_star": float(threshold_slider),
                    "bayesian_thresholds": {
                        "tau_real": float(tau_real),
                        "tau_fake": float(tau_fake),
                    },
                    "three_zone_verdict": three_zone.get("verdict", final_label),
                    "is_inconclusive": bool(is_ambiguous),
                    "telemetry": {
                        "spectral_gating_weight": round(float(res["spectral_gate"]), 4),
                        "spatial_gating_weight": round(float(res["spatial_gate"]), 4),
                        "snr_attenuator_gamma": round(float(res["snr_attenuator"]), 4),
                        "noise_power": round(float(res["noise_power"]), 6),
                        "laplacian_noise_variance": round(float(res["laplacian_var"]), 2),
                    },
                }
                st.download_button(
                    label="📥 Download Forensic Inspection Report (.json)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"forensic_report_{img_hash[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # ---------------------------------------------------------
    # Mode 2: Video Sequence Intake
    # ---------------------------------------------------------
    else:
        st.markdown("### Upload Video for Temporal Sequence Analysis")
        uploaded_video = st.file_uploader(
            "Select MP4, AVI, or MOV video file (Max 50MB)",
            type=["mp4", "avi", "mov"],
            key="video_file_uploader",
        )

        if uploaded_video:
            file_size_mb = uploaded_video.size / (1024 * 1024)
            if uploaded_video.size > 50 * 1024 * 1024:
                st.error(f"File size ({file_size_mb:.1f} MB) exceeds 50 MB limit.")
                st.stop()

            st.caption(f"📁 **Uploaded File:** `{uploaded_video.name}` ({file_size_mb:.2f} MB)")

            _hasher = hashlib.md5()
            _hasher.update(uploaded_video.name.encode())
            _hasher.update(str(uploaded_video.size).encode())
            uploaded_video.seek(0)
            _hasher.update(uploaded_video.read(65536))
            uploaded_video.seek(0)
            file_id = _hasher.hexdigest()

            if (
                "video_results" not in st.session_state
                or st.session_state.get("last_video_id") != file_id
            ):
                tmp_path: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        shutil.copyfileobj(uploaded_video, tmp)
                        tmp_path = tmp.name

                    prog_bar = st.progress(10)
                    prog_status = st.empty()
                    prog_status.markdown("*Seeking keyframes, aligning faces, and running dual-stream inference...*")

                    res = process_video_frames(
                        video_path=tmp_path,
                        pytorch_model=pytorch_model,
                        cropper=cropper,
                        classification_threshold=threshold_slider,
                        temperature=default_temperature,
                        has_pytorch_weights=has_pytorch_weights,
                        aggregation_method=aggregation_select,
                        num_frames=n_frames_slider,
                        temporal_model=temporal_model,
                        tau_real=tau_real,
                        tau_fake=tau_fake,
                    )
                    prog_bar.progress(100)
                    prog_status.empty()
                    prog_bar.empty()

                    st.session_state.video_results = res
                    st.session_state.last_video_id = file_id
                finally:
                    if tmp_path:
                        safe_remove_file(tmp_path)

            res = st.session_state.video_results

            if res is None:
                st.error("No clear facial detections were found across the sampled keyframes.")
            else:
                raw_video_prob = float(res["raw_video_prob"])
                all_probs: list[float] = [
                    float(p[0]) if isinstance(p, (list, tuple)) else float(p)
                    for p in res["all_probs"]
                ]
                sample_faces = res["sample_faces"]
                sample_probs: list[float] = [
                    float(p[0]) if isinstance(p, (list, tuple)) else float(p)
                    for p in res["sample_probs"]
                ]

                final_label = "Fake" if raw_video_prob > threshold_slider else "Real"
                final_conf = normalize_confidence(raw_video_prob, threshold_slider)

                fake_faces_count = sum(1 for p in all_probs if p > threshold_slider)
                real_faces_count = len(all_probs) - fake_faces_count

                st.markdown("<hr style='margin: 16px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

                col_video, col_results = st.columns([1, 1])

                with col_video:
                    st.markdown("#### Input Video Stream")
                    uploaded_video.seek(0)
                    try:
                        st.video(uploaded_video)
                    except (RuntimeError, TypeError, ValueError, OSError):
                        st.info(
                            "ℹ️ Native browser video preview unavailable for this codec. "
                            "Backend OpenCV detection engine analyzed all keyframes normally."
                        )

                with col_results:
                    st.markdown("#### Sequence Verdict")
                    three_zone = res.get("three_zone", {})
                    is_ambiguous = three_zone.get("is_inconclusive", False)

                    if is_ambiguous:
                        card_class = "result-card-ambiguous"
                        color = "#f59e0b"
                        card_title = "INCONCLUSIVE / AMBIGUITY ZONE"
                    elif final_label == "Fake":
                        card_class = "result-card-fake"
                        color = "#ef4444"
                        card_title = "DETECTED: DEEPFAKE SEQUENCE"
                    else:
                        card_class = "result-card-real"
                        color = "#22c55e"
                        card_title = "DETECTED: AUTHENTIC SEQUENCE"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <h2 style="color: {color}; margin: 0; font-size: 22px; font-weight: 700;">{card_title}</h2>
                            <h3 style="color: {color}; margin: 6px 0 0 0; font-size: 28px; font-weight: 800;">{final_conf:.1f}% Confidence</h3>
                            <p style="color: #94a3b8; font-size: 12px; margin: 6px 0 0 0;">
                                Sequence Anomaly Score: <b>{raw_video_prob:.4f}</b> | Threshold τ*: <b>{threshold_slider:.2f}</b>
                            </p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if is_ambiguous:
                        st.warning(
                            f"⚠️ **Forensic Ambiguity Warning**: Sequence anomaly score ({raw_video_prob:.4f}) falls between "
                            f"τ_real ({tau_real:.2f}) and τ_fake ({tau_fake:.2f}). Expert review recommended."
                        )
                    else:
                        zone_label = three_zone.get("verdict", final_label)
                        st.caption(
                            f"🛡️ Bayesian High-Certainty Band: **{zone_label}** "
                            f"(τ_real={tau_real:.2f}, τ_fake={tau_fake:.2f})"
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Analyzed Faces", len(all_probs))
                    col_m2.metric("Real Faces", real_faces_count)
                    col_m3.metric("Fake Faces", fake_faces_count)

                    report_data = {
                        "video_filename": uploaded_video.name,
                        "file_size_bytes": uploaded_video.size,
                        "file_md5_hash": file_id,
                        "verdict": final_label.upper(),
                        "calibrated_confidence_pct": round(float(final_conf), 2),
                        "aggregated_anomaly_score": round(float(raw_video_prob), 6),
                        "operating_decision_threshold": float(threshold_slider),
                        "optimal_temperature_T_star": float(default_temperature),
                        "bayesian_thresholds": {
                            "tau_real": float(tau_real),
                            "tau_fake": float(tau_fake),
                        },
                        "three_zone_verdict": three_zone.get("verdict", final_label),
                        "is_inconclusive": bool(is_ambiguous),
                        "analyzed_frame_count": len(all_probs),
                        "real_frame_count": real_faces_count,
                        "fake_frame_count": fake_faces_count,
                        "temporal_aggregation_policy": aggregation_select,
                        "temporal_model_active": temporal_model is not None,
                        "timestamps_sec": [round(float(t), 3) for t in res.get("timestamps", [])],
                        "frame_probabilities": [round(float(p), 6) for p in all_probs],
                        "temporal_attention": res.get("temporal_attention"),
                    }
                    st.download_button(
                        label="📥 Download Forensic Inspection Report (.json)",
                        data=json.dumps(report_data, indent=2),
                        file_name=f"forensic_report_{file_id[:8]}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                timestamps = res.get("timestamps", [])
                frame_indices = res.get("frame_indices", [])
                all_faces = res.get("all_faces", [])
                temporal_attn = res.get("temporal_attention")

                if timestamps and all_probs:
                    st.markdown("<hr style='margin: 24px 0 16px 0; border-color: #1e293b;'>", unsafe_allow_html=True)
                    st.markdown("### 📈 Temporal Video Anomaly Timeline")

                    fig = render_temporal_anomaly_timeline(
                        timestamps=timestamps,
                        probs=all_probs,
                        threshold=threshold_slider,
                        attention_weights=temporal_attn,
                    )
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
                        key="timeline_scrub_select",
                    )

                    if scrub_idx < len(all_faces):
                        s_face = all_faces[scrub_idx]
                        s_prob = all_probs[scrub_idx]
                        s_time = timestamps[scrub_idx]
                        s_frame = frame_indices[scrub_idx]

                        scrub_col1, scrub_col2 = st.columns([1, 2])
                        with scrub_col1:
                            st.image(
                                s_face,
                                caption=f"Scrubbed Face Crop at {s_time:.2f}s (Frame #{s_frame})",
                            )

                        with scrub_col2:
                            s_label = "Fake" if s_prob > threshold_slider else "Real"
                            s_color = "#ef4444" if s_label == "Fake" else "#22c55e"
                            s_conf = normalize_confidence(s_prob, threshold_slider)
                            st.markdown(
                                f"""
                                <div style="background: #0f172a; padding: 14px; border-radius: 10px; border: 1px solid #1e293b;">
                                    <h4 style="margin:0; color:{s_color};">Status: {s_label.upper()} ({s_conf:.1f}% Confidence)</h4>
                                    <p style="color:#94a3b8; font-size:13px; margin: 4px 0 0 0;">
                                        Timestamp: <b>{s_time:.2f}s</b> | Frame #{s_frame} | Probability: <b>{s_prob:.4f}</b>
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(
                                "🔬 Generate 4-Panel Diagnostics for Selected Frame",
                                key="btn_scrub_diag",
                            ):
                                unwrapped = (
                                    pytorch_model.module
                                    if isinstance(pytorch_model, torch.nn.DataParallel)
                                    else pytorch_model
                                )
                                with st.spinner(
                                    f"Computing Grad-CAM attention & spectral noise maps for timestamp {s_time:.2f}s..."
                                ):
                                    diag = generate_face_diagnostics(
                                        unwrapped,
                                        s_face,
                                        device=DEVICE,
                                        temperature=default_temperature,
                                    )

                                render_diagnostic_quad(diag, title_prefix="Scrubbed Frame Diagnostics")

                if sample_faces:
                    st.markdown("<hr style='margin: 24px 0 16px 0; border-color: #1e293b;'>", unsafe_allow_html=True)
                    st.markdown("### Top Anomaly Face Crops")
                    cols = st.columns(len(sample_faces))
                    for col, face_img, prob in zip(cols, sample_faces, sample_probs):
                        label = "Fake" if prob > threshold_slider else "Real"
                        conf = normalize_confidence(prob, threshold_slider)
                        with col:
                            st.image(face_img)
                            c_color = "#22c55e" if label == "Real" else "#ef4444"
                            st.markdown(
                                f"<p style='text-align:center; color:{c_color}; font-size: 12px; margin-top:4px;'><b>{label}</b><br>{conf:.1f}%</p>",
                                unsafe_allow_html=True,
                            )

    st.markdown("<hr style='margin: 30px 0 15px 0; border-color: #1e293b;'>", unsafe_allow_html=True)
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
