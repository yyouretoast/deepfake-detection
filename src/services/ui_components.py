"""Modular UI components and rendering helpers for the Streamlit deepfake forensics dashboard."""

import json
from typing import Any
import streamlit as st
from src.utils.visualization import render_temporal_anomaly_timeline


def render_header_and_controls(
    device_name: str,
    default_temp: float = 1.4788,
    default_thresh: float = 0.50,
) -> tuple[float, float, int, str]:
    """Renders dashboard header, device info, and sidebar parameters, returning user control values."""
    st.markdown(
        """
        <div style="padding: 1.2rem; border-radius: 12px; background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;">
                Dual-Stream Forensics Engine
            </h1>
            <p style="margin: 0.4rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Cross-Domain Spatial-Frequency Discrepancy Detection & Interpretability
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Runtime Controls")
        st.info(f"Compute Backend: **{device_name}**")

        temperature = st.slider(
            "Temperature Scale (T*)",
            min_value=0.5,
            max_value=3.0,
            value=float(default_temp),
            step=0.01,
            help="Post-hoc temperature scaling parameter calibrated via NLL minimization on validation split.",
        )

        threshold = st.slider(
            "Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(default_thresh),
            step=0.01,
            help="Decision boundary for classifying sample as manipulated.",
        )

        num_frames = st.slider(
            "Keyframes Sampled (N)",
            min_value=4,
            max_value=64,
            value=16,
            step=4,
            help="Number of uniform keyframes extracted from video.",
        )

        agg_method = st.selectbox(
            "Temporal Aggregation",
            options=["Top-K (k=3)", "SoftMax-Weighted", "Exponential Moving Average", "Arithmetic Mean"],
            index=0,
            help="Mathematical operator used to pool frame-level anomaly probabilities into video verdict.",
        )

    return temperature, threshold, num_frames, agg_method


def render_diagnostic_quad(diag: dict[str, Any], title_prefix: str = "Diagnostic Quad") -> None:
    """Renders 4-panel visual interpretability representations (RGB, SRM Residual, 2D FFT, Grad-CAM)."""
    st.markdown(f"#### {title_prefix}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(diag["original"], caption="RGB Face Crop", use_container_width=True)
    with c2:
        st.image(diag["srm_residual"], caption="SRM High-Pass Residual", use_container_width=True)
    with c3:
        st.image(diag["fft_spectrum"], caption="2D FFT Magnitude Spectrum", use_container_width=True)
    with c4:
        st.image(diag["gradcam_overlay"], caption="Spatial Grad-CAM Overlay", use_container_width=True)


def render_video_analysis_card(
    verdict: str,
    confidence: float,
    mean_score: float,
    peak_score: float,
    duration_s: float,
    n_frames: int,
    report_dict: dict[str, Any],
) -> None:
    """Renders verdict metrics banner and forensic JSON download button."""
    color = "#ef4444" if verdict == "MANIPULATED / FAKE" else "#10b981"
    bg = "rgba(239, 68, 68, 0.1)" if verdict == "MANIPULATED / FAKE" else "rgba(16, 185, 129, 0.1)"

    st.markdown(
        f"""
        <div style="padding: 1.5rem; border-radius: 12px; background: {bg}; border: 1px solid {color}; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: {color}; font-weight: 700;">
                        Analysis Verdict
                    </span>
                    <h2 style="margin: 0.2rem 0; color: {color}; font-size: 1.8rem; font-weight: 800;">
                        {verdict}
                    </h2>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.85rem; color: #94a3b8;">Calibrated Confidence</span>
                    <h2 style="margin: 0.2rem 0; font-size: 1.8rem; font-weight: 800;">
                        {confidence:.1f}%
                    </h2>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Anomaly Score", f"{mean_score:.3f}")
    m2.metric("Peak Anomaly Score", f"{peak_score:.3f}")
    m3.metric("Video Duration", f"{duration_s:.1f}s")
    m4.metric("Valid Faces Processed", f"{n_frames}")

    report_json = json.dumps(report_dict, indent=2)
    st.download_button(
        label="Download Complete Forensic JSON Dossier",
        data=report_json,
        file_name="deepfake_forensic_report.json",
        mime="application/json",
    )


def render_anomaly_timeline(
    scores: list[float],
    timestamps: list[float],
    threshold: float,
) -> None:
    """Renders Matplotlib glassmorphic timeline figure."""
    fig = render_temporal_anomaly_timeline(timestamps, scores, threshold=threshold)
    st.pyplot(fig)
