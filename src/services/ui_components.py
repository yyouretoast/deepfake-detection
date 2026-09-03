"""Modular UI components and rendering helpers for the Streamlit deepfake forensics dashboard."""

from typing import Any
import streamlit as st


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

