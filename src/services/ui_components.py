"""Modular UI components and rendering helpers for the Streamlit deepfake forensics dashboard."""

import inspect
from typing import Any

import numpy as np
import streamlit as st

from src.utils.checkpoint import classify_three_zone


def format_verdict(
    prob: float,
    tau_real: float,
    tau_fake: float,
    threshold: float,
    media_kind: str = "image",
) -> dict[str, Any]:
    """Compute consistent 3-zone triage card fields and JSON verdict from a calibrated probability."""
    p = float(np.clip(prob, 0.0, 1.0))
    three_zone = classify_three_zone(p, tau_real=tau_real, tau_fake=tau_fake)
    zone = three_zone["zone"]
    binary_decision = "FAKE" if p >= threshold else "REAL"

    if zone == "ambiguity_zone":
        return {
            "zone": zone,
            "three_zone": three_zone,
            "is_inconclusive": True,
            "verdict": "INCONCLUSIVE",
            "binary_decision": binary_decision,
            "card_class": "result-card-ambiguous",
            "color": "#f59e0b",
            "card_title": "INCONCLUSIVE — MANUAL REVIEW",
            "headline": f"Calibrated Prob: {p:.4f}",
            "confidence_pct": round(float(max(p, 1.0 - p)) * 100.0, 2),
        }
    if zone == "high_confidence_fake":
        conf_pct = round(p * 100.0, 2)
        return {
            "zone": zone,
            "three_zone": three_zone,
            "is_inconclusive": False,
            "verdict": "FAKE",
            "binary_decision": binary_decision,
            "card_class": "result-card-fake",
            "color": "#ef4444",
            "card_title": "DETECTED: SYNTHETIC" if media_kind == "image" else "DETECTED: SYNTHETIC SEQUENCE",
            "headline": f"Synthetic Confidence: {conf_pct:.1f}%",
            "confidence_pct": conf_pct,
        }
    conf_pct = round((1.0 - p) * 100.0, 2)
    return {
        "zone": zone,
        "three_zone": three_zone,
        "is_inconclusive": False,
        "verdict": "REAL",
        "binary_decision": binary_decision,
        "card_class": "result-card-real",
        "color": "#22c55e",
        "card_title": "DETECTED: AUTHENTIC" if media_kind == "image" else "DETECTED: AUTHENTIC SEQUENCE",
        "headline": f"Authentic Confidence: {conf_pct:.1f}%",
        "confidence_pct": conf_pct,
    }


def render_image(image: Any, caption: str | None = None, full_width: bool = True) -> None:
    """Render an image using the active Streamlit API's width parameter."""
    kwargs: dict[str, Any] = {}
    if caption is not None:
        kwargs["caption"] = caption
    if full_width:
        params = inspect.signature(st.image).parameters
        if "use_container_width" in params:
            kwargs["use_container_width"] = True
        elif "use_column_width" in params:
            kwargs["use_column_width"] = True
    st.image(image, **kwargs)


def render_diagnostic_quad(diag: dict[str, Any], title_prefix: str = "Diagnostic Quad") -> None:
    """Renders 4-panel visual interpretability representations (RGB, SRM Residual, 2D FFT, Grad-CAM)."""
    st.markdown(f"#### {title_prefix}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_image(diag["original"], caption="RGB Face Crop")
    with c2:
        render_image(diag["srm_residual"], caption="SRM High-Pass Residual")
    with c3:
        render_image(diag["fft_spectrum"], caption="2D FFT Magnitude Spectrum")
    with c4:
        render_image(diag["gradcam_overlay"], caption="Spatial Grad-CAM Overlay")
