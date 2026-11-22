"""Smoke tests for Streamlit UI helpers, image compatibility, and verdict consistency."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.services.ui_components import format_verdict, render_diagnostic_quad, render_image


def test_render_image_modern_streamlit_signature() -> None:
    """Uses use_container_width=True when supported by st.image."""
    captured: list[dict] = []

    def fake_image(image, caption=None, use_container_width=False):
        captured.append({"caption": caption, "use_container_width": use_container_width})

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with patch("src.services.ui_components.st.image", new=fake_image):
        render_image(img, caption="Modern", full_width=True)

    assert len(captured) == 1
    assert captured[0] == {"caption": "Modern", "use_container_width": True}


def test_render_image_legacy_streamlit_132_signature() -> None:
    """Falls back to use_column_width=True on Streamlit 1.32.0 where use_container_width is absent."""
    captured: list[dict] = []

    def legacy_image(image, caption=None, use_column_width=False):
        captured.append({"caption": caption, "use_column_width": use_column_width})

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with patch("src.services.ui_components.st.image", new=legacy_image):
        render_image(img, caption="Legacy", full_width=True)

    assert len(captured) == 1
    assert captured[0] == {"caption": "Legacy", "use_column_width": True}


def test_render_diagnostic_quad_renders_all_four_panels() -> None:
    """Renders RGB, SRM, FFT, and Grad-CAM panels without TypeError on legacy st.image."""
    calls: list[str] = []

    def legacy_image(image, caption=None, use_column_width=False):
        calls.append(caption or "")

    dummy = np.zeros((16, 16, 3), dtype=np.uint8)
    diag = {
        "original": dummy,
        "srm_residual": dummy,
        "fft_spectrum": dummy,
        "gradcam_overlay": dummy,
    }
    fake_cols = [MagicMock() for _ in range(4)]
    for col in fake_cols:
        col.__enter__ = MagicMock(return_value=col)
        col.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.services.ui_components.st.markdown"),
        patch("src.services.ui_components.st.columns", return_value=fake_cols),
        patch("src.services.ui_components.st.image", new=legacy_image),
    ):
        render_diagnostic_quad(diag, title_prefix="Test Quad")

    assert calls == [
        "RGB Face Crop",
        "SRM High-Pass Residual",
        "2D FFT Magnitude Spectrum",
        "Spatial Grad-CAM Overlay",
    ]


def test_format_verdict_consistency_across_zones() -> None:
    """Verifies card title, headline, and JSON verdict never contradict each other."""
    # prob=0.35 is above binary threshold 0.26 but inside tau_real <= 0.40:
    # card and JSON verdict must both agree on REAL with Authentic confidence.
    v_real = format_verdict(0.35, tau_real=0.40, tau_fake=0.60, threshold=0.26, media_kind="image")
    assert v_real["verdict"] == "REAL"
    assert v_real["card_title"] == "DETECTED: AUTHENTIC"
    assert v_real["headline"] == "Authentic Confidence: 65.0%"
    assert v_real["binary_decision"] == "FAKE"

    v_low = format_verdict(0.10, tau_real=0.40, tau_fake=0.60, threshold=0.26, media_kind="video")
    assert v_low["verdict"] == "REAL"
    assert v_low["card_title"] == "DETECTED: AUTHENTIC SEQUENCE"
    assert v_low["headline"] == "Authentic Confidence: 90.0%"
    assert v_low["binary_decision"] == "REAL"

    v_amb = format_verdict(0.50, tau_real=0.40, tau_fake=0.60, threshold=0.26, media_kind="image")
    assert v_amb["verdict"] == "INCONCLUSIVE"
    assert v_amb["is_inconclusive"] is True
    assert v_amb["card_title"] == "INCONCLUSIVE — MANUAL REVIEW"

    v_fake = format_verdict(0.85, tau_real=0.40, tau_fake=0.60, threshold=0.26, media_kind="image")
    assert v_fake["verdict"] == "FAKE"
    assert v_fake["card_title"] == "DETECTED: SYNTHETIC"
    assert v_fake["headline"] == "Synthetic Confidence: 85.0%"
    assert v_fake["binary_decision"] == "FAKE"
