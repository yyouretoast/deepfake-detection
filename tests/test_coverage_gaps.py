"""Unit tests closing coverage gaps: SHA-256 guards, timeline plotting, and calibration parity."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

from src.dataset import preprocess
from src.services.video_engine import _verify_sha256
from src.utils.calibration_release1 import (
    DISPLAY_EFFECTIVE_TEMPERATURE,
    DISPLAY_PLATT_BIAS_B,
    DISPLAY_PLATT_SCALE_A,
    OPTIMAL_THRESHOLD,
    TAU_FAKE,
    TAU_REAL,
    TEMPORAL_OPTIMAL_THRESHOLD,
)
from src.utils.visualization import render_temporal_anomaly_timeline


def test_verify_sha256_accepts_match_and_rejects_mismatch(tmp_path: Path) -> None:
    payload = b"trusted checkpoint bytes"
    expected = hashlib.sha256(payload).hexdigest()
    ckpt_file = tmp_path / "weights.pth"
    ckpt_file.write_bytes(payload)

    _verify_sha256(str(ckpt_file), expected)
    _verify_sha256(str(ckpt_file), expected.upper())

    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_sha256(str(ckpt_file), "0" * 64)


def test_get_yunet_model_path_rejects_bad_download_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pkg_file = tmp_path / "src" / "dataset" / "preprocess.py"
    fake_pkg_file.parent.mkdir(parents=True)
    fake_pkg_file.write_text("# stub", encoding="utf-8")

    monkeypatch.setattr(preprocess, "__file__", str(fake_pkg_file))
    monkeypatch.chdir(tmp_path)

    fake_resp = MagicMock()
    fake_resp.read.return_value = b"untrusted-onnx-payload-" * 100
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)

    with patch("src.dataset.preprocess.urllib.request.urlopen", return_value=fake_resp):
        resolved = preprocess.get_yunet_model_path()

    assert resolved is None
    expected_download = tmp_path / "models" / preprocess.YUNET_MODEL_FILENAME
    assert not expected_download.exists(), "Untrusted download must be deleted on SHA-256 mismatch"


def test_render_temporal_anomaly_timeline_with_and_without_attention() -> None:
    timestamps = [0.0, 0.5, 1.0, 1.5]
    probs = [0.15, 0.35, 0.72, 0.88]

    fig_plain = render_temporal_anomaly_timeline(timestamps, probs, threshold=0.26, attention_weights=None)
    assert isinstance(fig_plain, matplotlib.figure.Figure)
    plt.close(fig_plain)

    fig_attn = render_temporal_anomaly_timeline(
        timestamps, probs, threshold=0.26, attention_weights=[0.1, 0.5, 0.3, 0.1]
    )
    assert isinstance(fig_attn, matplotlib.figure.Figure)
    plt.close(fig_attn)


def test_calibration_release1_matches_sidecar_json() -> None:
    sidecar_path = Path(__file__).resolve().parents[1] / "models" / "dual_stream_detector.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert OPTIMAL_THRESHOLD == pytest.approx(sidecar["optimal_threshold"])
    assert TAU_REAL == pytest.approx(sidecar["tau_real"])
    assert TAU_FAKE == pytest.approx(sidecar["tau_fake"])
    assert DISPLAY_EFFECTIVE_TEMPERATURE == pytest.approx(sidecar["temperature"])
    assert DISPLAY_PLATT_SCALE_A == pytest.approx(sidecar["platt_a"])
    assert DISPLAY_PLATT_BIAS_B == pytest.approx(sidecar["platt_b"])
    assert TEMPORAL_OPTIMAL_THRESHOLD == pytest.approx(sidecar["temporal_optimal_threshold"])
