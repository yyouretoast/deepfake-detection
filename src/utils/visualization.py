"""Visualization utilities for Deepfake Detector Engine."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_temporal_anomaly_timeline(
    timestamps: list[float],
    probs: list[float],
    threshold: float,
) -> matplotlib.figure.Figure:
    """Renders dark glassmorphism timeline graph of frame-by-frame confidence scores."""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#0b0f19")
    try:
        ax.set_facecolor("#0f172a")

        times = np.array(timestamps)
        scores = np.array(probs)

        # Plot smooth confidence curve
        ax.plot(
            times,
            scores,
            color="#60a5fa",
            linewidth=2.2,
            marker="o",
            markersize=6,
            label="Frame Confidence Score",
        )

        # Threshold line
        ax.axhline(
            y=threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold (T*={threshold:.2f})",
        )

        # Shaded anomaly regions
        ax.fill_between(
            times,
            scores,
            threshold,
            where=(scores >= threshold),
            color="#ef4444",
            alpha=0.22,
            interpolate=True,
        )
        ax.fill_between(
            times,
            scores,
            threshold,
            where=(scores < threshold),
            color="#22c55e",
            alpha=0.12,
            interpolate=True,
        )

        # Color-coded scatter markers
        fake_mask = scores >= threshold
        real_mask = scores < threshold
        if np.any(fake_mask):
            ax.scatter(
                times[fake_mask],
                scores[fake_mask],
                color="#ef4444",
                s=55,
                zorder=5,
                label="Anomalous Frame (Fake)",
            )
        if np.any(real_mask):
            ax.scatter(
                times[real_mask],
                scores[real_mask],
                color="#22c55e",
                s=55,
                zorder=5,
                label="Authentic Frame (Real)",
            )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Video Timestamp (seconds)", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.set_ylabel("Fake Probability", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.set_title("Temporal Anomaly Sequence Analysis", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)

        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        ax.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5, loc="upper right")
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise
