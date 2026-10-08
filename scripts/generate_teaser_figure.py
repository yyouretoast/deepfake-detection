"""Generate Teaser Figure 1 for Page 1 of the manuscript.
Illustrates the Spatial-Frequency Asymmetry Paradox in Media Forensics:
(a) Failure Modes: Spatial identity memorization vs. Frequency polarity inversion under compression.
(b) Physical Signal Reality: Silicon PRNU sensor noise annihilation & generative upsampling Dirac spikes.
(c) Proposed Solution: SNR-Adaptive Complementary Gating (g_eff = g * gamma) dynamically arbitrating domains.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.environ["MPLCONFIGDIR"] = os.path.abspath(".mpl_cache")
os.makedirs(".mpl_cache", exist_ok=True)

OUTPUT_DIR = "manuscript/figures"
ROOT_FIGURES_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROOT_FIGURES_DIR, exist_ok=True)

# Academic Color Palette
DARK_NAVY = "#0F172A"
SLATE_DARK = "#1E293B"
SLATE_MED = "#475569"
SLATE_LIGHT = "#F8FAFC"
BORDER_GRAY = "#CBD5E1"
LINE_GRAY = "#94A3B8"

BLUE_MAIN = "#1D4ED8"
BLUE_BG = "#EFF6FF"
BLUE_BORDER = "#93C5FD"

PURPLE_MAIN = "#6D28D9"
PURPLE_BG = "#FAF5FF"
PURPLE_BORDER = "#C4B5FD"

GREEN_MAIN = "#15803D"
GREEN_BG = "#F0FDF4"
GREEN_BORDER = "#86EFAC"

RED_MAIN = "#B91C1C"
RED_BG = "#FEF2F2"
RED_BORDER = "#FCA5A5"

AMBER_MAIN = "#B45309"
AMBER_BG = "#FFFBEB"
AMBER_BORDER = "#FCD34D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.labelcolor": DARK_NAVY,
    "axes.edgecolor": BORDER_GRAY,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_teaser_figure():
    print("Generating Teaser Figure 1 (Spatial-Frequency Asymmetry Paradox)...")
    fig = plt.figure(figsize=(12.0, 3.8), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Helper: Card Box
    def draw_card(x, y, w, h, bg, border, lw=1.0):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                              facecolor=bg, edgecolor=border, lw=lw, zorder=1)
        ax.add_patch(rect)

    # Helper: Title Pill
    def draw_pill(x, y, w, h, text, col, bg, border, fontsize=8.0):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                              facecolor=bg, edgecolor=border, lw=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=col, zorder=3)

    # =========================================================================
    # Panel (a): The Conventional Failure Modes (Left Card)
    # =========================================================================
    draw_card(1.5, 2.0, 46.0, 45.0, RED_BG, RED_BORDER, lw=1.2)
    draw_pill(3.5, 42.0, 42.0, 3.8, "(a) The Spatial-Frequency Asymmetry Paradox", RED_MAIN, "white", RED_BORDER, fontsize=8.2)

    # Spatial Pitfall Sub-box
    draw_card(3.5, 23.5, 42.0, 16.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(5.5, 37.0, "Spatial RGB Backbones (ConvNeXt, Xception):", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(5.5, 33.5, "[x] Memorize biometric actor shortcuts (A -> B)", fontsize=7.4, color=RED_MAIN, zorder=3)
    ax.text(5.5, 30.5, "[x] Overfit background scenes; fail on zero-shot fakes", fontsize=7.4, color=RED_MAIN, zorder=3)
    ax.text(5.5, 27.5, "[!] Robust to blur, but blind to subtle sensor anomalies", fontsize=7.2, color=SLATE_MED, zorder=3)
    ax.text(5.5, 25.0, "Result: High in-domain AUC (>99%), collapse on unseen splits", fontsize=7.0, fontstyle="italic", color=SLATE_MED, zorder=3)

    # Spectral Pitfall Sub-box
    draw_card(3.5, 4.5, 42.0, 17.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(5.5, 18.5, "Standalone Frequency Detectors (SRM, 2D FFT):", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(5.5, 15.2, "[x] Extreme sensitivity to transmission channel noise", fontsize=7.4, color=RED_MAIN, zorder=3)
    ax.text(5.5, 12.2, "[x] Low-pass blur destroys high-pass PRNU residuals", fontsize=7.4, color=RED_MAIN, zorder=3)
    ax.text(5.5, 9.2, "[!] Polarity Inversion: pristine compressed images", fontsize=7.4, fontweight="bold", color=RED_MAIN, zorder=3)
    ax.text(7.5, 6.5, "misclassified as fakes (ROC AUC drops below 0.50)", fontsize=7.2, color=RED_MAIN, zorder=3)

    # =========================================================================
    # Panel (b): Digital Signal Physics & Ground Truth Artifacts (Middle Card)
    # =========================================================================
    draw_card(49.5, 2.0, 47.0, 45.0, BLUE_BG, BLUE_BORDER, lw=1.2)
    draw_pill(51.5, 42.0, 43.0, 3.8, "(b) Physical Acquisition vs. Generative Artifacts", BLUE_MAIN, "white", BLUE_BORDER, fontsize=8.2)

    # Real Sensor Micro-Noise
    draw_card(51.5, 23.5, 43.0, 16.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(53.5, 37.0, "Real Camera Silicon Fingerprint (PRNU):", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(53.5, 33.5, "* Multiplicative silicon wafer imperfections: K_PRNU", fontsize=7.4, color=GREEN_MAIN, zorder=3)
    ax.text(53.5, 30.5, "* High-pass residual power: sigma_real = 0.28 (continuous)", fontsize=7.4, color=GREEN_MAIN, zorder=3)
    ax.text(53.5, 27.5, "* Generative decoder PRNU annihilation: sigma_fake = 0.04", fontsize=7.4, color=BLUE_MAIN, zorder=3)
    ax.text(53.5, 25.0, "Result: Indelible micro-noise contrast in pristine frames", fontsize=7.0, fontstyle="italic", color=SLATE_MED, zorder=3)

    # Generative Periodic Dirac Spikes
    draw_card(51.5, 4.5, 43.0, 17.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(53.5, 18.5, "Transposed Convolution Aliasing & Blending:", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(53.5, 15.2, "* Transposed conv upsampling emits periodic Dirac spikes", fontsize=7.4, color=PURPLE_MAIN, zorder=3)
    ax.text(53.5, 12.2, "* +18 dB high-frequency spectral elevation over 1/f decay", fontsize=7.4, color=PURPLE_MAIN, zorder=3)
    ax.text(53.5, 9.2, "* Poisson boundary blending produces sharp C1 gradient", fontsize=7.4, color=PURPLE_MAIN, zorder=3)
    ax.text(55.5, 6.5, "step discontinuities along facial perimeter (sf = 1.50x)", fontsize=7.2, color=SLATE_MED, zorder=3)

    # =========================================================================
    # Panel (c): The Proposed Solution: SNR-Adaptive Gating (Right Card)
    # =========================================================================
    draw_card(98.5, 2.0, 50.0, 45.0, GREEN_BG, GREEN_BORDER, lw=1.2)
    draw_pill(100.5, 42.0, 46.0, 3.8, "(c) Proposed Solution: SNR-Adaptive Mediation", GREEN_MAIN, "white", GREEN_BORDER, fontsize=8.2)

    draw_card(100.5, 23.5, 46.0, 16.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(102.5, 37.0, "Dynamic Noise Attenuation Factor (gamma):", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(102.5, 33.2, "gamma = clip((P_noise - eps_0) / (eps_1 - eps_0), 0, 1)", fontsize=7.6, fontweight="bold", color=GREEN_MAIN, zorder=3)
    ax.text(102.5, 29.8, "* High Noise Power (Clean Stream): gamma -> 1.0", fontsize=7.4, color=PURPLE_MAIN, zorder=3)
    ax.text(104.5, 27.2, "Full spectral power leverages PRNU & Dirac spikes", fontsize=7.0, color=SLATE_MED, zorder=3)
    ax.text(102.5, 24.5, "* Low Noise Power (Compressed / Blurred): gamma -> 0.0", fontsize=7.4, color=BLUE_MAIN, zorder=3)

    draw_card(100.5, 4.5, 46.0, 17.5, "white", BORDER_GRAY, lw=0.8)
    ax.text(102.5, 18.5, "Effective Complementary Gated Fusion:", fontsize=7.8, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax.text(102.5, 15.0, "g_eff = [g_s,  g_f * gamma]^T  ==>  f_fused in R^1024", fontsize=7.6, fontweight="bold", color=GREEN_MAIN, zorder=3)
    ax.text(102.5, 11.8, "[+] Prevents spectral polarity inversion under compression", fontsize=7.4, color=GREEN_MAIN, zorder=3)
    ax.text(102.5, 9.0, "[+] Preserves 0.6722 ROC AUC even under extreme blur (sigma=4.0)", fontsize=7.4, color=GREEN_MAIN, zorder=3)
    ax.text(102.5, 6.2, "[+] Reaches 0.8656 Frame / 0.8994 Video AUC on Disjoint Test", fontsize=7.4, fontweight="bold", color=SLATE_DARK, zorder=3)

    # Save
    pdf_path = os.path.join(OUTPUT_DIR, "teaser_concept.pdf")
    png_path = os.path.join(OUTPUT_DIR, "teaser_concept.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "teaser_concept.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "teaser_concept.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Teaser Figure 1 generated successfully in manuscript/figures/ and figures/!")


if __name__ == "__main__":
    generate_teaser_figure()
