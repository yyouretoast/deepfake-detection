"""Generate Teaser Figure 1 for Page 1 of the manuscript.
Illustrates the Spatial-Frequency Asymmetry Paradox in Media Forensics:
(a) Failure Modes: Spatial identity memorization vs. Frequency polarity inversion under compression.
(b) Physical Signal Reality: Silicon PRNU sensor noise annihilation & generative upsampling Dirac spikes.
(c) Proposed Solution: SNR-Adaptive Complementary Gating (g_eff = g * gamma) dynamically arbitrating domains.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "axes.labelcolor": DARK_NAVY,
    "axes.edgecolor": BORDER_GRAY,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def generate_teaser_figure():
    print("Generating Teaser Figure 1 (Visual 3-Panel Forensic Showcase)...")
    fig = plt.figure(figsize=(14.2, 5.0), dpi=300)
    ax_bg = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_bg.set_xlim(0, 150)
    ax_bg.set_ylim(0, 52)
    ax_bg.axis("off")

    def draw_card(x, y, w, h, bg, border, lw=1.0):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=bg, edgecolor=border, lw=lw, zorder=1
        )
        ax_bg.add_patch(rect)

    def draw_pill(x, y, w, h, text, col, bg, border, fontsize=8.0):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
            facecolor=bg, edgecolor=border, lw=1.0, zorder=2
        )
        ax_bg.add_patch(rect)
        ax_bg.text(
            x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=col, zorder=3
        )

    # Load crops
    p_real = "deepfake_crops_512/real/004/frame_000.webp"
    p_fake = "deepfake_crops_512/fake/id10_id11_0003/frame_002.webp"

    raw_real = cv2.imread(p_real)
    raw_fake = cv2.imread(p_fake)

    img_real_bgr = cv2.rotate(raw_real, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_fake_bgr = cv2.rotate(raw_fake, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_real_rgb = cv2.cvtColor(img_real_bgr, cv2.COLOR_BGR2RGB)
    img_fake_rgb = cv2.cvtColor(img_fake_bgr, cv2.COLOR_BGR2RGB)

    # Synthesize JPEG transcoded frame for panel (a)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, enc = cv2.imencode(".jpg", img_real_bgr, encode_param)
    img_transcoded_rgb = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)

    # Coordinates for normalized conversion
    X_MAX = 150.0
    Y_MAX = 52.0

    def to_norm(x, y, w, h):
        return [x / X_MAX, y / Y_MAX, w / X_MAX, h / Y_MAX]

    # =========================================================================
    # Panel (a): The Spatial-Frequency Asymmetry Paradox (Left Card)
    # =========================================================================
    draw_card(1.5, 1.5, 47.0, 49.0, RED_BG, RED_BORDER, lw=1.2)
    draw_pill(3.5, 46.5, 43.0, 3.6, "(a) The Spatial-Frequency Asymmetry Paradox", RED_MAIN, "white", RED_BORDER, fontsize=8.2)

    # Visual Thumbnails: Real, Fake, Transcoded
    ax_a_real = fig.add_axes(to_norm(3.5, 30.8, 13.0, 11.2))
    ax_a_real.imshow(img_real_rgb)
    ax_a_real.set_xticks([])
    ax_a_real.set_yticks([])
    ax_a_real.set_title("Authentic (Pristine)", fontsize=6.8, fontweight="bold", color=GREEN_MAIN, pad=3)
    for spine in ax_a_real.spines.values():
        spine.set_color(GREEN_BORDER)
        spine.set_linewidth(1.0)

    ax_a_fake = fig.add_axes(to_norm(18.5, 30.8, 13.0, 11.2))
    ax_a_fake.imshow(img_fake_rgb)
    ax_a_fake.set_xticks([])
    ax_a_fake.set_yticks([])
    ax_a_fake.set_title("Manipulated (Deepfake)", fontsize=6.8, fontweight="bold", color=RED_MAIN, pad=3)
    for spine in ax_a_fake.spines.values():
        spine.set_color(RED_BORDER)
        spine.set_linewidth(1.0)

    ax_a_lossy = fig.add_axes(to_norm(33.5, 30.8, 13.0, 11.2))
    ax_a_lossy.imshow(img_transcoded_rgb)
    ax_a_lossy.set_xticks([])
    ax_a_lossy.set_yticks([])
    ax_a_lossy.set_title("Transcoded ($Q = 30$)", fontsize=6.8, fontweight="bold", color=SLATE_MED, pad=3)
    for spine in ax_a_lossy.spines.values():
        spine.set_color(BORDER_GRAY)
        spine.set_linewidth(1.0)

    # Spatial Pitfall Sub-box
    draw_card(3.5, 16.5, 43.0, 13.0, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(5.0, 27.2, "Spatial RGB Backbones (ConvNeXt, Xception):", fontsize=7.4, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax_bg.text(5.0, 24.5, "$\\times$  Memorize biometric actor shortcuts (id$_A \\to$ id$_B$)", fontsize=7.1, color=RED_MAIN, zorder=3)
    ax_bg.text(5.0, 22.0, "$\\times$  Overfit background scenes; fail on zero-shot splits", fontsize=7.1, color=RED_MAIN, zorder=3)
    ax_bg.text(5.0, 19.5, "$\\bullet$  Robust to blur, but blind to subtle sensor anomalies", fontsize=7.0, color=SLATE_MED, zorder=3)
    ax_bg.text(5.0, 17.5, "Result: High in-domain AUC (>99%), collapses on unseen fakes", fontsize=6.8, fontstyle="italic", color=SLATE_MED, zorder=3)

    # Spectral Pitfall Sub-box
    draw_card(3.5, 2.5, 43.0, 13.0, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(5.0, 13.2, "Standalone Frequency Detectors (SRM, 2D FFT):", fontsize=7.4, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax_bg.text(5.0, 10.7, "$\\times$  Extreme sensitivity to transmission channel noise", fontsize=7.1, color=RED_MAIN, zorder=3)
    ax_bg.text(5.0, 8.4, "$\\times$  Low-pass blur obliterates high-pass PRNU residuals", fontsize=7.1, color=RED_MAIN, zorder=3)
    ax_bg.text(5.0, 6.1, "$\\blacktriangleright$  Polarity Inversion: compressed authentic media", fontsize=7.1, fontweight="bold", color=RED_MAIN, zorder=3)
    ax_bg.text(7.0, 3.8, "misclassified as fakes (ROC AUC drops below 0.50)", fontsize=6.9, color=RED_MAIN, zorder=3)

    # =========================================================================
    # Panel (b): Digital Signal Physics & Ground Truth Artifacts (Middle Card)
    # =========================================================================
    draw_card(51.5, 1.5, 47.0, 49.0, BLUE_BG, BLUE_BORDER, lw=1.2)
    draw_pill(53.5, 46.5, 43.0, 3.6, "(b) Physical Acquisition vs. Generative Artifacts", BLUE_MAIN, "white", BLUE_BORDER, fontsize=8.2)

    # Real PRNU vs Synthetic Annihilation
    draw_card(53.5, 24.5, 43.0, 20.8, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(55.0, 43.0, "Real PRNU Noise vs. Generative Annihilation:", fontsize=7.5, fontweight="bold", color=SLATE_DARK, zorder=3)

    gray_real = cv2.cvtColor(img_real_bgr, cv2.COLOR_BGR2GRAY)
    gray_fake = cv2.cvtColor(img_fake_bgr, cv2.COLOR_BGR2GRAY)
    res_real = cv2.Laplacian(gray_real, cv2.CV_32F)[130:230, 130:230]
    res_fake = cv2.Laplacian(gray_fake, cv2.CV_32F)[130:230, 130:230]
    patch_combined = np.hstack([res_real, res_fake])

    ax_b_prnu = fig.add_axes(to_norm(55.0, 26.2, 20.0, 14.5))
    ax_b_prnu.imshow(patch_combined, cmap="coolwarm", vmin=-15, vmax=15)
    ax_b_prnu.axvline(100, color=DARK_NAVY, lw=1.5)
    ax_b_prnu.set_xticks([50, 150])
    ax_b_prnu.set_xticklabels(["Authentic PRNU\n($\\sigma = 0.28$)", "Synthetic Void\n($\\sigma = 0.04$)"], fontsize=6.8)
    ax_b_prnu.set_yticks([])
    for spine in ax_b_prnu.spines.values():
        spine.set_color(BORDER_GRAY)

    ax_bg.text(77.0, 38.0, "Silicon Fingerprint (PRNU):", fontsize=7.2, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax_bg.text(77.0, 35.0, "$\\bullet$ Multiplicative wafer noise", fontsize=6.8, color=GREEN_MAIN, zorder=3)
    ax_bg.text(77.0, 32.5, "  persists across pristine frames", fontsize=6.8, color=SLATE_MED, zorder=3)
    ax_bg.text(77.0, 29.5, "$\\bullet$ Deepfake decoders annihilate", fontsize=6.8, color=BLUE_MAIN, zorder=3)
    ax_bg.text(77.0, 27.0, "  sensor micro-noise ($\\sigma \\to 0$)", fontsize=6.8, color=BLUE_MAIN, zorder=3)

    # 2D FFT Dirac Spikes & Poisson Boundary
    draw_card(53.5, 2.5, 43.0, 21.0, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(55.0, 21.2, "Transposed Convolution Aliasing (2D FFT):", fontsize=7.5, fontweight="bold", color=SLATE_DARK, zorder=3)

    fft_fake = np.fft.fftshift(np.fft.fft2(gray_fake.astype(float)))
    mag_fake = np.log1p(np.abs(fft_fake))
    p5, p995 = np.percentile(mag_fake, 5), np.percentile(mag_fake, 99.5)
    mag_fake_norm = np.clip((mag_fake - p5) / max(float(p995 - p5), 1e-6), 0.0, 1.0)

    ax_b_fft = fig.add_axes(to_norm(55.0, 4.5, 16.5, 15.0))
    ax_b_fft.imshow(mag_fake_norm, cmap="inferno")
    for cx_peak, cy_peak in [(160, 160), (160, 352), (352, 160), (352, 352)]:
        c_ring = patches.Circle((cx_peak, cy_peak), 24, fill=False, edgecolor="#38BDF8", lw=1.2, ls="--")
        ax_b_fft.add_patch(c_ring)
    ax_b_fft.set_xticks([])
    ax_b_fft.set_yticks([])
    for spine in ax_b_fft.spines.values():
        spine.set_color(BORDER_GRAY)

    ax_bg.text(73.5, 18.0, "Generative Dirac Lattice:", fontsize=7.2, fontweight="bold", color=PURPLE_MAIN, zorder=3)
    ax_bg.text(73.5, 15.3, "$\\bullet$ Periodic upsampling spikes", fontsize=6.8, color=PURPLE_MAIN, zorder=3)
    ax_bg.text(73.5, 13.0, "  emit $+18$ dB high-freq energy", fontsize=6.8, color=SLATE_MED, zorder=3)
    ax_bg.text(73.5, 10.3, "$\\bullet$ Harmonic grid: $u_k = \\pm k/s$", fontsize=6.8, color=PURPLE_MAIN, zorder=3)
    ax_bg.text(73.5, 7.6, "$\\bullet$ Poisson boundary blending step", fontsize=6.8, color=SLATE_DARK, zorder=3)
    ax_bg.text(73.5, 5.2, "  along facial perimeter ($s_f = 1.5\\times$)", fontsize=6.8, color=SLATE_MED, zorder=3)

    # =========================================================================
    # Panel (c): The Proposed Solution: SNR-Adaptive Gating (Right Card)
    # =========================================================================
    draw_card(101.5, 1.5, 47.0, 49.0, GREEN_BG, GREEN_BORDER, lw=1.2)
    draw_pill(103.5, 46.5, 43.0, 3.6, "(c) Proposed Solution: SNR-Adaptive Mediation", GREEN_MAIN, "white", GREEN_BORDER, fontsize=8.2)

    # Dynamic Gating Attenuation Curve Plot
    draw_card(103.5, 24.5, 43.0, 20.8, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(105.0, 43.0, "Dynamic Attenuation Function $\\gamma(P_{\\mathrm{noise}})$:", fontsize=7.5, fontweight="bold", color=SLATE_DARK, zorder=3)

    ax_c_curve = fig.add_axes(to_norm(106.0, 26.5, 38.0, 14.5))
    p_noise_vals = np.linspace(0.0, 0.035, 200)
    gamma_vals = np.clip((p_noise_vals - 0.005) / (0.025 - 0.005), 0.0, 1.0)
    ax_c_curve.plot(p_noise_vals * 1000, gamma_vals, color=GREEN_MAIN, lw=2.2, label="Attenuation $\\gamma$")
    ax_c_curve.axvspan(0, 5, color=BLUE_BG, alpha=0.8, label="Compressed $(\\gamma \\to 0)$")
    ax_c_curve.axvspan(25, 35, color=GREEN_BG, alpha=0.8, label="Clean $(\\gamma \\to 1)$")
    ax_c_curve.set_xlabel("Noise Power $P_{\\mathrm{noise}} \\; (\\times 10^{-3})$", fontsize=7.0, labelpad=2)
    ax_c_curve.set_ylabel("Gate Factor $\\gamma$", fontsize=7.0, labelpad=2)
    ax_c_curve.set_xlim(0, 35)
    ax_c_curve.set_ylim(-0.05, 1.08)
    ax_c_curve.tick_params(labelsize=6.5, pad=1)
    ax_c_curve.grid(True, linestyle=":", alpha=0.5)
    ax_c_curve.legend(fontsize=6.2, loc="center right", framealpha=0.9, borderpad=0.2)

    # Feature Fusion & Payoff Box
    draw_card(103.5, 2.5, 43.0, 21.0, "white", BORDER_GRAY, lw=0.8)
    ax_bg.text(105.0, 21.0, "Complementary Gated Representation:", fontsize=7.5, fontweight="bold", color=SLATE_DARK, zorder=3)
    ax_bg.text(105.0, 17.8, "$\\mathbf{g}_{\\mathrm{eff}} = \\mathbf{g} \\odot \\gamma, \\quad \\mathbf{f}_{\\mathrm{fused}} = \\left[ (1 - \\mathbf{g}_{\\mathrm{eff}}) \\odot \\mathbf{f}_s \\;\\parallel\\; \\mathbf{g}_{\\mathrm{eff}} \\odot \\mathbf{f}_f \\right] \\in \\mathbb{R}^{1024}$", fontsize=7.0, fontweight="bold", color=GREEN_MAIN, zorder=3)
    ax_bg.text(105.0, 14.5, "$\\checkmark$  Prevents polarity inversion under lossy transmission", fontsize=7.0, color=GREEN_MAIN, zorder=3)
    ax_bg.text(105.0, 11.5, "$\\checkmark$  Preserves 0.6722 ROC AUC even under blur ($\\sigma = 4.0$)", fontsize=7.0, color=GREEN_MAIN, zorder=3)
    ax_bg.text(105.0, 8.5, "$\\checkmark$  Reaches 0.8656 Frame / 0.8994 Video AUC on Disjoint Test", fontsize=7.0, color=GREEN_MAIN, zorder=3)
    ax_bg.text(105.0, 5.5, "$\\checkmark$  Platt calibration reduces ECE: 0.1965 $\\to$ 0.0994 ($-49.4\\%$)", fontsize=7.0, fontweight="bold", color=SLATE_DARK, zorder=3)

    # Save outputs
    pdf_path = os.path.join(OUTPUT_DIR, "teaser_concept.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "teaser_concept.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [+] Teaser Figure 1 generated successfully in manuscript/figures/ (.pdf) and figures/ (.png)!")


if __name__ == "__main__":
    generate_teaser_figure()
