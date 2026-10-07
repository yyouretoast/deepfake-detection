"""Compute 1D Azimuthal Radial Power Spectrum Profile for Authentic vs Synthetic Crops.

Evaluates natural isotropic 1/f^alpha power-law decay curves vs. high-frequency
Dirac lattice harmonic spikes and spectral elevation across manipulation subdomains.
Generates manuscript/figures/radial_spectral_profiles.pdf for Figure 8 / Appendix.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def compute_azimuthal_radial_profile(img_gray: np.ndarray, num_bins: int = 128) -> np.ndarray:
    """Computes the 1D azimuthally integrated power spectrum of a square grayscale image."""
    h, w = img_gray.shape
    # 2D Hann window
    hann_1d_h = np.hanning(h)
    hann_1d_w = np.hanning(w)
    hann_2d = np.outer(hann_1d_h, hann_1d_w)
    windowed = (img_gray.astype(np.float32) / 255.0) * hann_2d

    # 2D FFT and power spectrum
    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    # Radial grid
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Concentric azimuthal integration
    max_radius = min(cx, cy)
    r_int = r.astype(np.int32)
    radial_profile = np.zeros(max_radius, dtype=np.float64)

    for i in range(max_radius):
        mask = (r_int == i)
        if np.any(mask):
            radial_profile[i] = np.mean(power[mask])

    return radial_profile

def main():
    print("[+] Computing 1D Azimuthal Radial Power Spectrum Profiles...", flush=True)
    manifest_path = os.path.join(REPO_ROOT, "deepfake_crops_512", "manifest.json")
    if not os.path.exists(manifest_path):
        print("[-] manifest.json not found.", flush=True)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Select representative samples per category
    samples_per_cat = 80
    categories = {
        "Authentic Natural": [],
        "Deepfakes (Autoencoder)": [],
        "Face2Face (3DMM)": [],
        "FaceSwap (Graphic)": [],
        "NeuralTextures (Neural)": [],
        "Celeb-DF v2": [],
    }

    for item in manifest:
        p = item["path"].replace("\\", "/")
        lbl = item["label"]
        if lbl == 0:
            if len(categories["Authentic Natural"]) < samples_per_cat:
                full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                if os.path.exists(full_p):
                    categories["Authentic Natural"].append(full_p)
        else:
            folder = p.split("/")[1] if len(p.split("/")) > 1 else ""
            if folder.startswith("id") and len(categories["Celeb-DF v2"]) < samples_per_cat:
                full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                if os.path.exists(full_p):
                    categories["Celeb-DF v2"].append(full_p)
            elif "_" in folder:
                pair_idx = folder.split("_")[0]
                if pair_idx.isdigit():
                    idx = int(pair_idx)
                    if idx < 100 and len(categories["Deepfakes (Autoencoder)"]) < samples_per_cat:
                        full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                        if os.path.exists(full_p):
                            categories["Deepfakes (Autoencoder)"].append(full_p)
                    elif 100 <= idx < 400 and len(categories["Face2Face (3DMM)"]) < samples_per_cat:
                        full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                        if os.path.exists(full_p):
                            categories["Face2Face (3DMM)"].append(full_p)
                    elif 400 <= idx < 600 and len(categories["FaceSwap (Graphic)"]) < samples_per_cat:
                        full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                        if os.path.exists(full_p):
                            categories["FaceSwap (Graphic)"].append(full_p)
                    elif 600 <= idx < 800 and len(categories["NeuralTextures (Neural)"]) < samples_per_cat:
                        full_p = os.path.join(REPO_ROOT, "deepfake_crops_512", p)
                        if os.path.exists(full_p):
                            categories["NeuralTextures (Neural)"].append(full_p)

    profiles = {}
    for cat_name, file_list in categories.items():
        cat_profiles = []
        for fp in file_list:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                p = compute_azimuthal_radial_profile(img, num_bins=128)
                cat_profiles.append(p)
        if cat_profiles:
            profiles[cat_name] = np.mean(cat_profiles, axis=0)
            print(f"  [+] Processed {len(cat_profiles)} crops for '{cat_name}'", flush=True)

    from scipy.ndimage import gaussian_filter1d

    # Professional academic palette matching publication figures
    DARK_NAVY = "#0F172A"
    BORDER_GRAY = "#CBD5E1"
    GREEN_MAIN = "#15803D"
    RED_MAIN = "#B91C1C"
    BLUE_MAIN = "#1D4ED8"
    AMBER_MAIN = "#B45309"
    PURPLE_MAIN = "#6D28D9"
    CYAN_MAIN = "#0284C7"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "axes.labelcolor": DARK_NAVY,
        "axes.edgecolor": BORDER_GRAY,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=300)
    plt.subplots_adjust(wspace=0.26)
    radii = np.arange(len(profiles["Authentic Natural"]))
    freq_normalized = radii / float(len(radii))  # cycles / pixel

    colors = {
        "Authentic Natural": GREEN_MAIN,
        "Deepfakes (Autoencoder)": RED_MAIN,
        "Face2Face (3DMM)": BLUE_MAIN,
        "FaceSwap (Graphic)": AMBER_MAIN,
        "NeuralTextures (Neural)": PURPLE_MAIN,
        "Celeb-DF v2": CYAN_MAIN,
    }
    styles = {
        "Authentic Natural": ("-", 2.2),
        "Deepfakes (Autoencoder)": ("--", 1.6),
        "Face2Face (3DMM)": ("-.", 1.6),
        "FaceSwap (Graphic)": (":", 1.8),
        "NeuralTextures (Neural)": ("--", 1.6),
        "Celeb-DF v2": ("-.", 1.6),
    }

    # Panel 1: Radial Power Spectrum P(r) (log10)
    for cat_name, prof in profiles.items():
        ls, lw = styles.get(cat_name, ("-", 1.5))
        c = colors.get(cat_name, "#333333")
        log_p = np.log10(prof + 1e-12)
        log_p_smooth = gaussian_filter1d(log_p, sigma=1.0)
        ax1.plot(freq_normalized, log_p_smooth, label=cat_name, color=c, linestyle=ls, linewidth=lw)

    ax1.axvspan(0.35, 0.50, color="#FEE2E2", alpha=0.4, label="High-Freq. Divergence Zone")
    ax1.set_title("(a) Azimuthal Radial Power Spectrum $\\log_{10} P(r)$", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Spatial Frequency Radius $r$ (cycles / pixel)", fontsize=9)
    ax1.set_ylabel("Log Power $\\log_{10} P(r)$", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax1.set_xlim(0.02, 0.50)

    # Panel 2: Relative Spectral Elevation Delta P(r) = P_fake(r) - P_real(r) (dB)
    p_real = profiles["Authentic Natural"]
    log_p_real = np.log10(p_real + 1e-12)
    ax2.axhline(0, color=GREEN_MAIN, linestyle="-", linewidth=1.5, alpha=0.8, label="Authentic Baseline (0 dB)")
    for cat_name, prof in profiles.items():
        if cat_name == "Authentic Natural":
            continue
        ls, lw = styles.get(cat_name, ("-", 1.5))
        c = colors.get(cat_name, "#333333")
        log_p_fake = np.log10(prof + 1e-12)
        delta_db = 10.0 * (log_p_fake - log_p_real)
        delta_db_smooth = gaussian_filter1d(delta_db, sigma=1.2)
        ax2.plot(freq_normalized, delta_db_smooth, label=cat_name, color=c, linestyle=ls, linewidth=lw)

    ax2.set_title("(b) Relative High-Frequency Elevation $\\Delta P(r)$ (dB)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Spatial Frequency Radius $r$ (cycles / pixel)", fontsize=9)
    ax2.set_ylabel("Power Discrepancy $\\Delta P(r)$ (dB)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=7.6, framealpha=0.92)
    ax2.set_xlim(0.05, 0.50)

    out_pdf = os.path.join(REPO_ROOT, "manuscript", "figures", "radial_spectral_profiles.pdf")
    out_png = os.path.join(REPO_ROOT, "manuscript", "figures", "radial_spectral_profiles.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    # Also sync to root figures
    fig.savefig(os.path.join(REPO_ROOT, "figures", "radial_spectral_profiles.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(REPO_ROOT, "figures", "radial_spectral_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Successfully saved radial profile figure:\n  - {out_pdf}\n  - {out_png}", flush=True)

if __name__ == "__main__":
    main()
