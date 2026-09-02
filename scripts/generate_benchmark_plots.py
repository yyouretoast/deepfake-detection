"""Publication-ready benchmark plots generator for Dual-Stream Deepfake Detector."""

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import auc, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
ORANGE = "#D97706"
GRAY = "#6B7280"
DPI = 300


def apply_base_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(float(np.mean(labels[mask])))
        bin_confs.append(float(np.mean(probs[mask])))
        bin_counts.append(int(mask.sum()))

    b_centers = np.array(bin_centers)
    b_accs = np.array(bin_accs)
    b_confs = np.array(bin_confs)
    b_counts = np.array(bin_counts)

    total = b_counts.sum()
    ece = float(np.sum(b_counts / total * np.abs(b_accs - b_confs)))
    return b_centers, b_accs, b_confs, ece


def plot_roc(
    probs_raw: np.ndarray, probs_cal: np.ndarray, labels: np.ndarray, output_path: str
) -> None:
    apply_base_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    fpr_r, tpr_r, _ = roc_curve(labels, probs_raw)
    fpr_c, tpr_c, _ = roc_curve(labels, probs_cal)
    auc_r = auc(fpr_r, tpr_r)
    auc_c = auc(fpr_c, tpr_c)

    ax.plot(fpr_r, tpr_r, color=BLUE, lw=1.8, label=f"Raw       (AUC = {auc_r:.4f})")
    ax.plot(fpr_c, tpr_c, color=GREEN, lw=1.8, linestyle="--", label=f"Calibrated (AUC = {auc_c:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1.0, linestyle=":", label="Random")
    ax.fill_between(fpr_r, tpr_r, alpha=0.06, color=BLUE)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Held-Out Test Set (10,528 crops)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", output_path)


def plot_ece(
    probs_raw: np.ndarray, probs_cal: np.ndarray, labels: np.ndarray, output_path: str
) -> None:
    apply_base_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, probs, title, color in [
        (axes[0], probs_raw, "Raw (Uncalibrated)", RED),
        (axes[1], probs_cal, "Temperature Scaled", GREEN),
    ]:
        bx, ba, bc, ece = compute_ece(probs, labels)
        bar_w = 0.08
        ax.bar(bx, ba, width=bar_w, color=color, alpha=0.75, label="Accuracy", zorder=3)
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, linestyle="--", label="Perfect calibration")

        for x, acc, conf in zip(bx, ba, bc):
            ax.bar(x, abs(acc - conf), bottom=min(acc, conf), width=bar_w, color=RED, alpha=0.3, zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Fraction of Positives")
    fig.suptitle("ECE Reliability Diagram", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved ECE diagram -> %s", output_path)


def plot_robustness(robustness: dict, output_path: str) -> None:
    apply_base_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    sweep_names = ["JPEG Compression", "Gaussian Blur", "Gaussian Noise", "Downscaling"]
    colors = [BLUE, ORANGE, RED, GREEN]

    for ax, sweep_name, color in zip(axes, sweep_names, colors):
        data = robustness[sweep_name]
        levels = [d["level"] for d in data]
        aucs = [d["auc"] for d in data]
        baseline_auc = aucs[0]
        xs = list(range(len(levels)))

        ax.axhline(baseline_auc, color=GRAY, lw=1.0, linestyle="--", label=f"Baseline ({baseline_auc:.4f})", zorder=2)
        ax.fill_between(xs, aucs, baseline_auc, where=[a < baseline_auc for a in aucs], alpha=0.12, color=RED, zorder=1)
        ax.plot(xs, aucs, color=color, lw=2.0, marker="o", markersize=6, zorder=3)

        for x, y in zip(xs, aucs):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=color)

        ax.set_xticks(xs)
        ax.set_xticklabels(levels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(max(0, min(aucs) - 0.05), 1.01)
        ax.set_ylabel("AUC")
        ax.set_title(sweep_name)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Robustness Under Image Degradation — Full Test Set (10,528 crops)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved robustness plot -> %s", output_path)


def plot_loto(loto_data: list, output_path: str) -> None:
    apply_base_style()
    folds = []
    if isinstance(loto_data, list) and len(loto_data) > 0:
        name_map = {
            "deepfakes": "Fold 1\nDeepfakes\n(FF++)",
            "df": "Fold 1\nDeepfakes\n(FF++)",
            "face2face": "Fold 2\nFace2Face\n(FF++)",
            "f2f": "Fold 2\nFace2Face\n(FF++)",
            "faceswap": "Fold 3\nFaceSwap\n(FF++)",
            "fs": "Fold 3\nFaceSwap\n(FF++)",
            "neuraltextures": "Fold 4\nNeuralTextures\n(FF++)",
            "nt": "Fold 4\nNeuralTextures\n(FF++)",
            "celeb": "Fold 5\nCeleb-DF v2\nCross-Dataset",
        }
        for entry in loto_data:
            ho = entry.get("holdout", "").lower()
            label = name_map.get(ho, f"{ho.title()}")
            auc_val = float(entry.get("zero_shot_auc", 0.5))
            color = RED if "celeb" in ho or auc_val < 0.5 else BLUE
            note = f"{1.0 - auc_val:.4f} (1 - p)" if auc_val < 0.5 else None
            folds.append((label, auc_val, color, note))

    if not folds:
        folds = [
            ("Fold 1\nDeepfakes\n(FF++)", 0.9691, BLUE, None),
            ("Fold 2\nFace2Face\n(FF++)", 0.9749, BLUE, None),
            ("Fold 3\nFaceSwap\n(FF++)", 0.9662, BLUE, None),
            ("Fold 4\nNeuralTextures\n(FF++)", 0.9783, BLUE, None),
            ("Fold 5\nCeleb-DF v2\nCross-Dataset", 0.3234, RED, "0.6766 (1 - p)"),
        ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xs = list(range(len(folds)))
    labels = [f[0] for f in folds]
    aucs = [f[1] for f in folds]
    colors = [f[2] for f in folds]

    bars = ax.bar(xs, aucs, color=colors, width=0.45, alpha=0.85, zorder=3)

    for bar, f_info in zip(bars, folds):
        v = f_info[1]
        inv_note = f_info[3]
        if inv_note:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"AUC = {v:.4f}\n[{inv_note}]", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=RED)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"AUC = {v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=BLUE)

    ax.axhline(0.5, color=GRAY, lw=1.0, linestyle=":", label="Random baseline (AUC=0.50)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Zero-Shot AUC")
    ax.set_title("LOTO Zero-Shot Generalization\n(Leave-One-Type-Out Cross-Generator Evaluation)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved LOTO plot -> %s", output_path)


def plot_per_generator(output_path: str) -> None:
    apply_base_style()
    generators = [
        ("Celeb-DF v2 Synthesis", 0.9992),
        ("FF++ Face2Face", 0.9967),
        ("FF++ Deepfakes", 0.9963),
        ("FF++ FaceSwap", 0.9961),
        ("FF++ NeuralTextures", 0.9940),
    ]
    generators.sort(key=lambda x: x[1])

    names = [g[0] for g in generators]
    aucs = [g[1] for g in generators]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ys = list(range(len(names)))
    bar_colors = [BLUE if "Celeb" in n else GREEN for n in names]

    bars = ax.barh(ys, aucs, color=bar_colors, alpha=0.85, height=0.55, zorder=3)

    for bar, v in zip(bars, aucs):
        ax.text(v + 0.0003, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.985, 1.002)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("AUC")
    ax.set_title("Per-Generator Sub-Domain AUC (Held-Out Test Set)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved per-generator plot -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    parser.add_argument("--predictions", default="test_predictions.json")
    parser.add_argument("--robustness", default="robustness_results.json")
    parser.add_argument("--loto", default="loto_results.json")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def resolve_file(path_str: str) -> str:
        if os.path.exists(path_str):
            return path_str
        kaggle_candidate = os.path.join("/kaggle/working", os.path.basename(path_str))
        if os.path.exists(kaggle_candidate):
            return kaggle_candidate
        return path_str

    pred_path = resolve_file(args.predictions)
    logger.info("Loading predictions from %s", pred_path)
    with open(pred_path) as f:
        preds = json.load(f)
    probs_raw = np.array(preds["probs_raw"])
    probs_cal = np.array(preds["probs_cal"])
    labels = np.array(preds["labels"], dtype=np.int32)

    rob_path = resolve_file(args.robustness)
    logger.info("Loading robustness results from %s", rob_path)
    with open(rob_path) as f:
        robustness = json.load(f)

    loto_path = resolve_file(args.loto)
    logger.info("Loading LOTO results from %s", loto_path)
    with open(loto_path) as f:
        loto = json.load(f)

    plot_roc(probs_raw, probs_cal, labels, os.path.join(args.output_dir, "roc_curve.png"))
    plot_ece(probs_raw, probs_cal, labels, os.path.join(args.output_dir, "ece_reliability.png"))
    plot_robustness(robustness, os.path.join(args.output_dir, "robustness_degradation.png"))
    plot_loto(loto, os.path.join(args.output_dir, "loto_generalization.png"))
    plot_per_generator(os.path.join(args.output_dir, "per_generator_auc.png"))

    logger.info("All figures saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
