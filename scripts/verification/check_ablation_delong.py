"""Replay ablation_cache.npz: reproduce ablation table metrics + fusion DeLong Z claims.

Paper claims under audit (main.tex lines 371, 423):
  - static gate AUC 0.8508 / EER 22.65
  - uniform concat (fusion_elementwise_sum) AUC 0.8522 / EER 22.85
  - full SNR gate AUC 0.8656 / EER 21.82, Youden F1 0.8466
  - DeLong Z = 25.65 (p < 10^-140) full vs static gate
  - DeLong Z = 27.03 (p < 10^-160) full vs uniform concat
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Platt constants as stored in checkpoint / used by compute_table5_ablations.py
SCALE_A = 0.27831
BIAS_B = 0.40891


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    k, n = predictions_sorted_transposed.shape
    m = label_1_count
    n_minus_m = n - m
    positive_submatrix = predictions_sorted_transposed[:, :m]
    negative_submatrix = predictions_sorted_transposed[:, m:]
    V_10 = np.empty((k, m), dtype=float)
    V_01 = np.empty((k, n_minus_m), dtype=float)
    for i in range(k):
        ranks = compute_midrank(predictions_sorted_transposed[i, :])
        V_10[i, :] = (ranks[:m] - compute_midrank(positive_submatrix[i, :])) / n_minus_m
        V_01[i, :] = 1.0 - (ranks[m:] - compute_midrank(negative_submatrix[i, :])) / m
    auc = np.mean(V_10, axis=1)
    S_10 = np.cov(V_10)
    S_01 = np.cov(V_01)
    S = S_10 / m + S_01 / n_minus_m
    return auc, S


def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true).astype(bool)
    order = np.argsort(~y_true)  # positives first
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted))
    preds = np.vstack([y_pred1[order], y_pred2[order]])
    aucs, cov = fast_delong(preds, m)
    auc1, auc2 = float(aucs[0]), float(aucs[1])
    diff = auc1 - auc2
    sigma_diff = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]))
    if sigma_diff < 1e-12:
        z_score, p_val = 0.0, 1.0
    else:
        z_score = diff / sigma_diff
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
    return auc1, auc2, diff, sigma_diff, float(z_score), float(p_val)


def find_youden_tau(y, p):
    """Grid-free Youden via ROC threshold sweep."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def main():
    with open("results/release1_run/release1_results.json", encoding="utf-8") as f:
        rel = json.load(f)
    stored = {
        "full": rel["ablation_study"]["full_dual_stream_snr_gate"],
        "static": rel["ablation_study"]["fusion_static_gate"],
        "sum": rel["ablation_study"]["fusion_elementwise_sum"],
        "spectral": rel["ablation_study"]["spectral_tower_alone"],
        "srm": rel["ablation_study"]["fixed_srm_alone"],
        "bayar": rel["ablation_study"]["learnable_bayar_alone"],
    }
    ck = np.load("results/release1_run/ablation_cache.npz")
    y = ck["y_true"].astype(int)

    probs = {}
    for name, key in [
        ("full", "logits_full"),
        ("static", "logits_static_gate"),
        ("spectral", "logits_spectral_alone"),
        ("sum", "logits_sum_fusion"),
        ("srm", "logits_srm_alone"),
        ("bayar", "logits_bayar_alone"),
    ]:
        probs[name] = 1.0 / (1.0 + np.exp(-(SCALE_A * ck[key] + BIAS_B)))

    print("=== Reproduce stored ablation_study values from cache ===")
    for name, p in probs.items():
        auc = roc_auc_score(y, p)
        eer_value = compute_eer_pct(y, p)
        tau = find_youden_tau(y, p)
        f1_y = f1_score(y, (p >= tau).astype(int), zero_division=0)
        s = stored[name]
        print(
            f"{name:9s} replay AUC={auc:.6f} (stored {s['auc']:.6f}, d={abs(auc-s['auc']):.6f}) | "
            f"EER={eer_value:.4f} (stored {s['eer']:.4f}) | YoudenF1={f1_y:.4f} (stored F1 {s['f1']:.4f})"
        )

    print("\n=== Youden-optimal F1 for full (paper claims 0.8466) ===")
    p_full = probs["full"]
    tau_full = find_youden_tau(y, p_full)
    f1_full_y = f1_score(y, (p_full >= tau_full).astype(int), zero_division=0)
    print(f"Youden tau_full={tau_full:.4f}, F1={f1_full_y:.4f}")

    print("\n=== DeLong tests between fusion variants (paper: Z=25.65 / Z=27.03) ===")
    pairs = [("full", "static"), ("full", "sum")]
    for a, b in pairs:
        auc1, auc2, diff, sig, z, p = delong_roc_test(y, probs[a], probs[b])
        print(f"{a} vs {b}: AUC {auc1:.6f} vs {auc2:.6f} dAUC={diff:+.6f} sigma={sig:.3e} Z={z:.4f} p={p:.3e}")

    # paper p bound checks
    _, _, _, _, _, p_s = delong_roc_test(y, probs["full"], probs["static"])
    _, _, _, _, _, p_c = delong_roc_test(y, probs["full"], probs["sum"])
    print(f"\nClaim 'p < 1e-140' for Z_static: actual {p_s:.3e} -> holds: {p_s < 1e-140}")
    print(f"Claim 'p < 1e-160' for Z_concat: actual {p_c:.3e} -> holds: {p_c < 1e-160}")


def compute_eer_pct(y, p):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)


if __name__ == "__main__":
    main()
