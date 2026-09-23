import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, f1_score

with open("test_predictions.json", "r") as f:
    d = json.load(f)

y_true = np.array(d["labels"])
threshold = d["threshold"]
optimal_threshold_platt = d.get("optimal_threshold_platt")

print(f"Total samples: {len(y_true)}, Positives (fake): {np.sum(y_true)}, Negatives (real): {np.sum(y_true==0)}")
print(f"Threshold: {threshold}, Optimal threshold platt: {optimal_threshold_platt}")

for prob_key in ["probs_raw", "probs_cal", "probs_temp_unadjusted", "probs_platt_shifted"]:
    if prob_key in d:
        p = np.array(d[prob_key])
        auc = roc_auc_score(y_true, p)
        print(f"\n--- {prob_key} ---")
        print(f"AUC: {auc:.4f}")
        for th in [0.5, threshold, optimal_threshold_platt]:
            if th is not None:
                preds = (p >= th).astype(int)
                bacc = balanced_accuracy_score(y_true, preds)
                acc = accuracy_score(y_true, preds)
                f1 = f1_score(y_true, preds)
                print(f"  At th={th:.4f}: BAcc={bacc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
