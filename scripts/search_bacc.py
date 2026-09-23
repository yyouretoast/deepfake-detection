import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score

with open("test_predictions.json", "r") as f:
    d = json.load(f)

y_true = np.array(d["labels"])
for key in ["probs_raw", "probs_cal", "probs_platt_shifted"]:
    p = np.array(d[key])
    max_bacc = 0.0
    best_th = 0.0
    matches = []
    for th in np.linspace(0.01, 0.99, 1000):
        bacc = balanced_accuracy_score(y_true, (p >= th).astype(int))
        if bacc > max_bacc:
            max_bacc = bacc
            best_th = th
        if abs(bacc - 0.8122) < 0.001:
            matches.append((th, bacc))
    print(f"{key}: Max BAcc = {max_bacc:.4f} at th={best_th:.4f}")
    if matches:
        print(f"  Matches for 0.8122 in {key}:", matches[:5])
