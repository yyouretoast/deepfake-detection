import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

with open("test_predictions.json", "r") as f:
    d = json.load(f)

y_true = np.array(d["labels"])
for key in ["probs_raw", "probs_cal", "probs_platt_shifted"]:
    p = np.array(d[key])
    preds = (p >= 0.4200).astype(int)
    cm = confusion_matrix(y_true, preds)
    print(f"\n--- {key} at th=0.4200 ---")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bacc = (rec + spec) / 2
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, Specificity: {spec:.4f}, Bal Acc: {bacc:.4f}, Fake F1: {f1:.4f}")
