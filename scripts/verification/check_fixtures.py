import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.audit_fixture_consistency import compute_auc

with open("results/release1_run/test_predictions.json") as f:
    d = json.load(f)
print("== test_predictions.json ==")
for k, v in d.items():
    if isinstance(v, list):
        print(f"{k}: list len {len(v)}")
    else:
        print(f"{k}: {v}")
recomputed = compute_auc(d["probs_cal"], d["labels"])
print("RECOMPUTED AUC from arrays:", round(recomputed, 6))
print("label counts:", collections.Counter(d["labels"]))
# sanity: sample prob ranges
probs = d["probs_cal"]
print("prob min/max:", min(probs), max(probs))

with open("results/release1_run/temporal_test_predictions.json") as f:
    t = json.load(f)
print("\n== temporal_test_predictions.json ==")
for k, v in t.items():
    if isinstance(v, list):
        print(f"{k}: list len {len(v)}")
    else:
        print(f"{k}: {v}")
rec_temporal = compute_auc(t["probs_temporal"], t["labels"])
rec_naive = compute_auc(t["probs_naive_avg"], t["labels"])
print("RECOMPUTED Bi-GRU AUC:", round(rec_temporal, 6))
print("RECOMPUTED naive-avg AUC:", round(rec_naive, 6))
print("label counts:", collections.Counter(t["labels"]))
for k in ("probs_max_pool", "probs_spatial_frame"):
    if k in t:
        print(f"recomputed {k} AUC:", round(compute_auc(t[k], t["labels"]), 6))
