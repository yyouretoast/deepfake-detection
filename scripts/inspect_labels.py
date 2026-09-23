import json
import collections

with open("deepfake_crops_512/splits_backup_original.json", "r") as f:
    splits_orig = json.load(f)

for sname, entries in splits_orig.items():
    labels = collections.Counter([e[1] for e in entries])
    print(f"Original {sname}: Total={len(entries)}, Reals={labels[0]}, Fakes={labels[1]}")

with open("deepfake_crops_512/splits.json", "r") as f:
    splits_curr = json.load(f)

for sname, entries in splits_curr.items():
    labels = collections.Counter([e[1] for e in entries])
    print(f"Current {sname}: Total={len(entries)}, Reals={labels[0]}, Fakes={labels[1]}")
