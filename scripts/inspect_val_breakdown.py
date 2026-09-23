import json
import collections
import re

with open("deepfake_crops_512/splits_backup_original.json", "r") as f:
    orig = json.load(f)

val_entries = orig["val"]
def categorize(p):
    if "01_02__" in p or re.search(r"\d{2}_\d{2}__", p): return "DFD_fake"
    if re.search(r"real/0\d{4}/", p): return "DFD_real"
    if "id" in p and "/fake/" in p: return "Celeb_fake"
    if "id" in p and "/real/" in p: return "Celeb_real"
    if "/real/" in p: return "FF_real"
    if "/fake/" in p: return "FF_fake"
    return "Other"

counts = collections.Counter([categorize(e[0]) for e in val_entries])
print("Original val breakdown:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
