import json
import collections
import re

with open("deepfake_crops_512/splits.json", "r") as f:
    splits = json.load(f)

def categorize(path):
    p = path.replace("\\", "/")
    if "01_02__" in p or re.search(r"\d{2}_\d{2}__", p):
        return "DFD_fake"
    if re.search(r"real/0\d{4}/", p):
        return "DFD_real"
    if "celeb" in p.lower() or re.search(r"[\\/_\-]id\d+", p):
        if "/fake/" in p:
            return "Celeb_fake"
        else:
            return "Celeb_real"
    if re.search(r"real/\d{3}/", p):
        return "FF_real"
    if "/fake/" in p and re.search(r"\d{3}_\d{3}", p):
        return "FF_fake"
    return "Other"

for sname, entries in splits.items():
    counts = collections.Counter()
    for e in entries:
        counts[categorize(e[0])] += 1
    print(f"\n--- Split: {sname} (Total: {len(entries)}) ---")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
