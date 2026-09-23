import json
import os
import sys
import re
sys.path.insert(0, ".")
from src.dataset.loader import extract_identities

# Load manifest
with open("deepfake_crops_512/manifest.json", "r") as f:
    manifest = json.load(f)

# Group all available crops by their sequence folder
manifest_dict = {}
for item in manifest:
    manifest_dict[item["path"]] = item["label"]

# Inspect all video folders
folders = set()
for path in manifest_dict.keys():
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        folders.add(f"{parts[0]}/{parts[1]}")

folders = sorted(list(folders))
print(f"Total unique video sequence folders in manifest: {len(folders)}")

# Classify folders:
dfd_fakes = [f for f in folders if f.startswith("fake/") and re.match(r"^\d{2}_\d{2}__", f.split("/")[1])]
dfd_reals = [f for f in folders if f.startswith("real/") and len(f.split("/")[1]) == 5 and f.split("/")[1].isdigit()]

celeb_fakes = [f for f in folders if f.startswith("fake/") and ("id" in f.split("/")[1])]
celeb_reals = [f for f in folders if f.startswith("real/") and ("id" in f.split("/")[1])]

ff_fakes = [f for f in folders if f.startswith("fake/") and re.match(r"^\d{3}_\d{3}$", f.split("/")[1])]
ff_reals = [f for f in folders if f.startswith("real/") and re.match(r"^\d{3}$", f.split("/")[1])]

print(f"DFD Fakes: {len(dfd_fakes)}, DFD Reals: {len(dfd_reals)}")
print(f"Celeb Fakes: {len(celeb_fakes)}, Celeb Reals: {len(celeb_reals)}")
print(f"FF++ Fakes: {len(ff_fakes)}, FF++ Reals: {len(ff_reals)}")
