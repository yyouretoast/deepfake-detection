import json
import os
import re
from collections import defaultdict, Counter

with open("deepfake_crops_512/splits.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

table = defaultdict(lambda: defaultdict(int))
folders = defaultdict(lambda: defaultdict(set))
actors = defaultdict(lambda: defaultdict(set))

def extract_actors(name):
    clean = name.replace(".webp", "")
    if re.match(r"^\d{3}_\d{3}$", clean):
        parts = clean.split("_")
        return {parts[0], parts[1]}
    if re.match(r"^\d{3}$", clean):
        return {clean}
    if "id" in clean:
        m = re.findall(r"id\d+", clean)
        return set(m) if m else {clean}
    if "__" in clean or len(clean) == 5 and clean.isdigit():
        return {clean}
    return {clean}

for split_name in ["train", "val", "test"]:
    for p, lbl in splits[split_name]:
        norm = p.replace("\\", "/")
        parts = norm.split("/")
        folder = f"{parts[0]}/{parts[1]}"
        sub_name = parts[1]
        cls = "Fake" if lbl == 1 else "Real"
        
        if "__" in norm or "/00" in norm and len(sub_name) == 5:
            src = "DFD"
        elif "id" in sub_name or "celeb" in norm.lower():
            src = "Celeb-DF"
        else:
            src = "FF++"
            
        table[split_name][(src, cls)] += 1
        folders[split_name][(src, cls)].add(folder)
        actors[split_name][src].update(extract_actors(sub_name))

print("=" * 70)
print(f"{'Split':8} | {'Source':10} | {'Class':5} | {'Crops':7} | {'Folders (Videos)':16} | {'Actors':6}")
print("-" * 70)
for split_name in ["train", "val", "test"]:
    for src in ["FF++", "Celeb-DF", "DFD"]:
        for cls in ["Real", "Fake"]:
            count = table[split_name][(src, cls)]
            n_f = len(folders[split_name][(src, cls)])
            n_act = len(actors[split_name][src])
            if count > 0:
                print(f"{split_name:8} | {src:10} | {cls:5} | {count:7d} | {n_f:16d} | {n_act:6d}")
    sub_crops = sum(table[split_name].values())
    sub_folders = sum(len(s) for s in folders[split_name].values())
    print("-" * 70)
    print(f"SUBTOTAL {split_name.upper():5} : {sub_crops:7d} crops | {sub_folders:4d} folders")
    print("-" * 70)

# Check video folder overlap:
train_folders = set.union(*folders["train"].values())
val_folders = set.union(*folders["val"].values())
test_folders = set.union(*folders["test"].values())

print("\n--- VIDEO FOLDER DISJOINTNESS CHECK ---")
print("Folder overlap Train & Val :", len(train_folders & val_folders))
print("Folder overlap Train & Test:", len(train_folders & test_folders))
print("Folder overlap Val & Test  :", len(val_folders & test_folders))

# Check actor overlap per source:
print("\n--- ACTOR IDENTITY DISJOINTNESS CHECK ---")
for src in ["FF++", "Celeb-DF", "DFD"]:
    tr_act = actors["train"].get(src, set())
    va_act = actors["val"].get(src, set())
    te_act = actors["test"].get(src, set())
    print(f"Source [{src:8}]: Train={len(tr_act)}, Val={len(va_act)}, Test={len(te_act)}")
    print(f"  Overlap Train & Val : {len(tr_act & va_act)}")
    print(f"  Overlap Train & Test: {len(tr_act & te_act)}")
    print(f"  Overlap Val & Test  : {len(va_act & te_act)}")
