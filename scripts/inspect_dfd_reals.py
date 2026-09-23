import os
import json

dir_set = set(d for d in os.listdir("deepfake_crops_512/real") if len(d) == 5 and d.isdigit())
print(f"5-digit real count: {len(dir_set)}")

with open("deepfake_crops_512/splits.json", "r") as f:
    splits = json.load(f)

for split_name, entries in splits.items():
    in_split = 0
    for e in entries:
        parts = e[0].replace("\\", "/").split("/")
        if len(parts) > 1 and parts[1] in dir_set:
            in_split += 1
    print(f"5-digit reals in split '{split_name}': {in_split}")

with open("deepfake_crops_512/splits_backup_original.json", "r") as f:
    splits_orig = json.load(f)

for split_name, entries in splits_orig.items():
    in_split = 0
    for e in entries:
        parts = e[0].replace("\\", "/").split("/")
        if len(parts) > 1 and parts[1] in dir_set:
            in_split += 1
    print(f"5-digit reals in split_orig '{split_name}': {in_split}")
