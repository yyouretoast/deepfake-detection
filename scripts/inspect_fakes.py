"""
Diagnostic script to inspect fake sample path structures in Kaggle splits.json
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
from scripts.train_dual_stream_ddp import find_dataset_root, dedupe_split

def inspect_fakes():
    root = find_dataset_root()
    splits_path = os.path.join(root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    train = dedupe_split(splits['train'])
    fakes = [s for s in train if s[1] == 1]
    
    print(f"Total Fake Samples: {len(fakes)}")
    print("\n--- FIRST 30 FAKE SAMPLE PATHS IN SPLITS.JSON ---")
    for i, s in enumerate(fakes[:30]):
        print(f"[{i+1}] {s[0]}")

    print("\n--- FAKE PATH DIRECTORY STRUCTURE (UNIQUE FOLDERS) ---")
    dirs = set()
    for s in fakes:
        parts = s[0].replace("\\", "/").split("/")
        if len(parts) > 1:
            dirs.add("/".join(parts[:-1]))
    
    for d in sorted(list(dirs))[:30]:
        print(f"  Folder: {d}")

if __name__ == '__main__':
    inspect_fakes()
