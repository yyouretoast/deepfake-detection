"""
Diagnostic script to inspect exact path formatting in Kaggle splits.json
"""
import os
import json
from scripts.train_dual_stream_ddp import find_dataset_root, dedupe_split

def inspect():
    root = find_dataset_root()
    splits_path = os.path.join(root, 'splits.json')
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    train = dedupe_split(splits['train'])
    print(f"Total Train Samples: {len(train)}")
    print("\n--- FIRST 20 SAMPLE PATHS IN SPLITS.JSON ---")
    for i, s in enumerate(train[:20]):
        print(f"[{i+1}] Path: {s[0]} | Label: {s[1]}")

    print("\n--- SEARCHING FOR KEYWORDS IN PATHS ---")
    keywords = ["neural", "nt", "deepfake", "df", "face2face", "f2f", "faceswap", "fs", "celeb", "fake", "real"]
    for kw in keywords:
        count = sum(1 for s in train if kw in s[0].lower())
        print(f"  Keyword '{kw}': {count} matching samples")

if __name__ == '__main__':
    inspect()
