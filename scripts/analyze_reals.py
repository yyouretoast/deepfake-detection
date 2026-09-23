import json
import collections
import re

with open("deepfake_crops_512/manifest.json", "r") as f:
    manifest = json.load(f)

real_paths = [item["path"] for item in manifest if item.get("label") == 0]
others = []
for p in real_paths:
    if not re.search(r"id\d+", p) and not re.search(r"[\\/]\d{3}[\\/]", p) and not p.startswith("real/0"):
        # check what doesn't match
        pass
    else:
        continue
    others.append(p)

print(f"Total unmatched: {len(others)}")
for p in others[:20]:
    print("Unmatched real:", p)

# Group unmatched by top-level dir or video_id
unmatched_vids = set(p.split("/")[1] if "/" in p else p for p in others)
print(f"Unique video IDs in unmatched ({len(unmatched_vids)}):", sorted(list(unmatched_vids))[:30])
