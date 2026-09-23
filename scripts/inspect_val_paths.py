import json

with open("deepfake_crops_512/splits_backup_original.json", "r") as f:
    orig = json.load(f)

val = orig["val"]
print(f"Total in orig val: {len(val)}")
print("First 5 val samples:")
for v in val[:5]:
    print(v)

# Check if any have "real" or "fake"
print("Last 5 val samples:")
for v in val[-5:]:
    print(v)

# Let's count patterns
import collections
top_dirs = collections.Counter([v[0].replace('\\', '/').split('/')[0] for v in val])
print("Top dirs:", top_dirs)
sub_dirs = collections.Counter([v[0].replace('\\', '/').split('/')[1] for v in val if len(v[0].replace('\\', '/').split('/')) > 1])
print("Sub dirs sample (first 20):", list(sub_dirs.items())[:20])
