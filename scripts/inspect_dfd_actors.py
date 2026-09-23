import os
import re
import collections

fake_dirs = os.listdir("deepfake_crops_512/fake")
dfd_fakes = [d for d in fake_dirs if re.match(r"^\d{2}_\d{2}__", d)]
print(f"Total DFD fake dirs: {len(dfd_fakes)}")

actors = collections.Counter()
pairs = collections.Counter()
for d in dfd_fakes:
    m = re.match(r"^(\d{2})_(\d{2})__", d)
    if m:
        a1, a2 = m.group(1), m.group(2)
        actors[a1] += 1
        actors[a2] += 1
        pairs[f"{a1}_{a2}"] += 1

print(f"Unique actors in DFD fakes: {len(actors)} -> {sorted(list(actors.keys()))}")
print(f"Total actor pairs in DFD fakes: {len(pairs)}")
