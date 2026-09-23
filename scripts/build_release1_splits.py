import json
import os
import sys
import re
import random
sys.path.insert(0, ".")
from src.dataset.loader import extract_identities, dedupe_split, perform_graph_split

random.seed(42)

# Load current splits to preserve the verified Celeb-DF sets
with open("deepfake_crops_512/splits.json", "r") as f:
    current_splits = json.load(f)

with open("deepfake_crops_512/manifest.json", "r") as f:
    manifest = json.load(f)

# Build map of path -> label
manifest_map = {item["path"]: item["label"] for item in manifest}

# All unique crops on disk per folder
folder_to_crops = {}
for p, l in manifest_map.items():
    parts = p.replace("\\", "/").split("/")
    folder = f"{parts[0]}/{parts[1]}"
    if folder not in folder_to_crops:
        folder_to_crops[folder] = []
    folder_to_crops[folder].append([p, l])

# Deduplicate crops within each folder
for folder in folder_to_crops:
    seen = set()
    deduped = []
    for entry in folder_to_crops[folder]:
        if entry[0] not in seen:
            seen.add(entry[0])
            deduped.append(entry)
    folder_to_crops[folder] = deduped

# 1. Classify folders
dfd_fake_folders = [f for f in folder_to_crops if f.startswith("fake/") and re.match(r"^\d{2}_\d{2}__", f.split("/")[1])]
dfd_real_folders = [f for f in folder_to_crops if f.startswith("real/") and len(f.split("/")[1]) == 5 and f.split("/")[1].isdigit()]

celeb_fake_folders = [f for f in folder_to_crops if f.startswith("fake/") and "id" in f.split("/")[1]]
celeb_real_folders = [f for f in folder_to_crops if f.startswith("real/") and "id" in f.split("/")[1]]

ff_fake_folders = [f for f in folder_to_crops if f.startswith("fake/") and re.match(r"^\d{3}_\d{3}$", f.split("/")[1])]
ff_real_folders = [f for f in folder_to_crops if f.startswith("real/") and re.match(r"^\d{3}$", f.split("/")[1])]

print(f"DFD: {len(dfd_fake_folders)} fakes, {len(dfd_real_folders)} reals")
print(f"Celeb: {len(celeb_fake_folders)} fakes, {len(celeb_real_folders)} reals")
print(f"FF++: {len(ff_fake_folders)} fakes, {len(ff_real_folders)} reals")

# 2. Celeb-DF splits: preserve verified sets from current_splits
current_train_folders = set(e[0].replace('\\', '/').split('/')[0] + '/' + e[0].replace('\\', '/').split('/')[1] for e in current_splits["train"])
current_val_folders = set(e[0].replace('\\', '/').split('/')[0] + '/' + e[0].replace('\\', '/').split('/')[1] for e in current_splits["val"])
current_test_folders = set(e[0].replace('\\', '/').split('/')[0] + '/' + e[0].replace('\\', '/').split('/')[1] for e in current_splits["test"])

celeb_train = [f for f in (celeb_fake_folders + celeb_real_folders) if f in current_train_folders]
celeb_val = [f for f in (celeb_fake_folders + celeb_real_folders) if f in current_val_folders]
celeb_test = [f for f in (celeb_fake_folders + celeb_real_folders) if f in current_test_folders]

print(f"Celeb folders: train={len(celeb_train)}, val={len(celeb_val)}, test={len(celeb_test)}")

# 3. DFD splits: 100% held out to test!
dfd_train = []
dfd_val = []
dfd_test = dfd_fake_folders + dfd_real_folders
print(f"DFD folders: train=0, val=0, test={len(dfd_test)}")

# 4. FF++ splits:
# In FF++, we have 1000 pairs in fake and 1000 single videos in real.
# Construct graph nodes based on actor ID (000 to 999).
# An actor pair 'xxx_yyy' connects actor xxx and actor yyy.
# Group connected components of FF++ actors.
import networkx as nx

G = nx.Graph()
for f in ff_fake_folders:
    p = f.split("/")[1]
    m = re.match(r"^(\d{3})_(\d{3})$", p)
    if m:
        u, v = m.group(1), m.group(2)
        G.add_edge(u, v)

for f in ff_real_folders:
    p = f.split("/")[1]
    if re.match(r"^\d{3}$", p):
        G.add_node(p)

components = list(nx.connected_components(G))
print(f"FF++ graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(components)} connected components.")

# Partition components randomly into train (~72%), val (~14%), test (~14%)
random.shuffle(components)
total_nodes = G.number_of_nodes()
train_target = int(0.72 * total_nodes)
val_target = int(0.14 * total_nodes)

ff_train_actors, ff_val_actors, ff_test_actors = set(), set(), set()
cur_train = 0
cur_val = 0

for comp in components:
    if cur_train + len(comp) <= train_target or (cur_train < train_target and cur_val >= val_target):
        ff_train_actors.update(comp)
        cur_train += len(comp)
    elif cur_val + len(comp) <= val_target or cur_val < val_target:
        ff_val_actors.update(comp)
        cur_val += len(comp)
    else:
        ff_test_actors.update(comp)

print(f"FF++ actor partition: Train={len(ff_train_actors)}, Val={len(ff_val_actors)}, Test={len(ff_test_actors)}")
assert len(ff_train_actors & ff_val_actors) == 0
assert len(ff_train_actors & ff_test_actors) == 0
assert len(ff_val_actors & ff_test_actors) == 0

# Map FF++ folders to train, val, test
ff_train_folders, ff_val_folders, ff_test_folders = [], [], []

for f in ff_fake_folders:
    m = re.match(r"^fake/(\d{3})_(\d{3})$", f)
    u, v = m.group(1), m.group(2)
    if u in ff_train_actors and v in ff_train_actors:
        ff_train_folders.append(f)
    elif u in ff_val_actors and v in ff_val_actors:
        ff_val_folders.append(f)
    elif u in ff_test_actors and v in ff_test_actors:
        ff_test_folders.append(f)

for f in ff_real_folders:
    u = f.split("/")[1]
    if u in ff_train_actors:
        ff_train_folders.append(f)
    elif u in ff_val_actors:
        ff_val_folders.append(f)
    elif u in ff_test_actors:
        ff_test_folders.append(f)

print(f"FF++ folders: Train={len(ff_train_folders)}, Val={len(ff_val_folders)}, Test={len(ff_test_folders)}")

# 5. Assemble final splits
train_crops = []
for f in celeb_train + ff_train_folders:
    train_crops.extend(folder_to_crops[f])

val_crops = []
for f in celeb_val + ff_val_folders:
    val_crops.extend(folder_to_crops[f])

test_crops = []
for f in celeb_test + ff_test_folders + dfd_test:
    test_crops.extend(folder_to_crops[f])

train_crops = dedupe_split(train_crops)
val_crops = dedupe_split(val_crops)
test_crops = dedupe_split(test_crops)

print(f"\nFinal Release 1 crop counts:")
print(f"Train: {len(train_crops)} (Reals: {sum(1 for c in train_crops if c[1]==0)}, Fakes: {sum(1 for c in train_crops if c[1]==1)})")
print(f"Val: {len(val_crops)} (Reals: {sum(1 for c in val_crops if c[1]==0)}, Fakes: {sum(1 for c in val_crops if c[1]==1)})")
print(f"Test: {len(test_crops)} (Reals: {sum(1 for c in test_crops if c[1]==0)}, Fakes: {sum(1 for c in test_crops if c[1]==1)})")

# 6. Verify zero identity overlap across all splits
def get_actor_ids(crops):
    actors = set()
    for c in crops:
        p = c[0].replace("\\", "/")
        parts = p.split("/")
        name = parts[1]
        a1, a2 = extract_identities(name)
        actors.add(a1)
        actors.add(a2)
    return actors

train_actors = get_actor_ids(train_crops)
val_actors = get_actor_ids(val_crops)
test_actors = get_actor_ids(test_crops)

print("\n--- Leakage Verification ---")
print(f"Train actors: {len(train_actors)}")
print(f"Val actors: {len(val_actors)}")
print(f"Test actors: {len(test_actors)}")

train_val_overlap = train_actors & val_actors
train_test_overlap = train_actors & test_actors
val_test_overlap = val_actors & test_actors

print(f"Train & Val overlap: {len(train_val_overlap)} -> {train_val_overlap}")
print(f"Train & Test overlap: {len(train_test_overlap)} -> {train_test_overlap}")
print(f"Val & Test overlap: {len(val_test_overlap)} -> {val_test_overlap}")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("SUCCESS: EXACTLY 0 IDENTITY LEAKAGE ACROSS ALL THREE SPLITS!")
    new_splits = {
        "train": train_crops,
        "val": val_crops,
        "test": test_crops
    }
    with open("deepfake_crops_512/splits.json", "w") as f:
        json.dump(new_splits, f)
    print("Saved clean splits to deepfake_crops_512/splits.json")
else:
    print("ERROR: Overlap detected, did not save!")
