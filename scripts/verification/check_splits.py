import collections
import csv
import re


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows

train, val, test = load("splits/train_split.csv"), load("splits/val_split.csv"), load("splits/test_split.csv")
print("rows: train", len(train), "val", len(val), "test", len(test))

# label counts per split
for name, rows in (("train", train), ("val", val), ("test", test)):
    c = collections.Counter(r["label"] for r in rows)
    doms = collections.Counter(r["domain"] for r in rows)
    gens = collections.Counter(r["generator"] for r in rows)
    print(f"\n== {name} == labels: {dict(c)}  domains: {dict(doms)}")
    print("   generators:", dict(gens))

# 1) no duplicate crop paths across splits (pair-level leakage probe)
paths = {}
dup_total = 0
for name, rows in (("train", train), ("val", val), ("test", test)):
    for r in rows:
        p = r["relative_path"]
        if p in paths and paths[p] != name:
            dup_total += 1
        paths[p] = name
print("\n[1] crops appearing in >1 split:", dup_total)

# 2) video_id overlap across splits per domain
def vids(rows):
    out = collections.defaultdict(set)
    for r in rows:
        out[r["domain"]].add(r["video_id"])
    return out

tv, vv, te = vids(train), vids(val), vids(test)
for dom in sorted(set(tv) | set(vv) | set(te)):
    print(f"[2] {dom}: train/val inter={len(tv[dom] & vv[dom])}, train/test inter={len(tv[dom] & te[dom])}, val/test inter={len(vv[dom] & te[dom])}")

# 3) FF++ pair-level disjointness: relative_path tokens like 'fake/..._0000/...' -> the video_id.
#    For Celeb-DF the video ids are idNN_idNN_XXXX; identities = the two id tokens.
def celeb_ids(rows):
    per_domain = collections.defaultdict(set)
    for r in rows:
        if r["domain"] != "celeb_df_v2":
            continue
        vid = r["video_id"]  # e.g. id39_id40_0000
        m = re.match(r"id(\d+)_id(\d+)_", vid)
        if m:
            per_domain["celeb_id"].add(int(m.group(1)))
            per_domain["celeb_id"].add(int(m.group(2)))
    return per_domain

ct, cv, ce = celeb_ids(train), celeb_ids(val), celeb_ids(test)
print("\n[3] Celeb-DF identity counts: train", len(ct["celeb_id"]), "val", len(cv["celeb_id"]), "test", len(ce["celeb_id"]))
print("    train&val overlap:", ct["celeb_id"] & cv["celeb_id"])
print("    train&test overlap:", ct["celeb_id"] & ce["celeb_id"])
print("    val&test overlap:", cv["celeb_id"] & ce["celeb_id"])

# 4) FF++ pairs: FF++ video ids are pair-level ids (e.g. ...). Check distinct pattern
def ffpp_videos(rows):
    out = collections.defaultdict(set)
    for r in rows:
        if r["domain"] != "faceforensics":
            continue
        out[r["video_id"]].add(r["relative_path"])
    return out

ft, fv, fe = ffpp_videos(train), ffpp_videos(val), ffpp_videos(test)
print("\n[4] FF++ unique video ids: train", len(ft), "val", len(fv), "test", len(fe))
print("    train&test FF++ video overlap:", len(ft & fe), "train&val:", len(ft & fv), "val&test:", len(fv & fe))
over = (ft & fe)
print("    example overlapping FF++ ids (first 5):", sorted(over)[:5])
# generator counts per FF++ test
gens_test = collections.Counter(r["generator"] for r in test if r["domain"] == "faceforensics")
print("    FF++ test generator counts:", dict(gens_test))
gens_train = collections.Counter(r["generator"] for r in train if r["domain"] == "faceforensics")
print("    FF++ train generator counts:", dict(gens_train))
gens_val = collections.Counter(r["generator"] for r in val if r["domain"] == "faceforensics")
print("    FF++ val generator counts:", dict(gens_val))
