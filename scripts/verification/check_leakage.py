import collections
import csv


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

train, val, test = load("splits/train_split.csv"), load("splits/val_split.csv"), load("splits/test_split.csv")

# ---- Celeb-DF identities: union over fake pair ids AND real video ids ----
def celeb_ids_all(rows):
    ids = collections.Counter()
    vids = collections.defaultdict(set)
    for r in rows:
        if r["domain"] != "celeb_df_v2":
            continue
        vid = r["video_id"]
        vids[r["label"]].add(vid)
        m = [int(x) for x in __import__("re").findall(r"id(\d+)", vid)]
        for x in m:
            ids[x] += 0
    return set(ids), vids

def celeb_real_id_format(rows):
    forms = set()
    for r in rows:
        if r["domain"] == "celeb_df_v2" and r["label"] == "0":
            forms.add(r["video_id"])
            if len(forms) > 8:
                break
    return sorted(forms)

print("sample real celeb video ids (train):", celeb_real_id_format(train))

def celeb_video_ids(rows):
    fakes, reals = set(), set()
    for r in rows:
        if r["domain"] != "celeb_df_v2":
            continue
        (fakes if r["label"] == "1" else reals).add(r["video_id"])
    return fakes, reals

tr_f, tr_r = celeb_video_ids(train)
va_f, va_r = celeb_video_ids(val)
te_f, te_r = celeb_video_ids(test)
print("celeb video counts: train fake", len(tr_f), "real", len(tr_r), "| val fake", len(va_f), "real", len(va_r), "| test fake", len(te_f), "real", len(te_r))

import re as _re


def id_set(pair_or_real_ids):
    s = set()
    for v in pair_or_real_ids:
        for tok in _re.findall(r"id(\d+)", v):
            s.add(int(tok))
    return s

tr_ids = id_set(tr_f | tr_r)
va_ids = id_set(va_f | va_r)
te_ids = id_set(te_f | te_r)
print("celeb identity totals: train", len(tr_ids), "val", len(va_ids), "test", len(te_ids), "sum", len(tr_ids) + len(va_ids) + len(te_ids))
print("celeb overlaps:", tr_ids & va_ids, tr_ids & te_ids, va_ids & te_ids)

# ---- FF++ actors across splits: pair i_j -> actors {i, j} (and real video i -> actor i) ----
def ffpp_actors(rows):
    s = set()
    vids = collections.defaultdict(set)
    for r in rows:
        if r["domain"] != "faceforensics":
            continue
        vid = r["video_id"]
        parts = vid.split("_")
        for p in parts:
            s.add(int(p))
        vids[r["label"]].add(vid)
    return s, vids

tr_a, tr_v = ffpp_actors(train)
va_a, va_v = ffpp_actors(val)
te_a, te_v = ffpp_actors(test)
print("\nFF++ unique actor tokens: train", len(tr_a), "val", len(va_a), "test", len(te_a))
print("FF++ actor overlap train&val:", len(tr_a & va_a), "train&test:", len(tr_a & te_a), "val&test:", len(va_a & te_a))
print("FF++ sample train&test shared actors:", sorted(tr_a & te_a)[:10])
# how many shared actors appear as source in test fake videos?
te_fake_pairs = [v for v in te_v["1"]]
src_in_test_fakes = {int(v.split("_")[0]) for v in te_fake_pairs}
tgt_in_test_fakes = {int(v.split("_")[1]) for v in te_fake_pairs}
print("shared actors serving as TEST FAKE source:", len(tr_a & src_in_test_fakes))
print("shared actors serving as TEST FAKE target:", len(tr_a & tgt_in_test_fakes))
# train fake videos whose target == a test-fake source actor etc. — measure blast radius:
tr_fake_pairs = [v for v in tr_v["1"]]
tr_src = {int(v.split("_")[0]) for v in tr_fake_pairs}
tr_tgt = {int(v.split("_")[1]) for v in tr_fake_pairs}
print("train fake sources that are also test fake sources:", len(tr_src & src_in_test_fakes), "of", len(src_in_test_fakes))
print("train fake targets that are also test fake targets:", len(tr_tgt & tgt_in_test_fakes), "of", len(tgt_in_test_fakes))
print("train fake TARGET == test fake SOURCE:", len(tr_tgt & src_in_test_fakes), " (identity as target in train, source in test)")
print("train fake SOURCE == test fake TARGET:", len(tr_src & tgt_in_test_fakes))

# ---- DFD ids ----
dfd_f = {r["video_id"] for r in test if r["domain"] == "google_dfd"}
print("\nDFD test fakes unique vids:", len(dfd_f))
yt = {r["video_id"] for r in test if r["domain"] == "youtube"}
print("YouTube reals unique vids:", len(yt))

# ---- frames per video ----
fc = collections.Counter(r["relative_path"].split("/frame")[0] for r in test if r["domain"] == "faceforensics")
counts = collections.Counter(fc.values())
print("\nFF++ frames-per-video histogram (test):", dict(counts))
cz = collections.Counter(r["relative_path"].split("/frame")[0] for r in test if r["domain"] == "google_dfd")
print("DFD frames-per-video histogram (test):", dict(collections.Counter(cz.values())))
