import json

with open("deepfake_crops_512/manifest.json", "r") as f:
    manifest = json.load(f)

# Find samples for 5-digit reals
samples_5digit = [item for item in manifest if "real/00" in item["path"] and len(item["path"].split("/")[1]) == 5]
print("5-digit real sample (first 5):")
for s in samples_5digit[:5]:
    print(s)

# Also find DFD fake sample
samples_dfd_fake = [item for item in manifest if "01_02__" in item["path"]]
print("DFD fake sample (first 5):")
for s in samples_dfd_fake[:5]:
    print(s)
