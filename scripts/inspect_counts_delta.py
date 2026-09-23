import os, json, re

with open('deepfake_crops_512/splits.json') as f:
    splits = json.load(f)

print("--- Inspecting Test Split ---")
test_items = splits['test']
print(f"Total test items: {len(test_items)}")

# Check classification of test items
dfd_fake = []
celeb_fake = []
ffpp_fake = []
reals = []

for path, label in test_items:
    norm_path = path.replace('\\', '/').lower()
    folder = norm_path.split('/')[1]
    if label == 0:
        reals.append((norm_path, folder))
    else:
        if '__' in folder:
            dfd_fake.append((norm_path, folder))
        elif re.match(r'^\d{3}_\d{3}$', folder):
            ffpp_fake.append((norm_path, folder))
        elif 'id' in folder:
            celeb_fake.append((norm_path, folder))
        else:
            print(f"Unknown fake: {folder}")

print(f"Test Reals: {len(reals)}")
print(f"Test FF++ Fakes: {len(ffpp_fake)}")
print(f"Test DFD Fakes (__ in folder): {len(dfd_fake)}")
print(f"Test Celeb-DF Fakes (id in folder): {len(celeb_fake)}")

# Unique video folders
dfd_vids = set(f for _, f in dfd_fake)
celeb_vids = set(f for _, f in celeb_fake)
ffpp_vids = set(f for _, f in ffpp_fake)
print(f"DFD Fake Unique Videos: {len(dfd_vids)}")
print(f"Celeb-DF Fake Unique Videos: {len(celeb_vids)}")
print(f"FF++ Fake Unique Videos: {len(ffpp_vids)}")

# Check why previous script counted 7,744 vs 11,992
# Check if some DFD fakes have 'id' in the folder name
dfd_with_id = [f for f in dfd_vids if 'id' in f.lower()]
print(f"DFD folders containing 'id': {len(dfd_with_id)}")
if dfd_with_id:
    print(f"Sample DFD folders with 'id': {dfd_with_id[:5]}")
    # Count crops in those folders
    crops_in_dfd_with_id = sum(1 for _, f in dfd_fake if f in dfd_with_id)
    print(f"Crops in DFD folders with 'id': {crops_in_dfd_with_id}")
