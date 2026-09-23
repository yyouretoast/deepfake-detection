import os, json, re

with open('deepfake_crops_512/splits.json') as f:
    splits = json.load(f)

test_items = splits['test']
reals = [item for item in test_items if item[1] == 0]

ffpp_reals = []
youtube_reals = []
celeb_reals = []
other_reals = []

for path, label in reals:
    norm_path = path.replace('\\', '/').lower()
    folder = norm_path.split('/')[1]
    if re.match(r'^\d{3}$', folder):
        ffpp_reals.append((norm_path, folder))
    elif re.match(r'^\d{5}$', folder):
        youtube_reals.append((norm_path, folder))
    elif folder.startswith('id'):
        celeb_reals.append((norm_path, folder))
    else:
        other_reals.append((norm_path, folder))

print(f"Total Test Reals: {len(reals)}")
print(f"FF++ Reals (000-999): {len(ffpp_reals)} crops, {len(set(f for _, f in ffpp_reals))} folders")
print(f"5-digit Reals (00000-00299): {len(youtube_reals)} crops, {len(set(f for _, f in youtube_reals))} folders")
print(f"Celeb-real (idXX): {len(celeb_reals)} crops, {len(set(f for _, f in celeb_reals))} folders")
print(f"Other Reals: {len(other_reals)}")

print("\n--- Also check Train and Val Reals ---")
for split_name in ['train', 'val']:
    items = splits[split_name]
    r_items = [it for it in items if it[1] == 0]
    ff = len([it for it in r_items if re.match(r'^\d{3}$', it[0].replace('\\', '/').split('/')[1])])
    yt = len([it for it in r_items if re.match(r'^\d{5}$', it[0].replace('\\', '/').split('/')[1])])
    cl = len([it for it in r_items if it[0].replace('\\', '/').split('/')[1].startswith('id')])
    print(f"{split_name.upper()} Reals: Total={len(r_items)} | FF++={ff} | 5-digit(00xxx)={yt} | Celeb-real(idXX)={cl}")
