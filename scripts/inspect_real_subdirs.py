import os
import re

dirs = sorted(os.listdir("deepfake_crops_512/real"))
print(f"Total dirs in deepfake_crops_512/real: {len(dirs)}")

len3_dirs = [d for d in dirs if len(d) == 3 and d.isdigit()]
len5_dirs = [d for d in dirs if len(d) == 5 and d.isdigit()]
celeb_dirs = [d for d in dirs if d.startswith("id")]
other_dirs = [d for d in dirs if d not in len3_dirs and d not in len5_dirs and d not in celeb_dirs]

print(f"3-digit numeric dirs (FF++ original sequences 000-999): {len(len3_dirs)}")
print(f"5-digit numeric dirs (e.g. 00000-00048): {len(len5_dirs)} -> sample: {len5_dirs[:10]}")
print(f"Celeb dirs (id...): {len(celeb_dirs)} -> sample: {celeb_dirs[:10]}")
print(f"Other dirs: {len(other_dirs)} -> sample: {other_dirs[:10]}")
