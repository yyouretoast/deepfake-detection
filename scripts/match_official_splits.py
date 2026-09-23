import json
import re

# Read the content files fetched
with open("C:/Users/Yassin/.gemini/antigravity/brain/ec9036db-1c43-4275-aae0-7e9523ad206f/.system_generated/steps/1974/content.md", "r") as f:
    text_val = f.read()

with open("C:/Users/Yassin/.gemini/antigravity/brain/ec9036db-1c43-4275-aae0-7e9523ad206f/.system_generated/steps/1979/content.md", "r") as f:
    text_test = f.read()

with open("C:/Users/Yassin/.gemini/antigravity/brain/ec9036db-1c43-4275-aae0-7e9523ad206f/.system_generated/steps/1981/content.md", "r") as f:
    text_train = f.read()

# Extract json substring
def extract_json(txt):
    idx = txt.find("[")
    if idx >= 0:
        return json.loads(txt[idx:])
    return []

official_val = extract_json(text_val)
official_test = extract_json(text_test)
official_train = extract_json(text_train)

print(f"Official val pairs: {len(official_val)}")
print(f"Official test pairs: {len(official_test)}")
print(f"Official train pairs: {len(official_train)}")

# Each entry is [id1, id2]
val_pairs = set(f"{p[0]}_{p[1]}" for p in official_val) | set(f"{p[1]}_{p[0]}" for p in official_val)
test_pairs = set(f"{p[0]}_{p[1]}" for p in official_test) | set(f"{p[1]}_{p[0]}" for p in official_test)
train_pairs = set(f"{p[0]}_{p[1]}" for p in official_train) | set(f"{p[1]}_{p[0]}" for p in official_train)

# Also single IDs
val_ids = set([p[0] for p in official_val] + [p[1] for p in official_val])
test_ids = set([p[0] for p in official_test] + [p[1] for p in official_test])
train_ids = set([p[0] for p in official_train] + [p[1] for p in official_train])

print(f"Official val unique single IDs: {len(val_ids)}")
print(f"Official test unique single IDs: {len(test_ids)}")
print(f"Official train unique single IDs: {len(train_ids)}")
print("Train & Val overlap:", len(train_ids & val_ids))
print("Train & Test overlap:", len(train_ids & test_ids))
print("Val & Test overlap:", len(val_ids & test_ids))
