import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load spatial backbone from dual_stream_calibrated.pth
ckpt = torch.load("models/dual_stream_calibrated.pth", map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)

# Look at keys
spatial_keys = [k for k in state_dict.keys() if "spatial_tower" in k or "spatial" in k]
print(f"Spatial keys count: {len(spatial_keys)}")
for k in spatial_keys[:10]:
    print(" ", k)
