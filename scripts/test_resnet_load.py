import os
import glob
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Check if pretrained model is accessible or if torchvision weights can be loaded
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    print("ResNet18 loaded successfully")
except Exception as e:
    print("Could not load pretrained weights:", e)
