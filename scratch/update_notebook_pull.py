import json

nb_path = "deepfake_detection_v2_pytorch.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell1_source = [
    "# Install dependencies and verify environment\n",
    "!pip install \"Pillow<11.0\" timm facenet-pytorch albumentations grad-cam onnx onnxruntime pyyaml -q\n",
    "\n",
    "import os, sys, re, random, time, cv2, torch, shutil, copy, numpy as np\n",
    "\n",
    "# Clone or pull latest repository on Kaggle/Colab\n",
    "if not os.path.exists(\"src\") and not os.path.exists(\"deepfake-detection\"):\n",
    "    !git clone https://github.com/yyouretoast/deepfake-detection.git\n",
    "elif os.path.exists(\"deepfake-detection\"):\n",
    "    !git -C deepfake-detection pull\n",
    "\n",
    "if os.path.exists(\"deepfake-detection\"):\n",
    "    abs_repo = os.path.abspath(\"deepfake-detection\")\n",
    "    if abs_repo not in sys.path:\n",
    "        sys.path.append(abs_repo)\n",
    "\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.fft\n",
    "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n",
    "from PIL import Image\n",
    "from tqdm import tqdm\n",
    "import timm\n",
    "from facenet_pytorch import MTCNN\n",
    "import albumentations as A\n",
    "from albumentations.pytorch import ToTensorV2\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from accelerate import Accelerator, notebook_launcher\n",
    "\n",
    "print(f\"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}\")\n",
    "for i in range(torch.cuda.device_count()):\n",
    "    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")"
]

nb["cells"][1]["source"] = cell1_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 1 with git -C deepfake-detection pull!")
