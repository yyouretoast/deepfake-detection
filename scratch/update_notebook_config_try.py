import json

nb_path = "deepfake_detection_v2_pytorch.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell3_source = [
    "# Load configuration and set random seeds\n",
    "try:\n",
    "    from src.config import load_config\n",
    "    CFG = load_config()\n",
    "except Exception:\n",
    "    CFG = {\n",
    "        'paths': {'kaggle_input': '/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23', 'output_dir': '/kaggle/working/frames_v2'},\n",
    "        'preprocessing': {'img_size': 512, 'padding_scale': 1.30, 'max_real_videos': 300, 'max_fake_per_dir': 150, 'frames_per_video': 15},\n",
    "        'training': {'batch_size': 32, 'epochs_phase1': 3, 'epochs_phase2': 15, 'lr_phase1': 1e-4, 'lr_backbone': 1e-5, 'lr_head': 1e-4, 'seed': 42},\n",
    "        'manipulation_types': {'all': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'], 'held_out_loto': 'FaceShifter'}\n",
    "    }\n",
    "\n",
    "SEED = CFG.get('training', {}).get('seed', 42)\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.manual_seed_all(SEED)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False\n",
    "\n",
    "BASE = CFG.get('paths', {}).get('kaggle_input', '/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23')\n",
    "OUTPUT_DIR = \"/kaggle/working/frames_v2\"\n",
    "IMG_SIZE = CFG.get('preprocessing', {}).get('img_size', 512)\n",
    "PADDING_SCALE = CFG.get('preprocessing', {}).get('padding_scale', 1.30)\n",
    "FAKE_DIRS = CFG.get('manipulation_types', {}).get('all', ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'])\n",
    "HELD_OUT_TYPE = CFG.get('manipulation_types', {}).get('held_out_loto', 'FaceShifter')\n",
    "\n",
    "shutil.rmtree(OUTPUT_DIR, ignore_errors=True)\n",
    "os.makedirs(f\"{OUTPUT_DIR}/real\", exist_ok=True)\n",
    "os.makedirs(f\"{OUTPUT_DIR}/fake\", exist_ok=True)\n",
    "print(\"Configuration loaded.\")"
]

nb["cells"][3]["source"] = cell3_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 3 to catch all exceptions gracefully!")
