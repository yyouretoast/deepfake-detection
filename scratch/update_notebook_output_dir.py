import json

nb_path = "deepfake_detection_v2_pytorch.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 7 (index 6 in zero-indexed list of code/markdown cells, find source containing ram_frames)
for idx, cell in enumerate(nb["cells"]):
    source_str = "".join(cell.get("source", []))
    if "RAM_DIR = \"/tmp/ram_frames\"" in source_str:
        cell_source = cell["source"]
        # Update source to ensure OUTPUT_DIR and SEED fallbacks exist
        new_source = [
            "# Identity-safe split — imported from src/ (single source of truth)\n",
            "# Uses graph-connected component partitioning to guarantee zero identity leakage.\n",
            "from src.dataset.loader import perform_graph_split, extract_identities, extract_video_id\n",
            "\n",
            "# Fallback definitions in case Cell 3 was skipped\n",
            "OUTPUT_DIR = globals().get(\"OUTPUT_DIR\", \"/kaggle/working/frames_v2\")\n",
            "SEED = globals().get(\"SEED\", 42)\n",
            "\n",
            "# Copy extracted frames to Linux RAM disk (/tmp/ram_frames) for zero-latency I/O\n",
            "RAM_DIR = \"/tmp/ram_frames\"\n",
            "if os.path.exists(OUTPUT_DIR) and not os.path.exists(RAM_DIR):\n",
            "    print(\"Copying dataset to Linux RAM Disk (/tmp/ram_frames)...\")\n",
            "    shutil.copytree(OUTPUT_DIR, RAM_DIR, dirs_exist_ok=True)\n",
            "    print(\"RAM Disk Copy Complete.\")\n",
            "\n",
            "DATASET_DIR = RAM_DIR if os.path.exists(RAM_DIR) else OUTPUT_DIR\n",
            "real_files = [(os.path.join(DATASET_DIR, \"real\", f), 0) for f in os.listdir(f\"{DATASET_DIR}/real\")] if os.path.exists(f\"{DATASET_DIR}/real\") else []\n",
            "fake_files = [(os.path.join(DATASET_DIR, \"fake\", f), 1) for f in os.listdir(f\"{DATASET_DIR}/fake\")] if os.path.exists(f\"{DATASET_DIR}/fake\") else []\n",
            "all_samples = real_files + fake_files\n",
            "\n",
            "print(f\"Total Real Files: {len(real_files)} | Total Fake Files: {len(fake_files)}\")\n",
            "train_samples, val_samples, test_samples = perform_graph_split(all_samples, seed=SEED)\n",
            "\n",
            "print(f\"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Test samples: {len(test_samples)}\")\n"
        ]
        cell["source"] = new_source
        print(f"Updated cell index {idx} with OUTPUT_DIR fallback!")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Saved notebook with OUTPUT_DIR fallback.")
