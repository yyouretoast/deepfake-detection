import json

path = "kaggle_pipeline/run_release1_kaggle.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 0 (markdown)
nb["cells"][0]["source"] = [
    "# Turnkey Release 1 Execution Pipeline (Kaggle Dual/Single Tesla T4)\n",
    "### Project: *Dual-Stream Spatial-Frequency Feature Fusion with SNR-Adaptive Gating for Cross-Generator Deepfake Detection*\n",
    "### Author: Yassin Yasser (Sadat Academy for Management Sciences)\n",
    "\n",
    "This turnkey notebook executes the complete Release 1 pipeline:\n",
    "1. **Dataset Integrity Verification**: Zero identity leakage check and verification of 0 DFD samples in train/val.\n",
    "2. **Dual-Stream & Spatial Baseline Training**: 5 epochs on clean 37,104 train split (4-way balanced sampling, differential LRs `1e-5` / `1e-4`, FP16 AMP with FP32 FFT isolation).\n",
    "3. **Macro-AUC Validation Checkpoint Selection**: Checkpoint selected based on balanced macro-AUC across FF++ and Celeb-DF Set A validation.\n",
    "4. **Validation Calibration & Optimal Thresholding**: 4-way stratified Platt scaling, Youden's $J$ threshold $\\tau^*$, and Bayesian dual thresholds $(\\tau_{\\mathrm{real}}, \\tau_{\\mathrm{fake}})$ with $\\ge 98.0\\%$ precision.\n",
    "5. **Held-Out Test Evaluation (26,981 Crops)**:\n",
    "   - Overall metrics, trivial baseline (always-fake), and Spatial-Only ConvNeXt baseline.\n",
    "   - Per-source evaluation (FF++, Celeb-DF, DFD vs FF++, DFD vs YouTube) with 1,000-sample video-level clustered bootstrap 95% CIs.\n",
    "   - Fine-grained subdomain breakdown.\n",
    "6. **Spatiotemporal Bi-GRU Video Modeling**: 2-layer Bi-GRU with velocity deltas ($\\Delta \\mathbf{e}_t$) and dual-path attention/max-pooling.\n",
    "7. **5-Fold Leave-One-Target-Out (LOTO)**: Generalization benchmark across unseen architectures.\n",
    "8. **Robustness Stress-Testing**: Sweeps across JPEG, blur, noise, and downscaling.\n",
    "9. **Single-T4 Latency Profiling**: Synchronized $B=1$ and $B=32$ throughput.\n",
    "10. **Provenance & LaTeX Verification**: Writes `release1_results.json` and runs `verify_latex_full.py`."
]

# Update cell 4 (printout) to include spatial_only
nb["cells"][4]["source"] = [
    "# 4. Display Formatted Release 1 Results Ledger\n",
    "import json\n",
    "results_path = \"/kaggle/working/release1_results.json\"\n",
    "if os.path.exists(results_path):\n",
    "    with open(results_path) as f:\n",
    "        res = json.load(f)\n",
    "    print(\"=\"*60)\n",
    "    print(\"MASTER RELEASE 1 RESULTS SUMMARY\")\n",
    "    print(\"=\"*60)\n",
    "    print(\"Provenance:\", json.dumps(res.get(\"provenance\", {}), indent=2))\n",
    "    print(\"\\nCalibration:\", json.dumps(res.get(\"calibration\", {}), indent=2))\n",
    "    print(\"\\nOverall Test:\", json.dumps(res.get(\"test_frame_evaluation\", {}).get(\"overall\", {}), indent=2))\n",
    "    print(\"\\nSpatial-Only Baseline:\", json.dumps(res.get(\"test_frame_evaluation\", {}).get(\"spatial_only\", {}), indent=2))\n",
    "    print(\"\\nPer-Source Test:\", json.dumps(res.get(\"test_frame_evaluation\", {}).get(\"per_source\", {}), indent=2))\n",
    "    print(\"\\nTemporal Bi-GRU:\", json.dumps(res.get(\"temporal_video_evaluation\", {}), indent=2))\n",
    "    print(\"\\nLOTO Folds:\", json.dumps(res.get(\"loto_cross_generator_benchmark\", {}), indent=2))\n",
    "    print(\"\\nLatency Profiling:\", json.dumps(res.get(\"latency_profiling\", {}), indent=2))\n",
    "else:\n",
    "    print(f\"Results file not found at {results_path}\")"
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook successfully.")
