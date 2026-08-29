# Experiment Reproduction Guide

This guide details the exact dataset setup, environment setup, CLI execution commands, and ablation protocols for reproducing the Dual-Stream Deepfake Detector metrics and Leave-One-Target-Out (LOTO) experiments.

## 1. Directory & Dataset Layout

Place pre-processed face crops in the dataset root specified in `src/config.py`:

```
/data/deepfake_dataset/
├── splits.json
├── real/
│   └── id0_id16/
│       └── *.png
└── fake/
    ├── ff_c23/
    │   ├── 000_003/ (Deepfakes)
    │   ├── 200_203/ (Face2Face)
    │   ├── 400_403/ (FaceSwap)
    │   └── 600_603/ (NeuralTextures)
    └── celeb_df_v2/
        └── id0_id16/
```

## 2. Unit Test Suite Verification

Run the full pytest suite before training or evaluation:

```bash
pytest tests/ -v
```

Expected output: `66/66 passed`.

## 3. Distributed Training (DDP via Accelerate)

To train the dual-stream model across 2x GPUs:

```bash
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_dual_stream_ddp.py
```

## 4. Held-Out Test Evaluation & Calibration

Run test set evaluation and SciPy L-BFGS-B log-temperature calibration:

```bash
python scripts/evaluate_test_set.py
```

Outputs:
- Calibrated checkpoint contract: `dual_stream_calibrated.pth`
- Test predictions log: `test_predictions.json`

## 5. Leave-One-Target-Out (LOTO) Folds

To train and evaluate a LOTO experiment holding out a specific generator domain:

```bash
# Fold 4: Leave-Out NeuralTextures
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_loto_experiment.py --holdout neuraltextures --epochs 3

# Fold 5: Leave-Out Celeb-DF v2
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_loto_experiment.py --holdout celeb --epochs 3
```

## 6. Celeb-DF Re-Compression Ablation Protocol

To test the hypothesis that compression artifacts drive the Fold 5 zero-shot drop:

```bash
# Step 1: Re-encode Celeb-DF test videos with FFmpeg H.264 at FF++ c23 bitrate
for vid in /data/celeb_df_v2/test_videos/*.mp4; do
    ffmpeg -i "$vid" -vcodec libx264 -crf 23 -preset medium "recompressed_$(basename $vid)"
done

# Step 2: Extract face crops from re-compressed videos
python scripts/extract_face_crops.py --input_dir /data/celeb_df_v2/recompressed/ --output_dir /data/celeb_df_v2_c23_crops/

# Step 3: Run inference on re-compressed test split
python scripts/evaluate_test_set.py --data_root /data/celeb_df_v2_c23_crops/
```

- **Validation Rule**: If AUC recovers significantly toward in-distribution levels ($\Delta\text{AUC} \gg 0$), the compression shortcut hypothesis is confirmed.
