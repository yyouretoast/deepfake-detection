# Changelog

All notable changes and architectural bug fixes for the Dual-Stream Deepfake Detector Engine.

## [v2.0.0] - 2026-08-12

### 🔴 Critical Bug Fixes & Refactors
- **Import Error Safety (`src/__init__.py`)**: Resolved silent `ImportError` where non-existent `group_video_split` was imported instead of `group_samples_by_video`. Added backward-compatible alias.
- **Identity Graph Splitting (`src/dataset/loader.py`)**: Fixed regex fallback `(\d+)_(\d+)` matching padded frame counter numbers (e.g., `0001`) as actor pair IDs, which previously collapsed the dataset identity graph.
- **Forensic Signal Preservation (`src/dataset/loader.py`)**: Removed `ImageCompression` and `GaussianBlur` from training augmentations as they destroy the high-frequency spectral signals used by the SRM + Bayar + 2D FFT branch.
- **ConvNeXt Spatial Feature Normalization (`src/models/hybrid_detector.py`)**: Extracted and applied ConvNeXt's `LayerNorm2d` (`spatial_norm`) prior to spatial pooling, ensuring scaled distributions when fusing spatial and frequency embeddings.
- **Symmetric Gated Residual Fusion (`src/models/hybrid_detector.py`)**: Replaced asymmetric gated fusion with symmetric complementary gating: `[f_spatial * (1 - g) || f_freq * g]`, preventing frequency gradient starvation.
- **Sequence Memory Optimization (`src/models/hybrid_detector.py`)**: Rewrote `forward_sequence` to process 5D video frame sequences in micro-chunks of 8 frames to prevent GPU OOM crashes.
- **Face Landmark Scale Factor Alignment (`src/dataset/preprocess.py`)**: Incorporated `scale_factor` expansion into 5-point canonical landmark alignment points to prevent crop scale mismatch between landmark-aligned and bounding-box crops.
- **Grad-CAM Memory Cleanup (`src/utils/interpretability.py`)**: Wrapped Grad-CAM forward/backward hook registration in `try/finally` blocks, guaranteeing hook removal even if exceptions occur during diagnostic rendering.
- **Decision Threshold Calibration (`scripts/evaluate_test_set.py`)**: Shifted threshold grid search to execute on temperature-calibrated probabilities rather than raw logits.
- **Sequential Video Seeking (`src/services/video_engine.py`)**: Replaced slow and inaccurate `CAP_PROP_POS_FRAMES` random seeking with sequential keyframe decoding for 10x-50x faster video frame extraction.

### 🟡 Training & Infrastructure Enhancements
- **Multi-GPU DDP SyncBatchNorm (`scripts/train_dual_stream_ddp.py`)**: Added `accelerator.sync_batch_norm` to synchronize Batch Normalization running statistics across DDP worker processes.
- **Gradient Accumulation & Effective Batching (`scripts/train_dual_stream_ddp.py`)**: Set default `GRAD_ACCUM_STEPS = 4` to simulate an effective batch size of 64 without extra VRAM.
- **Cosine Warmup Learning Rate Scheduler (`scripts/train_dual_stream_ddp.py`)**: Implemented 1-epoch `LinearLR` warmup chained into `CosineAnnealingLR` via `SequentialLR`.
- **Focal Loss & EMA Weight Tracking (`scripts/train_dual_stream_ddp.py`)**: Integrated `FocalLossWithLogits` ($\gamma=2.0$) and `ExponentialMovingAverage` parameter shadow tracking.
- **LOTO Fold Seed Rotation (`scripts/train_loto_experiment.py`)**: Derived unique fold-specific seeds from holdout domain names to rotate random generators across cross-generator folds.
- **Checkpoint Loading Security (`scripts/evaluate_subdomain_breakdown.py`)**: Enforced `weights_only=True` during `torch.load` with safe fallback handling.

### 🔵 Production & UI Updates
- **Startup Model Warm-Up (`app.py`)**: Added dummy inference tensor pass on startup inside `@st.cache_resource` to trigger PyTorch CUDA JIT compilation.
- **File Upload Limits & UI Preview (`app.py`)**: Added 50MB file size cap enforcement and live file size preview in the Streamlit UI.
- **Dependency Pinning (`requirements.txt`)**: Pinned `accelerate>=0.25.0,<2.0.0` to avoid breaking API changes.

### 🧪 Test Suite & Verification
- Expanded unit test suite from 51 to **62 passing unit tests**:
  - `test_extract_identities_frame_counter_not_parsed_as_actor_pair`: Identity graph regression test.
  - `test_generate_face_diagnostics_keys_and_hook_cleanup`: Grad-CAM diagnostics & hook cleanup test.
  - `test_process_video_frames_with_synthetic_mp4`: Synthetic OpenCV MP4 video processing test.
