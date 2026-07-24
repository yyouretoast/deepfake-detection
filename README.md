---
title: Deepfake Detection Engine
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# Deepfake Detection System

> Dual-stream PyTorch 2.x Deepfake Detection model combining **ConvNeXt-Base** spatial features and **2D FFT** frequency spectrum embeddings. Includes **Stratified Group-based Video-ID Partitioning**, **Test-Time Augmentation (TTA)**, **ONNX Runtime**, **Grad-CAM Visualizations**, and a **Streamlit UI**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Batched-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Model Architecture

- **Dual-Stream Adaptive Gated Fusion**: Fuses spatial features from ConvNeXt-Base (1024-d) with frequency embeddings (128-d) extracted via 2D Real FFT using a learnable gating network into an 1152-d representation.
- **Absolute Spectral Magnitude Scaling**: Applies static `/ 10.0` scaling to the 2D FFT log-spectrum, preserving absolute high-frequency intensity differences across samples while keeping inputs bounded for stable CNN convergence.
- **Test-Time Augmentation (TTA)**: Computes dual-pass predictions on original and horizontally flipped frame batches, averaging probabilities for maximum inference accuracy.
- **Youden's J Threshold Calibration**: Computes the optimal decision boundary (`TPR - FPR`) on validation ROC curves and saves this threshold into checkpoint metadata.
- **Robustness Augmentations**: Uses JPEG Compression, Affine transforms, Color Jitter, Downscaling, and Gaussian Blur during training to enhance cross-manipulation generalization.
- **Stratified Group-Based Video-ID Partitioning**: Prevents identity leakage and class imbalance by guaranteeing zero video-ID overlap between splits while enforcing a 50/50 Real/Fake class distribution.
- **Batched GPU Face Extraction**: Uses MTCNN with a 1.30x bounding box scale expansion. Extracts multiple faces per frame, sorting by bounding-box area and capping at `max_faces=3` to guarantee OOM safety. Empty frames are mathematically filtered.
- **Dynamic Inference Chunking**: Chunks extracted tensors into mini-batches (`BATCH_SIZE=16`) to ensure absolute VRAM stability during heavy multi-face scene parsing.
- **Centralized Configuration**: Configured dynamically via `config/default.yaml`.
- **ONNX Runtime Integration**: Supports exporting to ONNX format for accelerated CPU/GPU inference.
- **Grad-CAM Visualizations (Triple-Zipping)**: All extracted faces are passed through inference, then triple-zipped with their probabilities and source tensors. The system targets the top 4 "Most Fake" faces, ensuring exact mathematical alignment between the visualized heatmap and the face it explains.
- **Normalized Confidence Metrics**: The UI mathematically normalizes probability scores relative to the Youden's J dynamic threshold, guaranteeing that the decision boundary always represents exactly 50% confidence.
- **Automated `pytest` Integration Suite**: 7 integration test modules covering E2E Streamlit app execution, data leakage, Grad-CAM targeting, configuration loading, tensor shapes, cropper bounds safety, and ONNX parity.

---

## Quick Start

### 1. Local Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Automated Integration Test Suite

```bash
pytest tests/ -v
```

### 3. Run Streamlit Web Application

```bash
streamlit run app.py
```

---

## System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ Dynamic 1.30x Face Crops (224x224) ]
                                                            │
                            ┌───────────────────────────────┴──────────────────────────────┐
                            ▼                                                              ▼
               [ Spatial Stream (ConvNeXt-Base) ]               [ Frequency Stream (2D FFT) ]
               • 1024-d Feature Embeddings                        • Absolute Magnitude Log Spectrum
                                                            │     • 128-d Frequency Embeddings
                                                            └───────────────┬──────────────┘
                                                                            ▼
                                                              [ Adaptive Gated Fusion ]
                                                              • Learnable Gating Network (Bias Init -2.0)
                                                              • 1152-d Dynamic Representation
                                                                            │
                                                                            ▼
                                                                  [ Binary Classifier Head ]
                                                                  • Youden's J Calibrated Decision Threshold T*
```

---

## Benchmarking & Training Reproducibility

### 1. Empirical Latency Benchmarking

To measure PyTorch Native vs ONNX Runtime inference latency on your specific CPU/GPU hardware, execute the automated benchmarking script:

```bash
python benchmark.py
```

### 2. Kaggle GPU Training Execution (2x NVIDIA T4 GPUs)

Model weights (`deepfake_convnext_v2.pth`) and ONNX exports (`deepfake_convnext_v2.onnx`) are produced by running the self-contained PyTorch training notebook on Kaggle against FaceForensics++ (C23):

```bash
# Training notebook file
deepfake_detection_v2_pytorch.ipynb
```

The notebook pipeline executes:
- **Phase 1**: Frozen backbone head warmup (3 epochs, `lr=1e-3`).
- **Phase 2**: End-to-end differential LR fine-tuning with AMP fp16 (5 epochs, `lr_backbone=1e-5`, `lr_head=1e-4`).
- **Robustness Training**: Affine, Downscale, JPEG Compression, Color Jitter, and Gaussian Blur Albumentations pipeline.
- **Youden's J ROC Calibration**: Dynamically calculates optimal decision threshold `T*` on validation set and exports `T*` in checkpoint metadata.
- **Ablation Study**: Spatial-Only vs Dual-Stream (Adaptive Gated Fusion) accuracy comparison.
- **Generalization Benchmark**: Leave-One-Type-Out (LOTO) cross-manipulation evaluation.
- **ONNX Export**: Saves verified model to `deepfake_convnext_v2.onnx`.

---

## Docker Deployment

Build and run the production containerized web application:

```bash
# Build Docker image
docker build -t deepfake-detector .

# Run container on port 8501
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the web interface at `http://localhost:8501`.

---

## Known Limitations & Scope

1. **Model Checkpoint Generation**:
   - The repository provides the complete PyTorch architecture, preprocessing engine, test suite, and web application. Model weights (`deepfake_convnext_v2.pth`) must be generated by executing `deepfake_detection_v2_pytorch.ipynb` on GPU hardware.
2. **Dataset & Compression Generalization**:
   - Trained on FaceForensics++ (C23 compression). Performance on heavy social media re-compression (e.g. WhatsApp/TikTok re-encoding) or novel generative models (Sora, FLUX) requires fine-tuning on domain-specific datasets.
3. **Grad-CAM Interpretation**:
   - Grad-CAM heatmaps highlight feature activations within `spatial_backbone`. They serve as visual attention indicators, not definitive proof of manipulation.

---

## Directory Structure

```
deepfake-detection/
├── app.py                     # Streamlit web application with TTA & ONNX acceleration
├── benchmark.py               # Empirical inference latency benchmarking script
├── config/
│   └── default.yaml           # Global parameters, paths, and model hyperparams
├── src/
│   ├── config.py              # Centralized YAML configuration parser & defaults
│   ├── dataset/
│   │   ├── loader.py          # Stratified GroupKFold zero-leakage splitter & Albumentations
│   │   └── preprocess.py      # DynamicFaceCropper & GPU Batched MTCNN
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt-Base + 2D FFT PyTorch architecture
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter & ONNXRuntime engine
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM heatmap generator & overlay
├── tests/                     # Automated integration test suite
│   ├── test_app.py            # End-to-End Streamlit App video prediction integration tests
│   ├── test_config.py         # YAML configuration parsing tests
│   ├── test_data_leakage.py   # Stratified GroupKFold zero-leakage assertions
│   ├── test_gradcam.py        # Grad-CAM spatial layer targeting & import tests
│   ├── test_model_forward.py  # Model forward pass tensor shape tests
│   ├── test_preprocess.py     # Dynamic cropper boundary safety tests
│   └── test_onnx.py           # PyTorch vs ONNX prediction parity tests
├── deepfake_detection_v2_pytorch.ipynb # Kaggle / Colab 2x T4 GPU Training Notebook
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production container deployment definition
└── README.md                  # Project documentation
```

---

## Author & License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
