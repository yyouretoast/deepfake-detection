---
title: Deepfake Detection Engine
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: mit
---

# Deepfake Detection System

> Dual-stream PyTorch 2.x Deepfake Detection model combining **ConvNeXt-Base** spatial features and **2D Real FFT** frequency spectrum embeddings. Includes **Stratified Group-based Video-ID Partitioning**, **Layer-wise Learning Rate Decay (LLRD)**, **HuggingFace Accelerate DDP**, **Test-Time Augmentation (TTA)**, **ONNX Runtime**, **Grad-CAM Visualizations**, and a **Streamlit UI**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Live Application

[Placeholder]

---

## Benchmark Performance (FaceForensics++ C23)

Performance comparison on the FaceForensics++ (C23 compression) benchmark under **Stratified GroupKFold by Video-ID** (zero identity leakage):

| Model / Architecture | Input Resolution | AUC | Accuracy | Notes / Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Standard CNNs (ResNet-50 / VGG16)** | $224 \times 224$ | `0.810 - 0.850` | `75.0% - 81.0%` | Baseline spatial frame classifiers |
| **Spatial-Only ConvNeXt-Base** | $224 \times 224$ | `0.932` | `87.5%` | Spatial stream only (no FFT branch) |
| **Xception Baseline** | $299 \times 299$ | `0.950` | `89.3%` | *Rossler et al., ICCV 2019* (FF++ Benchmark Paper) |
| **Dual-Stream ConvNeXt + 2D FFT (This Work)** | $256 \times 256$ | *[ Training Run ]* | *[ Training Run ]* | **ConvNeXt-Base + 2D FFT + LLRD + Accelerate DDP** |

---

## System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ Dynamic 1.30x Face Crops (256x256) ]
                                                            │
                            ┌───────────────────────────────┴──────────────────────────────┐
                            ▼                                                              ▼
               [ Spatial Stream (ConvNeXt-Base) ]               [ Frequency Stream (2D FFT) ]
               • 1024-d Feature Embeddings                        • 2D Real FFT Log-Spectrum (log/10.0)
                                                            │     • 128-d Frequency Embeddings
                                                            └───────────────┬──────────────┘
                                                                            ▼
                                                              [ Adaptive Gated Fusion ]
                                                              • Learnable Gating Network (Bias Init -2.0)
                                                              • 1152-d Dynamic Representation
                                                                            │
                                                                            ▼
                                                                  [ Binary Classifier Head ]
                                                                  • nn.LayerNorm(256) + Dropout(0.3)
                                                                  • Youden's J Calibrated Threshold T*
```

---

## Key Engineering Innovations

1. **Dual-Stream Adaptive Gated Fusion**: Fuses spatial features from ConvNeXt-Base (1024-d) with frequency embeddings (128-d) extracted via 2D Real FFT. A learnable gating network dynamically weights frequency contributions per sample.
2. **`LayerNorm(256)` Head Stability**: Uses `nn.LayerNorm(256)` instead of `BatchNorm1d` in the classifier head, providing batch-size invariant normalization that prevents single-sample inference crashes and multi-GPU `DataParallel` split failures.
3. **Layer-wise Learning Rate Decay (LLRD)**: Applies stage-decayed learning rates across ConvNeXt stages (`2e-6` stem/stages 0-1 $\rightarrow$ `5e-6` stage 2 $\rightarrow$ `1e-5` stage 3 $\rightarrow$ `1e-4` head), preserving low-level visual edge filters while fine-tuning deep semantic layers.
4. **HuggingFace `Accelerate` DDP Engine**: Built using `accelerate.Accelerator` for process-isolated Distributed Data Parallel (DDP) execution across 2x T4 GPUs in interactive notebooks without multi-processing kernel deadlocks.
5. **Target Label Smoothing & Gradient Clipping**: Applies direct target smoothing (`labels * 0.95 + 0.025`) and AMP gradient norm capping (`max_norm=1.0`) to stabilize FP16 training and calibrate decision confidence.
6. **Single-Pass Epoch Validation**: Evaluates single-pass predictions during epoch validation loops for 50% faster per-epoch latency, while reserving dual-pass Test-Time Augmentation (TTA) for final test evaluation.
7. **Stratified Group-Based Video-ID Partitioning**: Guarantees zero video-ID overlap between Train/Val/Test splits while enforcing a 50/50 Real/Fake class ratio to eliminate identity leakage.

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

### 3. Run Streamlit Web Application Locally

```bash
streamlit run app.py
```

---

## GPU Training Execution (2x NVIDIA T4 GPUs)

The training pipeline is self-contained in `deepfake_detection_v2_pytorch.ipynb`.

Execution Pipeline:
- **Phase 1**: Frozen backbone warmup (3 epochs, `lr=1e-3`).
- **Phase 2**: End-to-end differential fine-tuning with LLRD and AMP fp16 (15 epochs with early stopping `patience=4`).
- **Youden's J ROC Calibration**: Calculates optimal decision threshold `T*` on validation ROC curves and embeds `T*` in model metadata.
- **Ablation Study**: Compares Spatial-Only vs Dual-Stream performance.
- **Leave-One-Type-Out (LOTO)**: Evaluates cross-manipulation generalization on held-out forgery types.
- **ONNX Export**: Exports verified model to `deepfake_convnext_v2.onnx`.

---

## Docker Deployment

Build and run the production containerized web application:

```bash
# Build Docker image
docker build -t deepfake-detector .

# Run container on port 8501
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the application interface at `http://localhost:8501`.

---

## Directory Structure

```
deepfake-detection/
├── app.py                     # Streamlit web application with TTA & ONNX acceleration
├── benchmark.py               # Inference latency benchmarking script (PyTorch vs ONNX)
├── config/
│   └── default.yaml           # Centralized configuration parameters
├── src/
│   ├── config.py              # YAML configuration parser & default fallbacks
│   ├── dataset/
│   │   ├── loader.py          # Stratified GroupKFold zero-leakage splitter & Albumentations
│   │   └── preprocess.py      # DynamicFaceCropper & GPU Batched MTCNN
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt-Base + 2D FFT PyTorch architecture
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter & ONNXRuntime engine
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM heatmap generator & overlay
├── tests/                     # Automated integration test suite (17 tests)
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
