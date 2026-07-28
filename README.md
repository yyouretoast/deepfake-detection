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

# Deepfake Detection Engine

Dual-stream PyTorch 2.x Deepfake Detection architecture combining **ConvNeXt-Base** spatial features, **2D Real FFT** frequency spectrum embeddings, **Graph-Connected Component Partitioning**, **4-Head Cross-Attention Fusion**, **Layer-wise Learning Rate Decay (LLRD)**, **Macro F1 Threshold Calibration**, **Test-Time Augmentation (TTA)**, **ONNX Runtime Acceleration**, **Grad-CAM Visualizations**, and a **Streamlit Web UI**. The **frame-level dual-stream baseline** is currently active.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Live Application

Application Interface: **[https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)**

---

## System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ 256x256 Face Crops ]
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ Spatial Stream (ConvNeXt-Base) ]                       [ Frequency Stream (2D Real FFT) ]
         • 256x256 RGB Image Input                                • 2D Real FFT (norm="ortho")
         • 1024-d Feature Embeddings                               • 128-d Frequency Spectrum Embeddings
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       [ 4-Head Cross-Attention Fusion ]
                                       • Spatial Query (128-d) ◄► Freq Key/Value (128-d)
                                       • Residual Connection + 1152-d Fusion
                                                      │
                                                      ▼
                                           [ Binary Classifier Head ]
                                           • LayerNorm(256) + Dropout(0.3)
                                           • Macro F1 Calibrated Decision Threshold T*
```

> [!NOTE]
> **Active Baseline**: The **frame-level dual-stream baseline** (Spatial ConvNeXt + Frequency 2D FFT) is currently active by default. Optional extensions such as 5D Temporal Sequence Transformer and LoRA parameter-efficient adapters are implemented as modular options in `src/models/`.

### Technical Formulation

**1. 2D Real FFT Spectrum Extraction**:

$$
\mathcal{F}_{\text{norm}} = \frac{1}{10} \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{gray}})| + 10^{-5} \right)
$$

Computes 2D real FFT log-magnitude frequency spectra to capture high-frequency grid and compression artifacts.

**2. 4-Head Cross-Attention Residual Fusion**:

$$
\mathbf{f}_{\text{enhanced}} = \mathbf{f}_{\text{freq}} + 0.1 \times \text{Attention}(Q_{\text{spatial}}, K_{\text{freq}}, V_{\text{freq}})
$$

---

## Technical Specifications

1. **Active Baseline**: Frame-level dual-stream ConvNeXt-Base + 2D FFT architecture is active by default.
2. **Frequency Spectrum Extraction**: Computes 2D `rfft2` log-magnitude frequency spectra to capture spectral artifacts.
3. **Identity Partitioning**: Uses `networkx.Graph` parsing source and target actor IDs (`id1`, `id2`) to isolate connected components and prevent identity leakage across splits.
4. **Layer-wise Learning Rate Decay (LLRD)**: Applies stage-decayed learning rates across ConvNeXt stages (`2e-6` stem/stages 0-1 $\rightarrow$ `5e-6` stage 2 $\rightarrow$ `1e-5` stage 3 $\rightarrow$ `1e-4` head).
5. **Per-Batch Scheduler & Warmup**: Steps `CosineAnnealingLR` per-batch (`T_max = total_steps`) with AMP `GradScaler` skip guards and dynamic Phase 1 `LinearLR` warmup (`warmup_ratio=0.1`).
6. **Test-Time Augmentation (TTA)**: Computes dual-pass predictions on original and horizontally flipped inputs (`torch.flip(imgs, dims=[-1])`) during evaluation.

---

## Benchmark Performance (FaceForensics++ C23 & Celeb-DF v2)

Evaluation under **Stratified GroupKFold by Video-ID** (zero identity leakage):

| Model / Architecture | Input Resolution | AUC | Accuracy | Notes / Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Standard CNNs (ResNet-50 / VGG16)** | $224 \times 224$ | `0.810 - 0.850` | `75.0% - 81.0%` | Baseline spatial frame classifiers |
| **Spatial-Only ConvNeXt-Base** | $256 \times 256$ | `0.932` | `87.5%` | Spatial stream only (no FFT branch) |
| **Xception Baseline** | $299 \times 299$ | `0.950` | `89.3%` | *Rossler et al., ICCV 2019* (FF++ Benchmark Paper) |
| **Dual-Stream ConvNeXt + 2D FFT (Baseline)** | $256 \times 256$ | `0.903+` | `87.1%+` | **Active Frame-Level Dual-Stream Baseline** |

### Benchmark Commands

```bash
# Fast Stratified Validation Benchmark (500 videos)
python benchmark.py --mode fast

# Full Paper Benchmark (Celeb-DF v2 + 5-Fold Leave-One-Type-Out)
python benchmark.py --mode paper
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Integration Tests

```bash
pytest tests/ -v
```

### 3. Streamlit Application

```bash
streamlit run app.py
```

---

## Model Training

Training execution via CLI or notebook (`deepfake_detection_v2_pytorch.ipynb`):

```bash
python train.py --epochs_phase1 3 --epochs_phase2 15
```

---

## Docker Deployment

```bash
docker build -t deepfake-detector .
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the interface at `http://localhost:8501`.

---

## Repository Structure

```
deepfake-detection/
├── app.py                     # Streamlit web application with active 5D sequence inference
├── benchmark.py               # Benchmark runner (--mode fast, --mode paper, 5-fold LOTO)
├── config/
│   └── default.yaml           # Centralized configuration parameters
├── src/
│   ├── config.py              # YAML configuration parser & fallback defaults
│   ├── dataset/
│   │   ├── loader.py          # Stratified GroupKFold zero-leakage splitter & dual ID parser
│   │   └── preprocess.py      # DynamicFaceCropper & 512x512 MTCNN extraction
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt + 2D FFT + Cross-Attention architecture
│   │   ├── lora.py            # LoRAConv2d, weight-folding (merge_weights), & micro-checkpoints
│   │   ├── temporal.py        # 5D Temporal Sequence Transformer Encoder
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter with dynamic sequence axes
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM 512x512 heatmap generator & overlay
├── tests/                     # Automated integration test suite
├── deepfake_detection_v2_pytorch.ipynb # GPU Training Notebook
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production Docker container definition
└── README.md                  # Project documentation
```

---

## Author & License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
- **HuggingFace Hub**: [yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
