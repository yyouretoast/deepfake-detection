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

> Production-grade, dual-stream PyTorch 2.x Deepfake Detection architecture combining **ConvNeXt-Base** spatial features, **Pre-Downsample 512x512 2D Real FFT** frequency spectrum embeddings, **LoRA (Low-Rank Adaptation)** parameter-efficient fine-tuning, and a **5D Temporal Sequence Transformer**. Features **Graph-Connected Component Partitioning**, **4-Head Cross-Attention Fusion**, **Layer-wise Learning Rate Decay (LLRD)**, **Macro F1 Threshold Calibration**, **Test-Time Augmentation (TTA)**, **ONNX Runtime Acceleration**, **Grad-CAM Visualizations**, and an interactive **Streamlit UI**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-28%2B_Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🚀 Live Application

Experience the live interactive web application on HuggingFace Spaces:

👉 **[Launch Deepfake Detection Web App](https://huggingface.co/spaces/yyouretoast/deepfake-detector)**

---

## 🏛️ System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ Native 512x512 Face Crops ]
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ Spatial Stream (ConvNeXt-Base) ]                       [ Frequency Stream (2D Real FFT) ]
         • 512x512 ──► GPU Downsample 256x256                      • Un-downsampled 512x512 Image Input
         • LoRA Low-Rank Adaptation (r=8)                         • 2D Real FFT (norm="ortho")
         • 1024-d Feature Embeddings                               • 128-d Frequency Spectrum Embeddings
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       [ 4-Head Cross-Attention Fusion ]
                                       • Spatial Query (128-d) ◄► Freq Key/Value (128-d)
                                       • 0.1x Residual Connection + 1152-d Fusion
                                                      │
                                                      ▼
                                       [ 5D Temporal Sequence Transformer ]
                                       • Chunked 8-Frame Sequence Extraction (<600MB VRAM)
                                       • 2-Layer Pre-LN Transformer Encoder
                                                      │
                                                      ▼
                                           [ Binary Classifier Head ]
                                           • LayerNorm(256) + Dropout(0.3)
                                           • Macro F1 Calibrated Decision Threshold T*
```

### Mathematical Foundations

1. **Pre-Downsample 512x512 2D Real FFT Spectrum Extraction**:
   $$\mathcal{F}_{\text{norm}} = \frac{1}{10} \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{gray}})| + 10^{-5} \right)$$
   Preserves sub-pixel high-frequency generative grid artifacts up to native Nyquist resolution ($f_{N,\text{native}} = 256 \text{ cycles}$) before spatial downsampling.

2. **LoRA Low-Rank Adaptation (Weight Folding)**:
   $$W_{\text{effective}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
   Injects trainable rank $r=8$ matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ into ConvNeXt blocks, reducing Phase 2 trainable parameters by 98.6% with zero inference latency penalty (`merge_weights()`).

3. **4-Head Cross-Attention Residual Fusion**:
   $$\mathbf{f}_{\text{enhanced}} = \mathbf{f}_{\text{freq}} + 0.1 \times \text{Attention}(Q_{\text{spatial}}, K_{\text{freq}}, V_{\text{freq}})$$

---

## 🛠️ Key Engineering Innovations

1. **Native 512x512 Frequency Spectrum Extraction**: Extracts 2D `rfft2` log-magnitude frequency spectra on uncompressed 512x512 face crops *before* spatial downsampling, eliminating low-pass filtering distortion and preserving sub-pixel up-sampling artifacts.
2. **LoRA Parameter-Efficient Fine-Tuning**: Injects low-rank adapters into ConvNeXt-Base 7x7 depthwise (`dwconv`) and 1x1 pointwise (`pwconv1`, `pwconv2`) conv blocks, cutting trainable parameters by 98.6% (88M $\rightarrow$ 1.2M) while preventing catastrophic forgetting.
3. **Active 5D Video Sequence Transformer**: Processes video frame sequences through a 2-layer Pre-LN `TemporalSequenceEncoder` with mini-batched 8-frame sequence chunking (`chunk_size=8`), guaranteeing peak VRAM memory allocations under 600MB during video inference.
4. **Graph-Connected Component Identity Partitioning**: Uses `networkx.Graph` parsing both source and target actor video IDs (`id1`, `id2`) to isolate connected component clusters, guaranteeing **0% identity leakage** between Train/Val/Test splits.
5. **Layer-wise Learning Rate Decay (LLRD)**: Applies stage-decayed learning rates across ConvNeXt stages (`2e-6` stem/stages 0-1 $\rightarrow$ `5e-6` stage 2 $\rightarrow$ `1e-5` stage 3 $\rightarrow$ `1e-4` head), preserving foundational visual edge filters.
6. **Per-Batch `CosineAnnealingLR` & Linear Warmup**: Steps `CosineAnnealingLR` per-batch (`T_max = total_steps`) with AMP `GradScaler` skip guards and a 500-step Phase 1 `LinearLR` warmup.
7. **Test-Time Augmentation (TTA)**: Computes dual-pass predictions on original and horizontally flipped inputs (`torch.flip(imgs, dims=[-1])`) during evaluation to maximize detection accuracy.

---

## 📊 Benchmark Performance (FaceForensics++ C23 & Celeb-DF v2)

Performance comparison under **Stratified GroupKFold by Video-ID** (zero identity leakage):

| Model / Architecture | Input Resolution | AUC | Accuracy | Notes / Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Standard CNNs (ResNet-50 / VGG16)** | $224 \times 224$ | `0.810 - 0.850` | `75.0% - 81.0%` | Baseline spatial frame classifiers |
| **Spatial-Only ConvNeXt-Base** | $256 \times 256$ | `0.932` | `87.5%` | Spatial stream only (no FFT branch) |
| **Xception Baseline** | $299 \times 299$ | `0.950` | `89.3%` | *Rossler et al., ICCV 2019* (FF++ Benchmark Paper) |
| **Dual-Stream ConvNeXt + 2D FFT + LoRA** | $512 \times 512$ | `0.903+` | `87.1%+` | **Pre-Downsample FFT + LoRA + Temporal Transformer** |

### Benchmark Reproduction Commands

Run automated benchmark suites directly via `benchmark.py`:

```bash
# Fast Stratified Validation Benchmark (500 videos)
python benchmark.py --mode fast

# Full Paper Benchmark (Celeb-DF v2 + 5-Fold Leave-One-Type-Out)
python benchmark.py --mode paper
```

---

## ⚡ Quick Start

### 1. Local Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Automated Integration Test Suite (28+ Tests)

```bash
pytest tests/ -v
```

### 3. Run Streamlit Web Application Locally

```bash
streamlit run app.py
```

---

## 🏋️ GPU Training Execution

Self-contained training execution in `deepfake_detection_v2_pytorch.ipynb` or via CLI:

```bash
python train.py --epochs_phase1 3 --epochs_phase2 15
```

Execution Pipeline:
- **Phase 1**: Head warmup with 500-step linear learning rate warmup (`lr=1e-4`).
- **Phase 2**: LoRA parameter-efficient fine-tuning with per-batch `CosineAnnealingLR` and AMP FP16 scaling (`patience=2`).
- **Calibration**: Calculates optimal decision threshold `T*` via Macro F1 calibration and reports Equal Error Rate (EER).

---

## 🐳 Docker Deployment

Build and run the production containerized web application:

```bash
# Build Docker image
docker build -t deepfake-detector .

# Run container on port 8501
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the application interface at `http://localhost:8501`.

---

## 📁 Directory Structure

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
├── tests/                     # Automated integration test suite (28+ tests)
├── deepfake_detection_v2_pytorch.ipynb # GPU Training Notebook
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production Docker container definition
└── README.md                  # Project documentation
```

---

## 👤 Author & License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
- **HuggingFace Hub**: [yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
