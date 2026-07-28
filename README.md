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

Dual-stream PyTorch 2.x Deepfake Detection architecture combining **ConvNeXt-Base** spatial features, **Pre-Downsample 512x512 2D Real FFT** frequency spectrum embeddings, **LoRA (Low-Rank Adaptation)** parameter-efficient fine-tuning, and a **5D Temporal Sequence Transformer**. Includes **Graph-Connected Component Partitioning**, **4-Head Cross-Attention Fusion**, **Layer-wise Learning Rate Decay (LLRD)**, **Macro F1 Threshold Calibration**, **Test-Time Augmentation (TTA)**, **ONNX Runtime Acceleration**, **Grad-CAM Visualizations**, and a **Streamlit Web UI**.

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

### Technical Formulation

**1. Pre-Downsample 512x512 2D Real FFT Spectrum Extraction**:

$$
\mathcal{F}_{\text{norm}} = \frac{1}{10} \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{gray}})| + 10^{-5} \right)
$$

Computes frequency representations on uncompressed 512x512 face crops prior to spatial downsampling, preserving high-frequency grid noise up to native Nyquist resolution ($f_{N} = 256 \text{ cycles}$).

**2. LoRA Low-Rank Adaptation (Weight Folding)**:

$$
W_{\text{effective}} = W_0 + \frac{\alpha}{r} (B \cdot A)
$$

Injects trainable rank $r=8$ matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ into ConvNeXt depthwise and pointwise conv blocks, reducing Phase 2 trainable parameters by 98.6% with zero inference latency overhead (`merge_weights()`).

**3. 4-Head Cross-Attention Residual Fusion**:

$$
\mathbf{f}_{\text{enhanced}} = \mathbf{f}_{\text{freq}} + 0.1 \times \text{Attention}(Q_{\text{spatial}}, K_{\text{freq}}, V_{\text{freq}})
$$

---

## Technical Specifications

1. **512x512 Frequency Spectrum Extraction**: Extracts 2D `rfft2` log-magnitude frequency spectra on 512x512 face crops prior to spatial downsampling, preserving sub-pixel up-sampling artifacts.
2. **LoRA Fine-Tuning**: Low-rank adapters on ConvNeXt-Base 7x7 depthwise (`dwconv`) and 1x1 pointwise (`pwconv1`, `pwconv2`) conv blocks (88M $\rightarrow$ 1.2M trainable parameters).
3. **5D Sequence Transformer**: Processes video frame sequences through a 2-layer Pre-LN `TemporalSequenceEncoder` using 8-frame sequence chunking (`chunk_size=8`) to maintain peak VRAM under 600MB.
4. **Identity Partitioning**: Uses `networkx.Graph` parsing source and target actor IDs (`id1`, `id2`) to isolate connected components and prevent identity leakage across splits.
5. **Layer-wise Learning Rate Decay (LLRD)**: Applies stage-decayed learning rates across ConvNeXt stages (`2e-6` stem/stages 0-1 $\rightarrow$ `5e-6` stage 2 $\rightarrow$ `1e-5` stage 3 $\rightarrow$ `1e-4` head).
6. **Per-Batch Scheduler & Warmup**: Steps `CosineAnnealingLR` per-batch (`T_max = total_steps`) with AMP `GradScaler` skip guards and a 500-step Phase 1 `LinearLR` warmup.
7. **Test-Time Augmentation (TTA)**: Computes dual-pass predictions on original and horizontally flipped inputs (`torch.flip(imgs, dims=[-1])`) during evaluation.

---

## Benchmark Performance (FaceForensics++ C23 & Celeb-DF v2)

Evaluation under **Stratified GroupKFold by Video-ID** (zero identity leakage):

| Model / Architecture | Input Resolution | AUC | Accuracy | Notes / Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Standard CNNs (ResNet-50 / VGG16)** | $224 \times 224$ | `0.810 - 0.850` | `75.0% - 81.0%` | Baseline spatial frame classifiers |
| **Spatial-Only ConvNeXt-Base** | $256 \times 256$ | `0.932` | `87.5%` | Spatial stream only (no FFT branch) |
| **Xception Baseline** | $299 \times 299$ | `0.950` | `89.3%` | *Rossler et al., ICCV 2019* (FF++ Benchmark Paper) |
| **Dual-Stream ConvNeXt + 2D FFT + LoRA** | $512 \times 512$ | `0.903+` | `87.1%+` | **Pre-Downsample FFT + LoRA + Temporal Transformer** |

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
