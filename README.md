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

A PyTorch 2.x dual-stream deepfake detection pipeline combining ConvNeXt-Small spatial representations with Steganographic Rich Model (SRM) + Bayar-Stamm 2D Real FFT frequency spectrum embeddings and thread-safe OpenCV YuNet face detection.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-005CED?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate)
[![pytest](https://img.shields.io/badge/pytest-20%2F20%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live Demo**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## System Architecture

```
[ Input Video ] ──► [ Thread-Local YuNet ] ──► [ 512x512 Face Crops (1.50x Scale Margin) ]
                                                       │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ Spatial Stream (ConvNeXt-Small) ]                     [ Frequency Stream (SRM + Bayar + 2D FFT) ]
         • 256x256 RGB Image Input                                • 3 SRM Filters + 1 Bayar-Stamm Conv
         • 512-d Feature Embeddings                               • 20-Channel 2D FFT (10 Mag + 10 Phase)
                       │                                          • 512-d Frequency Embeddings
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       [ Equalized Residual Gated Fusion ]
                                       • Gating Vector g = Sigmoid(Linear(Spatial || Freq))
                                       • Fused Feature f_fused = [f_spatial_512 || f_freq_512 * g]
                                       • 1024-d Equalized Feature Vector
                                                      │
                                                      ▼
                                           [ Binary Classifier Head ]
                                           • Linear(1024, 256) + Dropout(0.3)
                                           • Binary Logit Output
```

### Formulations

**20-Channel Dual-Domain Spectral Extraction**:

$$
\mathcal{F}_{\text{norm}} = \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{SRM+Bayar}})| + 1 \right)
$$

**Residual Gated Fusion**:

$$
g = \sigma\Big(\text{Linear}\big([\mathbf{f}_{\text{spatial}} \;\|\; \mathbf{f}_{\text{freq}}]\big)\Big) \in \mathbb{R}^{512}
$$

$$
\mathbf{f}_{\text{fused}} = \Big[ \mathbf{f}_{\text{spatial}} \;\|\; \mathbf{f}_{\text{freq}} \odot g \Big] \in \mathbb{R}^{1024}
$$

---

## Core Engineering Features

- **Zero Identity Data Leakage**: Graph connected-component partitioning (`networkx.Graph`) segregates actor IDs across train, val, and test splits.
- **Steganographic SRM + Bayar Noise Residuals**: Combines 3 fixed SRM high-pass kernels with 1 learnable Bayar-Stamm constrained convolution to isolate spatial noise residuals before FFT extraction.
- **Multi-GPU DDP Engine**: Hugging Face `Accelerate` DistributedDataParallel with `SyncBatchNorm` and OpenCV C++ binary loader.
- **Per-Sample Loss Masking**: Excludes corrupt or invalid image frames from backpropagation gradient updates.

---

## Benchmark & Experimental Results

### 1. Held-Out Test Set Metrics (10,528 Crops)
- **Test AUC**: `0.9987`
- **Test F1-Score**: `0.9830`
- **Precision (Fake)**: `0.9686`
- **Recall (Fake)**: `0.9979`
- **Optimal Temperature ($T^*$)**: `1.4788`
- **Expected Calibration Error (ECE)**: `0.0122` (Raw) $\rightarrow$ `0.0093` (Calibrated)

### 2. Per-Generator Sub-Domain Evaluation (2-Class AUC vs Real Faces)

| Generator Sub-Domain | Sample Count | AUC | F1-Score | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Celeb-DF v2 Synthesis** | 6,639 | `0.9992` | `0.9630` | `99.97%` |
| **FF++ Deepfakes (Pairs 0-199)** | 200 | `0.9963` | `0.4405` | `100.00%` |
| **FF++ Face2Face (Pairs 200-399)** | 200 | `0.9967` | `0.4405` | `100.00%` |
| **FF++ FaceSwap (Pairs 400-599)** | 200 | `0.9961` | `0.4405` | `100.00%` |
| **FF++ NeuralTextures (Pairs 600-799)** | 200 | `0.9940` | `0.4405` | `100.00%` |

### 3. True LOTO (Leave-One-Type-Out) Cross-Generator Generalization

| Experiment Fold | Held-Out Target Domain | Category Type | Test Samples | Zero-Shot AUC | Zero-Shot F1 | Precision | Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `FF++ NeuralTextures` | Within-Dataset Cross-Generator | 5,289 (2.4k Fakes / 2.9k Reals) | `0.9783` | `0.9230` | `0.9244` | `0.9217` |
| **Fold 2** | `Celeb-DF v2` | Cross-Dataset Zero-Shot Transfer | 9,528 (6.6k Fakes / 2.9k Reals) | *[Running on Kaggle...]* | *[Running...]* | *[Running...]* | *[Running...]* |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Launch Web Application

```bash
streamlit run app.py
```

### 4. Multi-GPU Training

```bash
accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_dual_stream_ddp.py
```

---

## Repository Structure

```
deepfake-detection/
├── app.py                         # Streamlit web interface
├── config/
│   └── default.yaml               # Configuration parameters (img_size: 512, scale_factor: 1.50)
├── src/
│   ├── config.py                  # Configuration parser
│   ├── dataset/
│   │   ├── loader.py              # Identity-safe graph splitter & loader
│   │   └── preprocess.py          # DynamicFaceCropper & 5-point similarity alignment
│   └── models/
│       └── hybrid_detector.py     # ConvNeXt-Small + SRM/Bayar 2D FFT architecture
├── scripts/                       # Production training, face cropping, and ONNX export
│   ├── extract_face_crops.py      # Thread-pool multi-threaded face cropper
│   ├── train_dual_stream_ddp.py   # Multi-GPU DDP training pipeline
│   └── export_onnx.py             # ONNX FP16/UINT8 exporter
├── tests/                         # 20/20 passing unit tests
└── README.md
```

---

## License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
