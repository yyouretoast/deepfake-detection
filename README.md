---
title: Deepfake Detection Engine
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Dual-Stream Deepfake Detection Engine

A PyTorch dual-stream deepfake detection framework fusing a ConvNeXt-Small spatial backbone with Steganographic Rich Model (SRM) and Bayar-Stamm 2D Real FFT spectral decomposition, calibrated with SciPy L-BFGS-B log-temperature scaling.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-005CED?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![pytest](https://img.shields.io/badge/pytest-99%2F99%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live Interactive Space**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)  
**GitHub Repository**: [https://github.com/yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)

---

## 4-Panel Interpretability Diagnostics

The detection engine exposes intermediate representations across both spatial and frequency domains simultaneously:

![4-Panel Forensic Diagnostics](figures/attention_maps/attention_map_05_fake.png)

*Figure 1: Dual-domain forensic diagnosis on a Celeb-DF v2 synthesized face ($p = 1.0000$, logit $z = +26.72$). (a) Aligned RGB facial crop via YuNet 5-point landmark similarity transform. (b) 9-filter Steganographic Rich Model (SRM) high-pass noise residual map isolating boundary blending seams. (c) 2D Real FFT log-magnitude spectrum exposing periodic Fourier upsampling harmonics (gating weight $g = 0.207$). (d) ConvNeXt-Small Grad-CAM overlay localizing spatial mask manipulation on facial contours.*

---

## Table of Contents

* [System Architecture](#system-architecture)
* [Quickstart & Python API](#quickstart--python-inference-api)
* [Dataset Composition & Graph Partitioning](#dataset-composition--graph-partitioning)
* [Empirical Benchmarks](#empirical-benchmarks)
* [Forensic Threat Model & Robustness](#forensic-threat-model--robustness)
* [Hardware Latency & Profiling](#hardware-latency--profiling)
* [CLI Reference & Reproduction](#cli-reference--reproduction-commands)
* [Repository Architecture](#repository-architecture)
* [Dataset Compliance & Citations](#dataset-licensing--compliance)

---

## System Architecture

```text
[ Input Video / Image Stream ]
               │
               ▼
   [ OpenCV YuNet 5-Point Alignment & Dynamic Crop (1.50x Box Expansion) ]
               │
               ▼  Unnormalized RGB Tensor [B, 3, 256, 256] ∈ [0.0, 1.0]
       ┌───────┴────────────────────────────────────────┐
       ▼                                                ▼
[ Spatial Stream ]                              [ Frequency Stream ]
• ImageNet Normalization (internal)             • 3 Fixed SRM High-Pass Kernels (9 ch)
• ConvNeXt-Small Backbone                       • 1 Learnable Bayar-Stamm Conv (1 ch)
• LayerNorm2d Feature Normalization             • 2D Real FFT (torch.fft.fft2, FP32)
• 512-d Spatial Embedding (f_s)                 • 10 Log-Mag + 10 Phase Angle Maps
                                                • 2-Layer Conv2d + AdaptiveAvgPool
                                                • 512-d Spectral Embedding (f_f)
       │                                                │
       └───────────────────────┬────────────────────────┘
                               ▼
            [ Symmetric Gated Residual Fusion ]
            • Gating: g = Sigmoid(Linear(1024, 512)([f_s || f_f]))
            • Fused Feature: f_fused = [f_s * (1 - g) || f_f * g] ∈ R^1024
                               │
                               ▼
                   [ Binary Classification Head ]
                   • Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
                   • Raw Logit Output z
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[ Temperature Calibration ]                 [ 4-Panel Diagnostic Engine ]
• Scaled Logit: z / T* (T* = 2.2018)        • (a) Aligned RGB Face Crop
• Probability: p = Sigmoid(z / T*)          • (b) SRM Noise Residual Map
• Operating Threshold: 0.0100               • (c) 2D FFT Magnitude Spectrum
• ECE: 0.0482 -> 0.0195                     • (d) ConvNeXt Grad-CAM Overlay
```

### Mathematical Formulations

1. **20-Channel Spectral Decomposition**:
   Noise residuals from 3 fixed $5\times5$ SRM filters ($9$ channels) and $1$ learnable Bayar-Stamm constrained convolution ($1$ channel) are passed to an orthonormal 2D Real Fast Fourier Transform:
   $$\mathcal{F}_{\text{norm}} = \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{SRM+Bayar}})| + 1 \right)$$
   Phase angles are computed with sub-epsilon magnitude autograd masking to eliminate infinite gradient singularities ($\frac{-y}{x^2 + y^2} \to \infty$):
   $$\theta = \frac{1}{\pi} \text{atan2}(I_{\text{imag}}, I_{\text{real}}) \quad \text{where} \quad |z| \ge 10^{-6}$$

2. **Symmetric Gated Residual Fusion**:
   Both streams are symmetrically gated so neither branch dominates early optimization or starves the frequency stream of gradient flow:
   $$\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{f}_s \parallel \mathbf{f}_f] + \mathbf{b}_g\right) \in \mathbb{R}^{512}$$
   $$\mathbf{f}_{\text{fused}} = \left[ \mathbf{f}_s \odot (1 - \mathbf{g}) \;\parallel\; \mathbf{f}_f \odot \mathbf{g} \right] \in \mathbb{R}^{1024}$$

3. **Temporal Softmax-Weighted Aggregation (Video Sequences)**:
   For multi-frame video inference, frame probabilities are pooled via temperature-scaled softmax weighting ($\tau = 0.10$), concentrating mass on high-confidence manipulation frames:
   $$S_{\text{video}} = \sum_{k=1}^K w_k \cdot p_k \quad \text{where} \quad w_k = \frac{e^{p_k / \tau}}{\sum_{j=1}^K e^{p_j / \tau}}$$

---

## Quickstart & Python Inference API

### 1. Installation & Environment Requirements

* **OS**: Linux, macOS, or Windows 10/11
* **Python**: 3.10 to 3.12
* **Hardware**: CUDA 11.8+ / 12.1+ GPU (minimum 6 GB VRAM for 256×256 inference; 16 GB for multi-GPU DDP training). CPU inference is fully supported.

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Python Inference (5-Line API)

```python
import torch
from src.models import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict

model = HybridDeepfakeDetector(pretrained=False).eval()
ckpt = torch.load("dual_stream_calibrated.pth", map_location="cpu", weights_only=False)
model.load_state_dict(clean_state_dict(ckpt.get("model_state_dict", ckpt)), strict=False)

# Input: Unnormalized RGB float tensor in [0.0, 1.0] range, shape [B, 3, 256, 256]
x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    logits = model(x)
    calibrated_prob = torch.sigmoid(logits / 2.2018).item()

print(f"Deepfake Probability: {calibrated_prob:.4f}")
```

### 3. Launch Local Streamlit Serving UI

```bash
streamlit run app.py
```
App opens at `http://localhost:8501` with support for live webcam, MP4 video uploads, and 4-panel diagnostic rendering.

---

## Dataset Composition & Graph Partitioning

To guarantee **100% zero identity leakage**, actor IDs (`id0_id16`) are partitioned using `networkx.Graph` connected-component subgraphs. Mutually interacting actor clusters are routed exclusively to a single split, guaranteeing $\text{Actors}_{\text{train}} \cap \text{Actors}_{\text{val}} \cap \text{Actors}_{\text{test}} = \emptyset$.

| Split | Total Samples | % of Dataset | Real Faces | Fake Faces | Fake:Real Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 79,097 | 69.2% | 18,373 | 60,724 | 3.31 : 1 |
| **Validation** | 20,544 | 18.0% | 2,112 | 18,432 | 8.73 : 1 |
| **Test** | 14,688 | 12.8% | 2,184 | 12,504 | 5.73 : 1 |
| **Total** | **114,329** | **100.0%** | **22,669** | **91,660** | **4.04 : 1** |

---

## Empirical Benchmarks

### 1. Main Full-Scale Model Evaluation (`dual_stream_calibrated.pth`)

Evaluated on the held-out test split (14,688 facial crops) across 2× NVIDIA Tesla T4 GPUs.

| Evaluation Metric | Measured Value | 95% Non-Parametric Bootstrap CI |
| :--- | :---: | :---: |
| **Test ROC AUC** | **`0.9883`** | `[0.9869, 0.9896]` |
| **Macro F1-Score** | **`0.9759`** | — |
| **Fake Precision** | **`97.35%`** | — |
| **Fake Recall** | **`97.83%`** | — |
| **Overall Accuracy** | **`96.22%`** | — |
| **Optimal Temperature ($T^*$)** | `2.2018` | SciPy L-BFGS-B optimization |
| **Calibrated ECE** | **`0.0195` (1.95%)** | Down from `0.0482` (4.82% uncalibrated) |
| **Operating Decision Threshold** | `0.0100` | Calibrated probability boundary |

### 2. In-Distribution Per-Generator Sub-Domain Breakdown

Evaluated against 2,184 authentic real face crops from the held-out test split.

| Generator Sub-Domain | Manipulation Family | Test ROC AUC | Fake Recall (p > 0.01) |
| :--- | :--- | :---: | :---: |
| **Celeb-DF v2** | High-Quality DeepFake Synthesis | **`0.9998`** | 99.98% |
| **FF++ Face2Face** | Facial Reenactment (Pairs 100–399) | **`1.0000`** | 100.00% |
| **FF++ NeuralTextures** | Neural Texture Rendering (Pairs 600–799) | **`0.9986`** | 100.00% |
| **FF++ Deepfakes** | Autoencoder Face Replacement (Pairs 0–99) | **`0.9839`** | 98.75% |
| **FF++ FaceSwap** | Classical Graphics Face Swapping (Pairs 400–599) | **`0.9599`** | 97.50% |

### 3. Leave-One-Target-Out (LOTO) Cross-Generator Generalization

To test whether the network memorizes specific generation artifacts or learns general forensic anomalies, a 5-Fold LOTO experiment was conducted by systematically excluding one generator family entirely from training:

| LOTO Fold | Excluded Holdout Generator | Zero-Shot AUC | Zero-Shot F1 | Precision | Recall | Threshold | Temp ($T^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `FF++ Deepfakes` (Pairs 0–99) | **`0.9507`** | **`0.8264`** | 76.64% | **`89.67%`** | 0.12 | 2.5082 |
| **Fold 2** | `FF++ Face2Face` (Pairs 100–399) | **`0.9292`** | **`0.8151`** | 77.80% | **`85.58%`** | 0.14 | 1.8170 |
| **Fold 3** | `FF++ FaceSwap` (Pairs 400–599) | **`0.9662`** | `0.8969` | 93.37% | 86.29% | 0.11 | 3.6684 |
| **Fold 4** | `FF++ NeuralTextures` (Pairs 600–799) | **`0.9783`** | `0.9230` | 92.44% | 92.17% | 0.12 | 2.3930 |
| **Fold 5** | `Celeb-DF v2` (Cross-Dataset) | `0.3234` | `0.1202` | 95.42% | 6.41% | 0.12 | 3.1047 |

> [!NOTE]
> **Fold 5 Cross-Dataset Domain Shift**: Excluding Celeb-DF v2 removes 88% of fake training crops, leaving only H.264-compressed FaceForensics++ clips for training. Because Celeb-DF contains uncompressed studio clips, the spectral stream mistakes the clean sensor noise for authentic faces, resulting in anti-correlated ranking ($1 - p = \mathbf{0.6766}$). This documents an authentic physical boundary condition of frequency-domain steganalysis under extreme codec shift.

![LOTO Generalization](figures/loto_generalization.png)

---

## Forensic Threat Model & Robustness

In real-world deployment, adversaries attempt to bypass forensic detection by applying post-processing transformations to erase steganographic artifacts. We evaluated the calibrated model across 4 degradation attacks on the full held-out test split:

![Robustness Degradation](figures/robustness_degradation.png)

### Quantitative Degradation Breakdown

| Perturbation Attack | Severity Parameter | ROC AUC | F1-Score | $\Delta$ AUC Relative to Clean |
| :--- | :--- | :---: | :---: | :---: |
| **Clean Baseline** | Unperturbed | `0.9883` | `0.9759` | — |
| **JPEG Compression** | Quality = 90 | `0.9872` | `0.9741` | −0.11% |
| **JPEG Compression** | Quality = 50 (Social Media Recompression) | `0.9685` | `0.9279` | −1.98% (High Resilience) |
| **JPEG Compression** | Quality = 30 (Aggressive Compression) | `0.9335` | `0.8528` | −5.48% |
| **Spatial Downscale** | Scale = 0.50× | `0.9780` | `0.9124` | −1.03% |
| **Spatial Downscale** | Scale = 0.25× | `0.9418` | `0.8590` | −4.65% |
| **Gaussian Noise** | $\sigma = 15$ | `0.8844` | `0.8732` | −10.39% |
| **Gaussian Noise** | $\sigma = 30$ (Wideband Sensor Noise) | `0.7544` | `0.8479` | **−23.39%** (Noise Vulnerability) |
| **Gaussian Blur** | $\sigma = 1.5$ | `0.9620` | `0.8540` | −2.63% |
| **Gaussian Blur** | $\sigma = 3.0$ (Aggressive Low-Pass Filtering) | `0.7375` | `0.8411` | **−25.08%** (Most Vulnerable) |

* **Why Gaussian Blur is the Most Damaging**: Spatial smoothing acts as an aggressive low-pass filter, attenuating the sub-pixel high-frequency SRM noise residuals and Fourier upsampling harmonics that the frequency stream relies upon.

---

## Hardware Latency & Profiling

*Evaluated at 256×256 facial crop resolution across PyTorch 2.1 FP16 / FP32 execution providers.*

| Execution Device | Precision | Batch Size | Latency per Crop | Processing Throughput | Benchmark Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 1 | `18.62 ms` | `53.7 FPS` | Kaggle Dual-T4 Kernel |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 32 | `16.41 ms` | `60.9 FPS` | Kaggle Dual-T4 Kernel |
| **Intel Xeon CPU (Multi-thread)** | FP32 | BS = 1 | `188.25 ms` | `5.3 FPS` | Local Multi-threaded Host |
| **Intel Xeon CPU (Multi-thread)** | FP32 | BS = 32 | `4.77 ms` | `209.6 FPS` | Local Multi-threaded Host |

---

## CLI Reference & Reproduction Commands

All scripts feature standard `argparse` CLI interfaces:

### 1. Full Unit Test Suite (99 Tests)
```bash
pytest tests/ -v
```

### 2. Distributed Training (Multi-GPU DDP)
```bash
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_dual_stream_ddp.py \
    --data_dir /path/to/dataset \
    --epochs 5 \
    --batch_size 16 \
    --save_path /kaggle/working/dual_stream_best.pth
```

### 3. Leave-One-Target-Out (LOTO) Training
```bash
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_loto_experiment.py \
    --holdout neuraltextures \
    --epochs 3 \
    --batch_size 16
```

### 4. Held-Out Evaluation & Temperature Fitting
```bash
python scripts/evaluate_test_set.py \
    --weights_path /kaggle/working/dual_stream_best.pth
```

### 5. Subdomain Breakdown Evaluation
```bash
python scripts/evaluate_subdomain_breakdown.py \
    --weights_path /kaggle/working/dual_stream_calibrated.pth
```

### 6. Robustness Degradation Stress Testing
```bash
python scripts/evaluate_robustness.py \
    --checkpoint /kaggle/working/dual_stream_calibrated.pth \
    --data_root /path/to/dataset \
    --output_json /kaggle/working/robustness_results.json
```

### 7. Export Model to ONNX
```bash
python scripts/export_onnx.py \
    --weights dual_stream_calibrated.pth \
    --output models/dual_stream.onnx \
    --img_size 256
```

### 8. Benchmark Latency & Throughput
```bash
python scripts/benchmark_latency.py \
    --weights dual_stream_calibrated.pth \
    --img_size 256 \
    --batch_size 1 \
    --device cuda
```

### 9. Render 4-Panel Diagnostic Figures
```bash
python scripts/visualize_attention_maps.py \
    --checkpoint dual_stream_calibrated.pth \
    --output_dir figures/attention_maps \
    --n_samples 6
```

---

## Repository Architecture

```text
deepfake-detection/
├── app.py                         # Streamlit web application & serving dashboard
├── config/
│   └── default.yaml               # Hyperparameter & resolution configuration
├── figures/                       # Publication-grade evaluation figures
│   ├── attention_maps/            # 4-panel diagnostic PNG figures
│   ├── ece_reliability.png        # Calibration reliability diagram
│   ├── loto_generalization.png    # LOTO zero-shot AUC bar chart
│   ├── per_generator_auc.png      # Per-generator sub-domain AUC chart
│   ├── robustness_degradation.png # 4-panel degradation perturbation curves
│   └── roc_curve.png              # Held-out test set ROC curve
├── notebooks/
│   └── master_pipeline.ipynb      # End-to-end research training & analysis notebook
├── scripts/                       # Thin executable CLI entry points
│   ├── benchmark_latency.py       # Inference latency & throughput benchmarking
│   ├── evaluate_robustness.py     # Degradation perturbation sweeps
│   ├── evaluate_subdomain_breakdown.py # Subdomain breakdown evaluation
│   ├── evaluate_test_set.py       # Held-out test evaluation & temperature fitting
│   ├── export_onnx.py             # ONNX format export
│   ├── export_test_predictions.py # Sample predictions JSON exporter
│   ├── extract_face_crops.py      # Multi-threaded YuNet face crop extraction
│   ├── generate_benchmark_plots.py# 300 DPI plot rendering
│   ├── rebalance_splits.py        # Split manifest re-balancer
│   ├── train_dual_stream_ddp.py   # Multi-GPU DDP training runner
│   ├── train_loto_experiment.py   # LOTO cross-generator training runner
│   └── visualize_attention_maps.py# Diagnostic map rendering runner
├── src/                           # Modular core library
│   ├── dataset/
│   │   ├── datasets.py            # Unified FaceCropDataset with valid-flag masking
│   │   ├── domains.py             # DomainClassifier & manipulation taxonomy
│   │   ├── loader.py              # Identity-disjoint graph component partitioner
│   │   ├── preprocess.py          # DynamicFaceCropper & 5-point similarity alignment
│   │   └── resolver.py            # DatasetResolver for paths and manifests
│   ├── evaluation/
│   │   ├── evaluator.py           # ModelEvaluator with TTA & AMP autocast
│   │   └── metrics.py             # Safe AUC, ECE, & classification metrics
│   ├── models/
│   │   ├── fusion.py              # LayerNorm2d, GatedResidualFusion, ClassifierHead
│   │   ├── hybrid_detector.py     # Dual-stream hybrid detector architecture
│   │   ├── spectral.py            # RealFFT2DModule with sub-epsilon autograd
│   │   └── steganography.py       # SRMConv2d & BayarConv2d high-pass filters
│   ├── services/
│   │   ├── ui_components.py       # Decoupled Streamlit visual rendering components
│   │   └── video_engine.py        # Video prediction engine & sequential seek
│   ├── training/
│   │   ├── ema.py                 # ExponentialMovingAverage with context manager
│   │   ├── loss.py                # FocalLossWithLogits & MaskedBCEWithLogits
│   │   ├── optimization.py        # Differential param groups & lr schedulers
│   │   └── trainer.py             # DualStreamTrainer distributed execution engine
│   └── utils/
│       ├── checkpoint.py          # State dict cleaning & temperature scaling
│       ├── interpretability.py    # Thread-safe ConvNeXt Grad-CAM implementation
│       └── temporal_aggregation.py# Softmax, Top-k, EMA temporal frame pooling
├── tests/                         # Full PyTest test suite (99 passing tests)
├── LICENSE                        # MIT License
├── pyproject.toml                 # Ruff & pytest configuration
├── README.md                      # Authoritative single-source-of-truth documentation
└── requirements.txt               # Pinned project dependencies
```

---

## Dataset Licensing & Compliance

This repository was evaluated on **FaceForensics++** and **Celeb-DF v2**:
* **FaceForensics++**: Rössler et al., *IEEE/CVF ICCV 2019*. Access granted under the FaceForensics Non-Commercial Research Agreement.
* **Celeb-DF v2**: Li et al., *IEEE/CVF CVPR 2020*. Access granted under the Celeb-DF Release Agreement.

Model weights and code are provided solely for non-commercial academic research, forensic verification, and reproducible evaluation.

---

## Academic References

1. **ConvNeXt**: Liu, Z., et al. (2022). *A ConvNet for the 2020s*. IEEE/CVF CVPR.
2. **Steganographic Rich Model (SRM)**: Fridrich, J., & Kodovsky, J. (2012). *Rich models for steganalysis of digital images*. IEEE TIFS.
3. **Bayar-Stamm Constrained Convolution**: Bayar, B., & Stamm, M. C. (2016). *A deep learning approach to universal image manipulation detection*. IEEE IH&MMSec.
4. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE/CVF ICCV.
5. **Temperature Scaling Calibration**: Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
6. **FaceForensics++**: Rössler, A., et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. IEEE/CVF ICCV.
7. **Celeb-DF**: Li, Y., et al. (2020). *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*. IEEE/CVF CVPR.
