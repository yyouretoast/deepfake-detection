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

A PyTorch dual-stream deepfake detection architecture fusing a ConvNeXt-Small spatial backbone with Steganographic Rich Model (SRM) and Bayar-Stamm 2D Real FFT spectral decomposition, calibrated with SciPy L-BFGS-B log-temperature scaling.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-005CED?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![pytest](https://img.shields.io/badge/pytest-99%2F99%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live Interactive Space**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)  
**GitHub Repository**: [https://github.com/yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)

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
• Scaled Logit: z / T* (T* = 1.4788)        • (a) Aligned RGB Face Crop
• Probability: p = Sigmoid(z / T*)          • (b) SRM Noise Residual Map
• Operating Threshold: 0.01                 • (c) 2D FFT Magnitude Spectrum
• ECE: 0.0122 -> 0.0093                     • (d) ConvNeXt Grad-CAM Overlay
```

### Mathematical Specifications

1. **20-Channel Spectral Decomposition**:
   $$\mathcal{F}_{\text{norm}} = \ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{SRM+Bayar}})| + 1 \right)$$
   Phase angles are computed via sub-epsilon magnitude masked autograd to prevent infinite gradients:
   $$\theta = \text{atan2}(I_{\text{imag}}, I_{\text{real}}) / \pi \quad \text{where} \quad |z| \ge 10^{-6}$$

2. **Symmetric Gated Residual Fusion**:
   $$\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{f}_s \parallel \mathbf{f}_f] + \mathbf{b}_g\right) \in \mathbb{R}^{512}$$
   $$\mathbf{f}_{\text{fused}} = \left[ \mathbf{f}_s \odot (1 - \mathbf{g}) \;\parallel\; \mathbf{f}_f \odot \mathbf{g} \right] \in \mathbb{R}^{1024}$$

3. **Temporal Softmax Pooling (Video Inference)**:
   $$S_{\text{video}} = \sum_{k=1}^K w_k \cdot p_k \quad \text{where} \quad w_k = \frac{e^{p_k / \tau}}{\sum_{j=1}^K e^{p_j / \tau}}, \quad \tau = 0.10$$

---

## Quickstart & Python Inference API

### 1. Installation

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
    calibrated_prob = torch.sigmoid(logits / 1.4788).item()

print(f"Deepfake Probability: {calibrated_prob:.4f}")
```

### 3. Launch Local Streamlit Web UI

```bash
streamlit run app.py
```
App opens at `http://localhost:8501`.

---

## Empirical Benchmarks & Experimental Results

*All evaluations conducted with PyTorch FP16 mixed precision on 2x NVIDIA Tesla T4 GPUs at 256x256 resolution.*

### 1. Held-Out Test Set Performance (10,528 Face Crops)

Identity-disjoint partition (`networkx.Graph` connected components on actor IDs).

| Metric | Measured Value | 95% Non-Parametric Bootstrap CI |
| :--- | :---: | :---: |
| **ROC AUC** | **`0.9988`** | `[0.9985, 0.9991]` |
| **F1-Score** | **`0.9830`** | `[0.9809, 0.9850]` |
| **Precision (Fake)** | `0.9686` | `[0.9647, 0.9725]` |
| **Recall (Fake)** | `0.9979` | `[0.9966, 0.9987]` |
| **Optimal Temperature ($T^*$)** | `1.4788` | L-BFGS-B on validation split |
| **Expected Calibration Error (ECE)** | `0.0122` $\to$ **`0.0093`** | −23.8% calibration error |
| **Operating Decision Threshold** | `0.0100` | Calibrated probability threshold |

### 2. Per-Generator Sub-Domain Evaluation

Evaluated against 2,889 real face crops from the held-out test split.

| Generator Sub-Domain | Fake Samples | ROC AUC | Recall (p > 0.01) | F1-Score* |
| :--- | :---: | :---: | :---: | :---: |
| **Celeb-DF v2 Synthesis** | 6,639 | `0.9992` | 99.97% | `0.9630` |
| **FF++ Deepfakes (Pairs 0–99)** | 200 | `0.9963` | 100.00% | `0.4405` |
| **FF++ Face2Face (Pairs 100–399)** | 200 | `0.9967` | 100.00% | `0.4405` |
| **FF++ FaceSwap (Pairs 400–599)** | 200 | `0.9961` | 100.00% | `0.4405` |
| **FF++ NeuralTextures (Pairs 600–799)** | 200 | `0.9940` | 100.00% | `0.4405` |

*\*Note: Sub-domain F1 scores reflect extreme test set class imbalance (200 fakes vs 2,889 reals) at threshold 0.01. ROC AUC is independent of class balance.*

### 3. Leave-One-Target-Out (LOTO) Cross-Generator Generalization

| LOTO Fold | Excluded Holdout Generator | Category | Test Samples | Zero-Shot AUC | Inverted AUC (1 - p) | Zero-Shot F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | `FF++ Deepfakes` | Within-Dataset LOTO | 5,289 | **`0.9691`** | `0.0309` | **`0.9065`** |
| **Fold 2** | `FF++ Face2Face` | Within-Dataset LOTO | 5,289 | **`0.9749`** | `0.0251` | **`0.9179`** |
| **Fold 3** | `FF++ FaceSwap` | Within-Dataset LOTO | 5,289 | **`0.9662`** | `0.0338` | **`0.8969`** |
| **Fold 4** | `FF++ NeuralTextures` | Within-Dataset LOTO | 5,289 | **`0.9783`** | `0.0217` | **`0.9230`** |
| **Fold 5** | `Celeb-DF v2` | Cross-Dataset Shift | 82,549 | `0.3234` | **`0.6766`** | `0.1202` |

> [!NOTE]
> **Fold 5 Physical Mechanism**: Holding out Celeb-DF v2 removes 88% of fake training crops, training exclusively on H.264-compressed FaceForensics++ clips. Because Celeb-DF contains uncompressed studio clips, the spectral stream mistakes the clean sensor noise for authentic faces, resulting in anti-correlated ranking ($1 - p = 0.6766$).

![LOTO Generalization](figures/loto_generalization.png)

### 4. Robustness Degradation Sweeps

Evaluated across 10,528 test crops at 256x256 resolution.

| Perturbation | Parameter | ROC AUC | F1-Score | $\Delta$ AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Clean Baseline** | None | `0.9988` | `0.9677` | — |
| **JPEG Compression** | Quality = 90 | `0.9971` | `0.9693` | −0.17% |
| **JPEG Compression** | Quality = 50 | `0.9685` | `0.9279` | −3.03% |
| **JPEG Compression** | Quality = 30 | `0.9335` | `0.8528` | −6.53% |
| **Resolution Downscale** | Scale = 0.50× | `0.9910` | `0.9059` | −0.78% |
| **Resolution Downscale** | Scale = 0.25× | `0.9518` | `0.8631` | −4.70% |
| **Gaussian Noise** | $\sigma = 15$ | `0.8844` | `0.8732` | −11.44% |
| **Gaussian Noise** | $\sigma = 30$ | `0.7544` | `0.8479` | **−24.44%** |
| **Gaussian Blur** | $\sigma = 1.5$ | `0.9748` | `0.8675` | −2.40% |
| **Gaussian Blur** | $\sigma = 3.0$ | `0.7375` | `0.8411` | **−26.13%** |

![Robustness Degradation](figures/robustness_degradation.png)

### 5. Hardware Latency & Throughput Profile

| Execution Target | Precision | Batch Size | Latency per Crop | Throughput | Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 1 | `18.62 ms` | `53.7 FPS` | Kaggle GPU Kernel |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 32 | `16.41 ms` | `60.9 FPS` | Kaggle GPU Kernel |
| **Intel Xeon CPU** | FP32 | BS = 1 | `188.25 ms` | `5.3 FPS` | Local Multi-threaded Host |
| **Intel Xeon CPU** | FP32 | BS = 32 | `4.77 ms` | `209.6 FPS` | Local Multi-threaded Host |

---

## 4-Panel Interpretability Diagnostics

The diagnostic pipeline exposes intermediate representations from both streams simultaneously:

![4-Panel Diagnostic Figure](figures/attention_maps/attention_map_05_fake.png)

*Figure: Diagnostic breakdown of a Celeb-DF v2 fake face ($p = 1.0000, z = +26.72$). (a) Aligned RGB crop, (b) SRM high-pass noise residual map displaying face-swap boundary truncation, (c) 2D Real FFT log-magnitude spectrum (fusion gate $g = 0.207$), and (d) ConvNeXt-Small Grad-CAM overlay localizing spatial mask artifacts.*

---

## CLI Reference & Reproduction Commands

All scripts include standardized CLI argument parsers:

### 1. Unit Test Suite (99 Tests)
```bash
pytest tests/ -v
```

### 2. Distributed Training (Multi-GPU DDP)
```bash
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_dual_stream_ddp.py \
    --data_dir /path/to/dataset \
    --epochs 5 \
    --batch_size 16 \
    --save_path models/dual_stream_best.pth
```

### 3. Leave-One-Target-Out (LOTO) Training
```bash
accelerate launch --mixed_precision fp16 --num_processes 2 scripts/train_loto_experiment.py \
    --holdout neuraltextures \
    --epochs 3 \
    --batch_size 16 \
    --data_dir /path/to/dataset
```

### 4. Test Set Evaluation & Calibration
```bash
python scripts/evaluate_test_set.py \
    --data_dir /path/to/dataset \
    --weights_path models/dual_stream_best.pth
```

### 5. Subdomain Breakdown Evaluation
```bash
python scripts/evaluate_subdomain_breakdown.py \
    --data_dir /path/to/dataset \
    --weights_path dual_stream_calibrated.pth
```

### 6. Robustness Stress Testing
```bash
python scripts/evaluate_robustness.py \
    --checkpoint dual_stream_calibrated.pth \
    --data_root /path/to/dataset \
    --output_json robustness_results.json
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

### 9. Render 4-Panel Attention Maps
```bash
python scripts/visualize_attention_maps.py \
    --checkpoint dual_stream_calibrated.pth \
    --data_root /path/to/dataset \
    --output_dir figures/attention_maps \
    --n_samples 6
```

---

## Repository Architecture

```text
deepfake-detection/
├── app.py                         # Streamlit serving application & dashboard
├── config/
│   └── default.yaml               # Model & crop hyperparameter configuration
├── figures/                       # Publication-grade evaluation & diagnostic figures
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
│   ├── evaluate_robustness.py     # Degradation stress testing
│   ├── evaluate_subdomain_breakdown.py # Subdomain evaluation breakdown
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
