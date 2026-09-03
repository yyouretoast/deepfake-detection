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

# Dual-Stream Deepfake Detection Framework

A PyTorch dual-stream deepfake detection framework fusing a ConvNeXt-Small spatial backbone with Steganographic Rich Model (SRM) and Bayar-Stamm 2D Real FFT spectral decomposition, calibrated with SciPy L-BFGS-B log-temperature scaling and spatiotemporal sequence modeling.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-005CED?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![pytest](https://img.shields.io/badge/pytest-108%2F108%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Live Interactive Hugging Face Space**](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## 4-Panel Interpretability Diagnostics

Intermediate representations exposed across spatial, residual, and frequency domains simultaneously:

![4-Panel Forensic Diagnostics](figures/attention_maps/attention_map_05_fake.png)

*Figure 1: Dual-domain forensic diagnosis on a Celeb-DF v2 synthesized face ($p = 1.0000$, logit $z = +26.72$). (a) Aligned RGB facial crop via YuNet 5-point landmark similarity transform. (b) 9-filter Steganographic Rich Model (SRM) high-pass noise residual map isolating boundary blending seams. (c) 2D Real FFT log-magnitude spectrum exposing periodic Fourier upsampling harmonics (gating weight $g = 0.207$). (d) ConvNeXt-Small Grad-CAM overlay localizing spatial mask manipulation on facial contours.*

---

## Key Differentiators

* **Dual-Domain Signal Analysis**: Combines semantic spatial features (ConvNeXt-Small) with sub-pixel noise residuals (SRM + Bayar-Stamm) and orthonormal 2D Real FFT spectral decomposition.
* **100% Zero Identity Leakage**: Actor clusters (`id0_id16`) are partitioned using `networkx.Graph` connected components to guarantee $\text{Actors}_{\text{train}} \cap \text{Actors}_{\text{val}} \cap \text{Actors}_{\text{test}} = \emptyset$.
* **Bayesian 3-Zone Decision Bands**: Post-hoc probability calibration ($T^* = 2.2018$) establishes high-precision boundaries $(\tau_{\text{real}}, \tau_{\text{fake}})$, guaranteeing $\ge$ 98% precision on confirmed verdicts and routing ambiguous boundary media to manual inspection.
* **Real-Time Video Engine**: 60.9 FPS inference on an NVIDIA Tesla T4 with an attention-pooled Bidirectional GRU head to detect inter-frame flickering, boundary jitter, and unnatural blink patterns.

---

## Quickstart & Python API

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

### 2. Download Baseline Checkpoint (195 MB)

```bash
wget -O dual_stream_calibrated.pth https://huggingface.co/spaces/yyouretoast/deepfake-detector/resolve/main/models/dual_stream_calibrated.pth
```

### 3. Python Inference Snippet

```python
import torch
from src.models import HybridDeepfakeDetector
from src.utils.checkpoint import clean_state_dict, classify_three_zone

# 1. Load calibrated detector
model = HybridDeepfakeDetector(pretrained=False).eval()
ckpt = torch.load("dual_stream_calibrated.pth", map_location="cpu", weights_only=False)
model.load_state_dict(clean_state_dict(ckpt.get("model_state_dict", ckpt)), strict=False)

# 2. Input unnormalized RGB float tensor [B, 3, 256, 256] in [0.0, 1.0]
x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    logits = model(x)
    temp = float(ckpt.get("temperature", 2.2018))
    prob = float(torch.sigmoid(logits / temp).item())

# 3. Classify into 3-zone forensic certainty
tau_real = float(ckpt.get("tau_real", 0.35))
tau_fake = float(ckpt.get("tau_fake", 0.65))
verdict = classify_three_zone(prob, tau_real=tau_real, tau_fake=tau_fake)

print(f"Deepfake Probability: {prob:.4f}")
print(f"Forensic Verdict:     {verdict['verdict']} (Zone: {verdict['zone']})")
```

### 4. Launch Local Serving Dashboard

```bash
streamlit run app.py
```
Opens interactive UI at `http://localhost:8501` with support for webcam capture, MP4 video uploads, temporal anomaly timelines, and 4-panel diagnostic Grad-CAM rendering.

---

## Empirical Benchmarks

### 1. Published Baseline Evaluation (`dual_stream_calibrated.pth`)

Evaluated on the held-out test split (14,688 facial crops) across 2× NVIDIA Tesla T4 GPUs.

| Evaluation Metric | Measured Value | 95% Non-Parametric Bootstrap CI | Notes |
| :--- | :---: | :---: | :--- |
| **Test ROC AUC** | **`0.9883`** | `[0.9869, 0.9896]` | Full test split discriminative power |
| **Macro F1-Score** | **`0.9759`** | — | Balanced across real and fake classes |
| **Fake Precision** | **`97.35%`** | — | 2.65% false alarm rate on fakes |
| **Fake Recall** | **`97.83%`** | — | Catches 97.83% of synthetic crops |
| **Overall Accuracy** | **`96.22%`** | — | Overall sample classification rate |
| **Calibrated ECE** | **`0.0195` (1.95%)** | — | Down from 4.82% uncalibrated ($-59.5\%$ relative) |
| **Optimal Temperature ($T^*$)** | `2.2018` | — | SciPy L-BFGS-B optimization |
| **Operating Decision Threshold** | `0.0100` | — | Calibrated probability boundary |

> [!NOTE]
> **Baseline Weights vs. Enhanced Rerun**: These published benchmark metrics reflect the original dual-stream baseline checkpoint (ConvNeXt-Small + 2-layer CNN). Retraining via the reproduction suite below upgrades the engine with the 4-stage ResSE-Spectral Tower (~2.98M params), degradation-hardened augmentations, and the spatiotemporal Bi-GRU head.

### 2. In-Distribution Per-Generator Breakdown

Evaluated against 2,184 authentic real face crops from the held-out test split:

| Generator Sub-Domain | Manipulation Family | Test ROC AUC | Fake Recall ($p > 0.01$) |
| :--- | :--- | :---: | :---: |
| **Celeb-DF v2** | High-Quality DeepFake Synthesis | **`0.9998`** | 99.98% |
| **FF++ Face2Face** | Facial Reenactment (Pairs 100–399) | **`1.0000`** | 100.00% |
| **FF++ NeuralTextures** | Neural Texture Rendering (Pairs 600–799) | **`0.9986`** | 100.00% |
| **FF++ Deepfakes** | Autoencoder Face Replacement (Pairs 0–99) | **`0.9839`** | 98.75% |
| **FF++ FaceSwap** | Classical Graphics Face Swapping (Pairs 400–599) | **`0.9599`** | 97.50% |

### 3. Leave-One-Target-Out (LOTO) Cross-Generator Generalization

To evaluate whether the detector memorizes specific generator artifacts or learns general forensic anomalies, a 5-fold LOTO experiment was conducted by systematically excluding one generator family entirely from training:

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

### 4. Robustness Under Real-World Degradation

Evaluated across 4 distortion attacks on the full held-out test split:

![Robustness Degradation](figures/robustness_degradation.png)

| Perturbation Attack | Severity Parameter | ROC AUC | F1-Score | Δ AUC Relative to Clean |
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
| **Gaussian Blur** | $\sigma = 3.0$ (Aggressive Low-Pass Filtering) | `0.7375` | `0.8411` | **−25.08%** (High Vulnerability) |

### 5. Hardware Latency & Profiling

*Evaluated at 256×256 facial crop resolution across PyTorch 2.1 execution providers:*

| Execution Device | Precision | Batch Size | Latency per Crop | Throughput | Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 1 | `18.62 ms` | `53.7 FPS` | Kaggle Dual-T4 Kernel |
| **NVIDIA Tesla T4 GPU** | FP16 | BS = 32 | `16.41 ms` | `60.9 FPS` | Kaggle Dual-T4 Kernel |
| **Intel Xeon CPU (Multi-thread)** | FP32 | BS = 1 | `188.25 ms` | `5.3 FPS` | Multi-threaded Host |
| **Intel Xeon CPU (Multi-thread)** | FP32 | BS = 32 | `4.77 ms` | `209.6 FPS` | Multi-threaded Host |

---

## System Architecture & Methodology

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
                                                • ResSE-Spectral Tower (4 stages + SE, 2.98M)
                                                • 512-d Spectral Embedding (f_f)
                                                • Auxiliary Supervision Head (λ = 0.3)
       │                                                │
       └───────────────────────┬────────────────────────┘
                               ▼
            [ Symmetric Gated Residual Fusion ]
            • Gating: g = Sigmoid(Linear(1024, 512)([f_s || f_f]))
            • Fused Feature: f_fused = [f_s * (1 - g) || f_f * g] ∈ R^1024
            • Video Embedding: e_t = f_s * (1 - g) + f_f * g ∈ R^512
                               │
       ┌───────────────────────┴────────────────────────┐
       ▼                                                ▼
[ Frame-Level Classifier Head ]              [ Bi-GRU Spatiotemporal Head ]
• Linear(1024, 256) -> Linear(256, 1)        • 2-Layer Bidirectional GRU (2.46M params)
• Temperature Scaled: z / T* (T* = 2.2018)   • Temporal Attention Context Pooling (α_t)
• Dual Bayesian Thresholds (τ_real, τ_fake)  • Inter-frame flickering / glitch detection
• 3 Forensic Certainty Zones                 • 60.9 FPS Real-Time Video Engine
```

### 1. Spatial Stream
* **Backbone**: ConvNeXt-Small pre-trained on ImageNet-1K, outputting a 768-dimensional feature representation normalized via `LayerNorm2d` and projected to a 512-dimensional spatial embedding $\mathbf{f}_s \in \mathbb{R}^{512}$.
* **Alignment**: Faces are dynamically localized using OpenCV's YuNet detector, expanded by $1.50\times$ to capture blending boundaries around the hairline and jaw, and aligned using 5-point facial landmark similarity transformations.

### 2. Frequency Stream: SRM, Bayar-Stamm, and 2D Real FFT
Noise residuals from 3 fixed $5\times5$ Steganographic Rich Model (SRM) kernels (9 channels) and 1 learnable Bayar-Stamm constrained convolution (1 channel) isolate high-frequency spatial discrepancies:

$$
\mathcal{F}_{\text{norm}} = \ln\left( \left| \mathcal{F}_{\text{ortho}}(I_{\text{SRM+Bayar}}) \right| + 1 \right)
$$

Phase angles are computed with sub-epsilon magnitude autograd masking to eliminate infinite gradient singularities:

$$
\theta = \frac{1}{\pi} \text{atan2}(I_{\text{imag}}, I_{\text{real}}) \quad \text{where} \quad |z| \ge 10^{-6}
$$

The resulting 20-channel representation (10 log-magnitude + 10 phase maps) is processed by the **ResSE-Spectral Tower**: a 4-stage residual network ($48 \to 96 \to 192 \to 384$ channels, 2.98M parameters) with Squeeze-and-Excitation (`SEBlock`) channel attention:

$$
\mathbf{z} = \text{AdaptiveAvgPool2d}(\mathbf{X}) \in \mathbb{R}^C
$$

$$
\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z})\right) \quad \text{where} \quad \mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}, \; \mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}
$$

$$
\widetilde{\mathbf{X}} = \mathbf{s} \odot \mathbf{X}
$$

To prevent the spatial stream from dominating gradient updates during training, an auxiliary linear head supervises the frequency representation directly:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}} + 0.3 \cdot \mathcal{L}_{\text{freq}}
$$

### 3. Symmetric Gated Residual Fusion
Both streams are symmetrically gated so neither stream starves the other of gradient flow:

$$
\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{f}_s \parallel \mathbf{f}_f] + \mathbf{b}_g\right) \in \mathbb{R}^{512}
$$

$$
\mathbf{f}_{\text{fused}} = \left[ \mathbf{f}_s \odot (1 - \mathbf{g}) \;\parallel\; \mathbf{f}_f \odot \mathbf{g} \right] \in \mathbb{R}^{1024}
$$

### 4. Spatiotemporal Sequence Modeling (Bi-GRU Head)
For video inference, frozen 512-dimensional sequence embeddings $\mathbf{e}_t = \mathbf{f}_s \odot (1 - \mathbf{g}) + \mathbf{f}_f \odot \mathbf{g}$ are processed by a 2-layer Bidirectional GRU (2.46M parameters) with temporal self-attention:

$$
\mathbf{h}_t = [\text{GRU}_{\text{fwd}}(\mathbf{e}_t) \parallel \text{GRU}_{\text{bwd}}(\mathbf{e}_t)] \in \mathbb{R}^{2H}
$$

$$
\alpha_t = \frac{\exp\left(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_t)\right)}{\sum_{j=1}^T \exp\left(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_j)\right)} \quad \text{where} \quad \sum_{t=1}^T \alpha_t = 1.0
$$

$$
\mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t \in \mathbb{R}^{2H}, \quad \hat{y}_{\text{video}} = \text{Classifier}(\mathbf{c})
$$

### 5. Dual-Threshold Bayesian Confidence Bands
Rather than enforcing a fixed 0.50 cutoff on ambiguous or compressed inputs, high-precision decision boundaries $(\tau_{\text{real}}, \tau_{\text{fake}})$ partition outputs into three certainty zones:
* **Confirmed Authentic**: $p \le \tau_{\text{real}}$ (Precision $\ge$ 98%)
* **Inconclusive / Perturbation Detected**: $\tau_{\text{real}} < p < \tau_{\text{fake}}$ (Flagged for manual inspection)
* **Confirmed Synthetic**: $p \ge \tau_{\text{fake}}$ (Precision $\ge$ 98%)

---

## Dataset Layout & Composition

To guarantee **100% zero identity leakage**, actor IDs (`id0_id16`) are partitioned using `networkx.Graph` connected components:

$$
\text{Actors}_{\text{train}} \cap \text{Actors}_{\text{val}} \cap \text{Actors}_{\text{test}} = \emptyset
$$

| Split | Total Samples | % of Dataset | Real Faces | Fake Faces | Fake:Real Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 79,097 | 69.2% | 18,373 | 60,724 | 3.31 : 1 |
| **Validation** | 20,544 | 18.0% | 2,112 | 18,432 | 8.73 : 1 |
| **Test** | 14,688 | 12.8% | 2,184 | 12,504 | 5.73 : 1 |
| **Total** | **114,329** | **100.0%** | **22,669** | **91,660** | **4.04 : 1** |

```text
dataset_root/
├── splits.json        # Manifest containing train/val/test relative image paths and labels
├── real/              # Extracted authentic facial crops (256x256 PNGs)
└── fake/              # Extracted synthesized facial crops (organized by generator family)
```

> [!NOTE]
> **Automatic Dataset Discovery**: If `--data_dir` is omitted, `DatasetResolver` automatically locates standard dataset roots across local directories (`./data/cropped`, `./data`) and Kaggle environments (`/kaggle/input/**`).

---

## Full Kaggle Reproduction Guide (2× Tesla T4)

### Phase 1: Distributed Training & Calibration

```bash
# 1. Run unit test suite (108 tests)
pytest tests/ -v

# 2. Train dual-stream backbone with ResSE tower & hardened augmentations (~25 min)
accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 \
    scripts/train_dual_stream_ddp.py \
    --epochs 5 --batch_size 16 --frequency_backbone resse --hardened \
    --save_path /kaggle/working/dual_stream_best.pth

# 3. Fit optimal temperature T* and derive Bayesian dual thresholds (~3 min)
python scripts/evaluate_test_set.py \
    --weights_path /kaggle/working/dual_stream_best.pth \
    --save_calibrated /kaggle/working/dual_stream_calibrated.pth

# 4. Train lightweight spatiotemporal Bi-GRU video consistency head (~4 min)
python scripts/train_temporal_head.py \
    --backbone_weights /kaggle/working/dual_stream_best.pth \
    --save_path /kaggle/working/temporal_head_best.pth \
    --epochs 5 --batch_size 8 --seq_len 8
```

### Phase 2: Diagnostic & Generalization Evaluation

```bash
# 5. Export test predictions JSON
python scripts/export_test_predictions.py \
    --checkpoint /kaggle/working/dual_stream_calibrated.pth \
    --output_json /kaggle/working/test_predictions.json

# 6. Evaluate subdomain breakdown across generators
python scripts/evaluate_subdomain_breakdown.py \
    --weights_path /kaggle/working/dual_stream_calibrated.pth

# 7. Run robustness degradation stress tests (JPEG, blur, noise, downscaling)
python scripts/evaluate_robustness.py \
    --checkpoint /kaggle/working/dual_stream_calibrated.pth \
    --output_json /kaggle/working/robustness_results.json

# 8. Run 5-fold Leave-One-Technology-Out (LOTO) cross-generator suite (~45 min)
rm -f /kaggle/working/loto_results.json
for fold in deepfakes face2face faceswap neuraltextures celeb; do
    accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 2 \
        scripts/train_loto_experiment.py \
        --holdout $fold --epochs 3 --batch_size 16 --frequency_backbone resse --hardened
done
```

### Phase 3: Export & Interpretability

```bash
# 9. Generate 300 DPI publication benchmark figures
python scripts/generate_benchmark_plots.py \
    --predictions /kaggle/working/test_predictions.json \
    --robustness /kaggle/working/robustness_results.json \
    --loto /kaggle/working/loto_results.json \
    --output_dir /kaggle/working/figures

# 10. Export trained backbone to ONNX
python scripts/export_onnx.py \
    --weights /kaggle/working/dual_stream_calibrated.pth \
    --output /kaggle/working/models/dual_stream.onnx --img_size 256

# 11. Benchmark inference latency & FPS
python scripts/benchmark_latency.py \
    --weights /kaggle/working/dual_stream_calibrated.pth --batch_size 32 --device cuda

# 12. Render 4-panel diagnostic Grad-CAM maps
python scripts/visualize_attention_maps.py \
    --checkpoint /kaggle/working/dual_stream_calibrated.pth \
    --output_dir /kaggle/working/figures/attention_maps --n_samples 6
```

---

## Core Repository Architecture

```text
deepfake-detection/
├── app.py                         # Streamlit web application & serving dashboard
├── config/default.yaml            # Hyperparameters and preprocessing resolution
├── figures/                       # Rendered publication-grade benchmark figures
├── scripts/                       # Thin executable CLI entry points
│   ├── train_dual_stream_ddp.py   # Multi-GPU DDP training (ResSE / legacy)
│   ├── evaluate_test_set.py       # Held-out evaluation, T* & dual thresholds
│   ├── train_temporal_head.py     # Spatiotemporal Bi-GRU video head training
│   ├── evaluate_robustness.py     # Degradation perturbation stress sweeps
│   ├── train_loto_experiment.py   # LOTO cross-generator training runner
│   ├── generate_benchmark_plots.py# 300 DPI publication figure rendering
│   └── export_onnx.py             # Production ONNX model exporter
├── src/                           # Modular core library
│   ├── dataset/                   # Graph partitioning, YuNet alignment, datasets
│   ├── evaluation/                # Test evaluators, safe metrics, ECE calculation
│   ├── models/                    # ConvNeXt, SRM/Bayar, FFT, ResSE, Gated Fusion, Bi-GRU
│   ├── services/                  # Video prediction engine & Streamlit components
│   ├── training/                  # Distributed trainer, focal loss, EMA, schedulers
│   └── utils/                     # Bayesian thresholds, Grad-CAM, checkpoint tools
└── tests/                         # Full PyTest test suite (108 passing tests)
```

---

## Dataset Compliance & Citations

Evaluated on **FaceForensics++** and **Celeb-DF v2**:
* **FaceForensics++**: Rössler et al., *IEEE/CVF ICCV 2019*. Access granted under the FaceForensics Non-Commercial Research Agreement.
* **Celeb-DF v2**: Li et al., *IEEE/CVF CVPR 2020*. Access granted under the Celeb-DF Release Agreement.

Model weights and code are provided solely for non-commercial academic research, forensic verification, and reproducible evaluation.

---

## Academic References

1. **ConvNeXt**: Liu, Z., et al. (2022). *A ConvNet for the 2020s*. IEEE/CVF CVPR.
2. **Steganographic Rich Model (SRM)**: Fridrich, J., & Kodovsky, J. (2012). *Rich models for steganalysis of digital images*. IEEE TIFS.
3. **Bayar-Stamm Constrained Convolution**: Bayar, B., & Stamm, M. C. (2016). *A deep learning approach to universal image manipulation detection*. IEEE IH&MMSec.
4. **Squeeze-and-Excitation Networks**: Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks*. IEEE/CVF CVPR.
5. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE/CVF ICCV.
6. **Temperature Scaling Calibration**: Guo, C., et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
7. **FaceForensics++**: Rössler, A., et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. IEEE/CVF ICCV.
8. **Celeb-DF**: Li, Y., et al. (2020). *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*. IEEE/CVF CVPR.
