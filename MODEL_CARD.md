---
language:
- en
license: mit
library_name: pytorch
tags:
- deepfake-detection
- computer-vision
- forensic-analysis
- convnext
- fourier-transform
- steganography
- srm
- sequence-modeling
metrics:
- roc_auc
- accuracy
- ece
pipeline_tag: image-classification
model-index:
- name: Dual-Stream Deepfake Detector (ResSE + Bi-GRU)
  results:
  - task:
      type: image-classification
      name: Deepfake Detection
    metrics:
    - name: Video ROC AUC
      type: roc_auc
      value: 0.8719
    - name: Single-Frame ROC AUC
      type: roc_auc
      value: 0.8248
    - name: Expected Calibration Error (ECE)
      type: ece
      value: 0.0050
    - name: Equal Error Rate (EER)
      type: eer
      value: 0.1898
---

# Dual-Stream Deepfake Detector & Spatiotemporal Forensics

A PyTorch deepfake detection architecture fusing a **ConvNeXt-Small** spatial backbone with a **4-stage ResSE-Spectral Tower** (Steganographic Rich Model + Bayar-Stamm 2D Real FFT decomposition), **High-Frequency SNR-Adaptive Gating**, Platt probability calibration, and a **Dual-Path (Attention + Global Max-Pooling) Bi-GRU** temporal video sequence head.

- **GitHub Repository**: [yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)
- **Live Hugging Face Space**: [yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## Model Description

- **Developed by**: Yassin
- **Model Type**: Dual-Stream Spatial + Frequency Hybrid Deepfake Detector
- **Spatial Backbone**: ConvNeXt-Small (512-d feature projection)
- **Frequency Backbone**: ResSE-Spectral Tower (4 stages, 20-channel SRM/Bayar 2D Real FFT input, Squeeze-and-Excitation channel attention)
- **Video Temporal Head**: 2-Layer Bidirectional GRU with Dual-Path Pooling (Attention + Max-Pooling)
- **Calibration**: Temperature Scaling ($T^* = 4.2880$) and Bayesian 3-Zone Bands ($\tau_{\text{real}}=0.40, \tau_{\text{fake}}=0.60$)

---

## Hosted Checkpoints

| Checkpoint | File Size | Description | SHA-256 Checksum |
| :--- | :---: | :--- | :--- |
| **`dual_stream_calibrated.pth`** | **214.7 MB** | Primary spatial-spectral hybrid backbone with ResSE Tower and post-hoc calibration ($T^* = 4.288$, $\tau^* = 0.4200$). | `cacdd1f63a089f157d7cec384df632a37ee107830bd52a71d7faabeb50fd5237` |
| **`temporal_head_best.pth`** | **13.3 MB** | 2-Layer Bi-GRU video sequence head evaluating multi-frame velocity deltas and attention weights. | `5976689a9132bdad51bb09cedc48c8ebbd425f353849ad15aa67073058d0700e` |

---

## How to Use

### 1. Automated Hub Download & Inference

```python
import torch
from huggingface_hub import hf_hub_download

# Download calibrated weights
ckpt_path = hf_hub_download(
    repo_id="yyouretoast/deepfake-detector",
    filename="dual_stream_calibrated.pth"
)

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
optimal_threshold = float(checkpoint.get("optimal_threshold", 0.4200))
temperature = float(checkpoint.get("temperature", 4.2880))
tau_real = float(checkpoint.get("tau_real", 0.40))
tau_fake = float(checkpoint.get("tau_fake", 0.60))

print(f"Loaded model calibrated with T*={temperature:.3f}, optimal threshold τ*={optimal_threshold:.2f}")
```

### 2. Forensic Single Image Analysis

Using the official repository engine:

```python
from src.services.video_engine import process_single_image

# Runs YuNet 5-point face alignment, dual-stream inference, and telemetry
report = process_single_image("suspect_face.jpg")

print("Verdict:", report["three_zone"]["verdict"])
print("Calibrated Probability:", report["prob"])
print("Spectral vs Spatial Gate:", f"{report['spectral_gate']*100:.1f}% / {report['spatial_gate']*100:.1f}%")
print("SNR Attenuation (γ):", report["snr_attenuator"])
```

---

## Empirical Benchmark Performance

Evaluated across **13,444 facial crops** (756 authentic real faces, 12,688 deepfakes across 5 generator families) and **1,120 video sequences**:

| Metric | Single-Frame Spatial Model | Video Spatiotemporal Bi-GRU | Delta / Impact |
| :--- | :---: | :---: | :--- |
| **ROC AUC** | **`0.8248`** | **`0.8719`** | **+4.71%** discriminative improvement |
| **PR AUC** | **`0.9860`** *(1:1 Bal: `0.8298`)* | **`0.9904`** | Precision-recall area under curve (16.78:1 skew) |
| **Equal Error Rate (EER)** | `24.10%` | **`18.98%`** | **-5.12%** biometric verification error drop |
| **Fake Precision** | **`98.00%`** | **`98.60%`** | **+0.60%** false alarm suppression |
| **Fake Recall** | **`77.04%`** | **`79.75%`** | **+2.71%** detection coverage |
| **Overall Accuracy** | **`76.85%`** | **`79.82%`** | **+2.97%** classification rate |
| **Expected Calibration Error (ECE)** | **`0.0050`** | -- | Platt-calibrated ($98.9\%$ error drop from raw $0.4525$) |

### In-Distribution Per-Generator AUC:
- **FF++ Face2Face**: **`0.9975`**
- **FF++ NeuralTextures**: **`0.9749`**
- **FF++ Deepfakes**: **`0.9625`**
- **FF++ FaceSwap**: **`0.9091`**
- **Celeb-DF v2**: **`0.8166`**

---

## Citation

```bibtex
@misc{deepfake_forensics_2026,
  author = {Yassin},
  title = {Dual-Stream Deepfake Forensics Engine: Spatial ConvNeXt and ResSE-Spectral Gated Fusion with Spatiotemporal Sequence Modeling},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/yyouretoast/deepfake-detector}}
}
```
