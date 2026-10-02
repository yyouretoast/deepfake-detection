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
      value: 0.8994
    - name: Single-Frame ROC AUC
      type: roc_auc
      value: 0.8656
    - name: Expected Calibration Error (ECE)
      type: ece
      value: 0.0994
    - name: Equal Error Rate (EER)
      type: eer
      value: 0.2182
---

# Dual-Stream Deepfake Detector & Spatiotemporal Forensics

A PyTorch deepfake detection architecture coupling an ImageNet-modernized **ConvNeXt-Small** spatial backbone with a **20-channel ResSE-Spectral Tower** (Steganographic Rich Model + Bayar-Stamm 2D Real FFT decomposition), **High-Frequency SNR-Adaptive Gating**, affine Platt probability calibration, and a **Dual-Path (Attention + Global Max-Pooling) Bi-GRU** temporal video sequence head.

- **Preprint**: *"Dual-Stream Spatial-Frequency Feature Fusion with SNR-Adaptive Gating for Deepfake Detection: An Empirical Evaluation on Disjoint Partitions"* (arXiv:cs.CV / cs.CR)
- **GitHub Repository**: [yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)
- **Live Hugging Face Space**: [yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## Model Description

- **Developed by**: Yassin Yasser (Department of Artificial Intelligence, Sadat Academy for Management Sciences, Cairo, Egypt)
- **Model Type**: Dual-Stream Spatial + Frequency Hybrid Deepfake Detector
- **Spatial Backbone**: ConvNeXt-Small (512-d feature projection, 50.2M parameters)
- **Frequency Backbone**: ResSE-Spectral Tower (4 stages, 20-channel SRM/Bayar 2D Real FFT input, Squeeze-and-Excitation channel attention, 2.99M parameters)
- **Video Temporal Head**: 2-Layer Bidirectional GRU with First-Order Feature Velocity ($\Delta \mathbf{e}_t$) and Dual-Path Pooling (Attention + Extreme-Value Max-Pooling, 3.32M parameters)
- **Calibration**: Affine Platt Scaling ($a = 0.2783, b = 0.4089$, $T_{\text{eff}} = 3.5931$) with Bayesian 3-Zone Bands ($\tau_{\text{real}}=0.40, \tau_{\text{fake}}=0.60$) and Youden's optimal threshold ($\tau^* = 0.2600$)

---

## Hosted Checkpoints

| Checkpoint | File Size | Description | SHA-256 Checksum |
| :--- | :---: | :--- | :--- |
| **`dual_stream_calibrated.pth`** | **204.7 MB** | Primary spatial-spectral hybrid backbone with ResSE Tower and post-hoc Platt calibration ($a=0.2783, b=0.4089$, $T_{\text{eff}}=3.5931$, $\tau^*=0.2600$). | `e21846894bba34e5fb12e7e01cc045a081d3a755d24be079ad7974b154a5c3ba` |
| **`dual_stream_best.pth`** | **204.7 MB** | Uncalibrated raw checkpoint for primary dual-stream architecture (He-normal + Cosine Annealing, 5 epochs). | `efacdbd4bfd69ac1f49f61748b9797108f071fd87b7754f42edf265fbf2b5955` |
| **`spatial_convnext_best.pth`** | **193.3 MB** | Standalone ConvNeXt-Small baseline trained on identical split without spectral stream. | `8c1d978f114ff2ffc1ceb448f110dd53ed5004aaf400e675c8ec9d8dd1a67494` |
| **`temporal_head_best.pth`** | **12.7 MB** | 2-Layer Bi-GRU video sequence head evaluating multi-frame velocity deltas and attention weights. | `b81eed741d93218a5c88b93d660f97d3266ffd3e7501c70470fc611772951abf` |

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
optimal_threshold = float(checkpoint.get("optimal_threshold", 0.2600))
temperature = float(checkpoint.get("effective_temperature", 3.5931))
tau_real = float(checkpoint.get("tau_real", 0.40))
tau_fake = float(checkpoint.get("tau_fake", 0.60))

print(f"Loaded model calibrated with T_eff={temperature:.3f}, optimal threshold τ*={optimal_threshold:.2f}")
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

All metrics are benchmarked on our strictly disjoint evaluation protocol enforcing pair-disjoint separation on FaceForensics++, actor-disjoint separation on Celeb-DF v2, and holding Google DeepFakeDetection (DFD) entirely held-out for zero-shot testing (37,104 train, 35,016 val, 26,981 held-out test crops across 2,248 video sequences with zero identity leakage).

### Single-Frame vs Spatiotemporal Evaluation (Held-Out Test Set)

| Architecture / Method | Modality | ROC AUC $\uparrow$ | PR AUC $\uparrow$ | Fake F1 $\uparrow$ | Balanced Acc. $\uparrow$ | Fake Prec. $\uparrow$ | Fake Rec. $\uparrow$ | EER (\%) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Spatial Baseline (ConvNeXt-Small) | Frame | 0.8370 | 0.9226 | 0.7810 | 75.17% | 90.50% | 68.68% | 24.43% |
| **Dual-Stream Gated Fusion (Ours)** | **Frame** | **0.8656** | **0.9374** | **0.8292** | **78.04%** | **90.51%** | **76.50%** | **21.82%** |
| Naive Uniform Frame Averaging | Video | 0.8962 | 0.9559 | 0.8515 | 81.98% | 93.28% | 78.31% | 19.43% |
| Temporal Max-Pooling | Video | 0.8544 | 0.9167 | 0.8785 | 77.48% | 87.00% | 88.72% | 21.48% |
| **Spatiotemporal Bi-GRU (Ours)** | **Video** | **0.8994** | **0.9571** | **0.8517** | **82.28%** | **93.62%** | **78.13%** | **18.54%** |

### Fine-Grained Subdomain Performance

| Manipulation Subdomain | Test Fakes ($N$) | ROC AUC | Fake F1 | Fake Precision | Fake Recall | Test Class Skew |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FF++ Deepfakes** | 192 | 0.9592 | 0.5869 | 0.4282 | 0.9323 | 1 : 8.75 |
| **FF++ Face2Face** | 576 | 0.9463 | 0.7845 | 0.6876 | 0.9132 | 1 : 2.92 |
| **FF++ FaceSwap** | 240 | 0.9658 | 0.6353 | 0.4827 | 0.9292 | 1 : 7.00 |
| **FF++ NeuralTextures** | 348 | 0.9509 | 0.6941 | 0.5662 | 0.8966 | 1 : 4.83 |
| **Celeb-DF v2** | 5,700 | 0.9128 | 0.8754 | 0.9162 | 0.8381 | 2.45 : 1 |
| **Google DFD (Zero-Shot)** | 11,992 | 0.8651 | 0.8201 | 0.9727 | 0.7090 | 7.14 : 1 |
| **Complete Test Cohort** | **19,372** | **0.8656** | **0.8292** | **0.9051** | **0.7650** | **2.55 : 1** |

### Canonical 4-Fold Leave-One-Manipulation-Out (LOMO) Benchmark

In strict accordance with the FaceForensics++ benchmark protocol, models are evaluated on canonical 1:1 balanced cohorts ($N_{\text{real}} = N_{\text{fake}}$) where all synthetic media from the targeted manipulation architecture is quarantined from training and validation splits:

| Fold | Held-Out Unseen Target | Holdout Fakes | Fitted $T^*$ | Zero-Shot ROC AUC $\uparrow$ | Balanced Acc. $\uparrow$ | Zero-Shot Fake F1 $\uparrow$ | Zero-Shot Precision $\uparrow$ | EER (\%) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | FF++ Deepfakes (Pairs 0–99) | 192 | 2.3923 | **0.9413** | 88.54% | **0.8785** | 93.53% | 12.24% |
| **Fold 2** | FF++ Face2Face (Pairs 100–399) | 576 | 2.3647 | **0.9434** | 88.45% | **0.8783** | 92.84% | 12.33% |
| **Fold 3** | FF++ FaceSwap (Pairs 400–599) | 240 | 2.1169 | **0.9558** | 87.29% | **0.8732** | 87.14% | 12.29% |
| **Fold 4** | FF++ NeuralTextures (Pairs 600–799) | 348 | 1.9302 | **0.9651** | 89.08% | **0.8848** | 93.59% | 10.06% |
| **Macro** | **Macro-Average Across All 4 Folds** | **1,356** | — | **0.9514** | **88.34%** | **0.8787** | **91.77%** | **11.73%** |

*Note: For cross-dataset acquisition shifts, zero-shot transfer on Celeb-DF v2 without target adaptation achieves 0.5961 ROC AUC, matching published literature baselines (MesoNet 0.548, Capsule 0.575, Xception 0.653, F3-Net 0.652). Fully calibrated evaluation achieves 0.9128 ROC AUC on Celeb-DF v2 and 0.8651 zero-shot ROC AUC on Google DeepFakeDetection.*

---

## Data Use Agreements & Ethical Compliance

This repository and model card distribute **only** algorithmic code, neural network weight tensors, and deterministic graph partition metadata manifests (`splits/`). In strict compliance with upstream Data Use Agreements:
- **FaceForensics++ (TUM)**: Subject to the FaceForensics Terms of Use. No original video sequences or facial crops are hosted or redistributed.
- **Celeb-DF v2 (SUNY Buffalo)**: Subject to the Celeb-DF Research Agreement. Images of public figures were utilized exclusively for non-commercial academic benchmarking.
- **Google DeepFakeDetection (DFD)**: Utilized strictly as an external zero-shot test set. No source frames are redistributed.

---

## Citation

```bibtex
@article{yasser2026dualstream,
  author    = {Yassin Yasser},
  title     = {Dual-Stream Spatial-Frequency Feature Fusion with SNR-Adaptive Gating for Deepfake Detection: An Empirical Evaluation on Disjoint Partitions},
  journal   = {arXiv preprint arXiv:cs.CV/cs.CR},
  year      = {2026},
  url       = {https://github.com/yyouretoast/deepfake-detection}
}
```
