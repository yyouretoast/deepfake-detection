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

A PyTorch 2.x dual-stream deepfake detection pipeline combining ConvNeXt-Base spatial representations with centered 2D Real FFT log-magnitude frequency spectrum embeddings.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![pytest](https://img.shields.io/badge/pytest-51%2F51%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live Demo**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Tests (< 5s Fast Feedback Loop)

```bash
# Fast Unit Test Suite (< 5 seconds)
pytest -m fast -v

# Full Integration Suite (51/51 Passing)
pytest tests/ -v
```

### 3. Launch Web Application

```bash
streamlit run app.py
```

### 4. Run Training & Benchmarks

```bash
# CLI Fine-Tuning
python train.py --epochs_phase1 3 --epochs_phase2 15

# Fast Benchmark (500 samples)
python benchmark.py --mode fast
```

---

## System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ 256x256 Face Crops ]
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         [ Spatial Stream (ConvNeXt-Base) ]                       [ Frequency Stream (2D Real FFT) ]
         • 256x256 RGB Image Input                                • 2D Real FFT (fftshift centered)
         • 1024-d Feature Embeddings                               • 128-d Frequency Spectrum Embeddings
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       [ 4-Head Cross-Attention Fusion ]
                                       • Spatial Query (128-d) ◄► Freq Key/Value (128-d)
                                       • Learnable Residual Parameter γ + 1152-d Fusion
                                                      │
                                                      ▼
                                           [ Binary Classifier Head ]
                                           • LayerNorm(256) + Dropout(0.3)
                                           • Log-Temperature Calibrated Probabilities σ(z / T*)
```

### Formulations

**2D Real FFT Spectrum Extraction**:

$$
\mathcal{F}_{\text{norm}} = \frac{\ln\left( |\mathcal{F}_{\text{ortho}}(I_{\text{gray}})| + 10^{-5} \right) - \mu}{\sigma + 10^{-6}}
$$

**Cross-Attention Residual Fusion**:

$$
\mathbf{f}_{\text{enhanced}} = \mathbf{f}_{\text{freq}} + \gamma \times \text{Attention}(Q_{\text{spatial}}, K_{\text{freq}}, V_{\text{freq}})
$$

Where $\gamma = \text{nn.Parameter}(\text{torch.tensor}(0.1))$ is a learnable scalar controlling frequency feature injection.

---

## Core Engineering Features

- **Zero Identity Data Leakage**: Graph connected-component partitioning (`networkx.Graph`) segregates actor IDs across train, val, and test splits.
- **10-Group LLRD Optimization**: Splits 5 LLRD tiers into 10 parameter groups (decayed 2D/4D weights `1e-2` vs non-decayed 1D biases/LayerNorms/$\gamma$ `0.0`).
- **Log-Temperature Calibration**: Calibrates raw output logits using LBFGS-optimized log-temperature $\alpha = \ln(T)$, ensuring $T = \exp(\alpha) \ge 0.05$.
- **Logit-Space TTA**: Evaluates original and horizontally flipped inputs ($\sigma((z + z_{\text{flip}})/2)$).
- **Dual-Stream Grad-CAM**: Generates signed target class saliency maps for spatial (`target_stream="spatial"`) or 2D FFT frequency (`target_stream="frequency"`) channels with thread-safe `GRADCAM_LOCK` guards.

---

## Benchmark Performance

Evaluated under **Stratified GroupKFold by Video-ID** (zero identity leakage):

| Model / Architecture | Input Resolution | AUC | Accuracy | Reference |
| :--- | :---: | :---: | :---: | :--- |
| Standard CNNs (ResNet-50 / VGG16) | $224 \times 224$ | `0.810 - 0.850` | `75.0% - 81.0%` | Baseline spatial frame classifiers |
| Spatial-Only ConvNeXt-Base | $256 \times 256$ | `0.932` | `87.5%` | Spatial stream only (no FFT branch) |
| Xception Baseline | $299 \times 299$ | `0.950` | `89.3%` | *Rossler et al., ICCV 2019* (FF++ Paper) |
| **Dual-Stream ConvNeXt + 2D FFT** | $256 \times 256$ | **`0.9377`** | `87.4%` | **Active Baseline** |

---

## Repository Structure

```
deepfake-detection/
├── app.py                     # Streamlit web UI with T* calibration & Grad-CAM
├── benchmark.py               # Benchmark runner (--mode fast, --mode paper, 5-fold LOTO)
├── train.py                   # Two-phase fine-tuning CLI entrypoint
├── pytest.ini                 # PyTest configuration with fast/slow test markers
├── config/
│   └── default.yaml           # Configuration parameters
├── src/
│   ├── config.py              # Configuration parser
│   ├── dataset/
│   │   ├── loader.py          # Identity-safe graph splitter & loader
│   │   └── preprocess.py      # DynamicFaceCropper & 5-point similarity alignment
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt + 2D FFT architecture
│   │   ├── temporal.py        # 5D Temporal Sequence Transformer Encoder
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter
│   ├── training/
│   │   ├── trainer.py         # TwoPhaseTrainer & 10-group LLRD setup
│   │   └── evaluator.py       # Crash-proof EER, adaptive ECE & temperature scaling
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM heatmap generator & overlay
├── tests/                     # 51/51 passing unit & integration tests
├── scripts/                   # Ablation, LOTO, cross-dataset & robustness scripts
├── deepfake_detection_v2_pytorch.ipynb # GPU Training Notebook
├── Dockerfile                 # Production Docker container
└── README.md
```

---

## License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
