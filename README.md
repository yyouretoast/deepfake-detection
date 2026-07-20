# Deepfake Detection Engine (PyTorch + ConvNeXt + 2D FFT)

> Dual-stream PyTorch 2.x Deepfake Detection Engine combining **ConvNeXt-Small** spatial features and **2D FFT Log-Magnitude Frequency Spectrum** embeddings. Features **Group-based Video-ID Partitioning**, **ONNX Runtime Acceleration**, **PyTorch Grad-CAM Explainability**, and an interactive **Streamlit UI**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Batched-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Architectural Features

- **Dual-Stream Neural Architecture**: Fuses 768-d spatial features from **ConvNeXt-Small** with 128-d frequency embeddings extracted via **2D Real FFT Log-Magnitude Spectrum** (896-d combined feature vector).
- **Group-Based Video-ID Partitioning**: Eliminates frame-level identity/background data leakage by strictly partitioning underlying video IDs across Train, Validation, and Test sets.
- **Batched GPU Face Extraction**: Uses `facenet-pytorch` MTCNN with dynamic **1.30x relative bounding box scale expansion** running in single-pass GPU batches.
- **Centralized Configuration Management**: Configured via `config/default.yaml` and parsed through `src/config.py` for global parameter control.
- **ONNX Runtime Acceleration**: Supports single-click export to ONNX Runtime format (`deepfake_convnext_v2.onnx`) for high-throughput CPU/GPU web inference.
- **PyTorch Grad-CAM Explainability**: Interactive heatmap overlay targeted specifically at `spatial_backbone` feature maps to reveal face manipulation artifacts on facial regions.
- **Automated `pytest` Integration Suite**: 6 integration test modules covering data leakage, Grad-CAM targeting, configuration loading, tensor shapes, cropper bounds safety, and ONNX parity.

---

## System Architecture

```
[ Input Video ] ──► [ GPU Batched MTCNN ] ──► [ Dynamic 1.30x Face Crops (224x224) ]
                                                            │
                            ┌───────────────────────────────┴──────────────────────────────┐
                            ▼                                                              ▼
              [ Spatial Stream (ConvNeXt) ]                              [ Frequency Stream (2D FFT) ]
              • 768-d Feature Embeddings                                 • 2D Centered Log-Magnitude Spectrum
                                                            │            • 128-d Frequency Embeddings
                                                            └───────────────┬──────────────┘
                                                                            ▼
                                                                  [ Feature Concatenation ]
                                                                  • 896-d Fused Representation
                                                                            │
                                                                            ▼
                                                                  [ Binary Classifier Head ]
                                                                  • Sigmoid Probability (Real / Fake)
```

---

## Benchmarking & Training Reproducibility

### 1. Empirical Latency Benchmarking

To measure PyTorch Native vs ONNX Runtime inference latency on your specific CPU/GPU hardware, execute the automated benchmarking script:

```bash
python benchmark.py
```

### 2. Kaggle GPU Training Execution (2x NVIDIA T4 GPUs)

Model weights (`deepfake_convnext_v2.pth`) and ONNX exports (`deepfake_convnext_v2.onnx`) are produced by running the self-contained PyTorch training notebook on Kaggle against FaceForensics++ (C23):

```bash
# Training notebook file
deepfake_detection_v2_pytorch.ipynb
```

The notebook pipeline executes:
- **Phase 1**: Frozen backbone training (8 epochs, `lr=3e-4`).
- **Phase 2**: End-to-end fine-tuning with AMP fp16 (12 epochs, `lr=3e-5`).
- **Ablation Study**: Spatial-Only vs Dual-Stream (Spatial + 2D FFT) accuracy comparison.
- **Generalization Benchmark**: Leave-One-Type-Out (LOTO) cross-manipulation evaluation.
- **ONNX Export**: Saves verified model to `deepfake_convnext_v2.onnx`.

---

## Known Limitations & Scope

1. **Model Checkpoint Generation**:
   - The repository provides the complete PyTorch architecture, preprocessing engine, test suite, and web application. Model weights (`deepfake_convnext_v2.pth`) must be generated by executing `deepfake_detection_v2_pytorch.ipynb` on GPU hardware.
2. **Dataset & Compression Generalization**:
   - Trained on FaceForensics++ (C23 compression). Performance on heavy social media re-compression (e.g. WhatsApp/TikTok re-encoding) or novel generative models (Sora, FLUX) requires fine-tuning on domain-specific datasets.
3. **Grad-CAM Interpretation**:
   - Grad-CAM heatmaps highlight feature activations within `spatial_backbone`. They serve as visual attention indicators, not definitive proof of manipulation.

---

## Quick Start

### 1. Local Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Automated Integration Test Suite

```bash
pytest tests/ -v
```

### 3. Run Streamlit Web Application

```bash
streamlit run app.py
```

---

## Docker Deployment

Build and run the production containerized web application:

```bash
# Build Docker image
docker build -t deepfake-detector .

# Run container on port 8501
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the web interface at `http://localhost:8501`.

---

## Directory Structure

```
deepfake-detection/
├── app.py                     # Streamlit web application with ONNX acceleration
├── benchmark.py               # Empirical inference latency benchmarking script
├── config/
│   └── default.yaml           # Global parameters, paths, and model hyperparams
├── src/
│   ├── config.py              # Centralized YAML configuration parser & defaults
│   ├── dataset/
│   │   ├── loader.py          # GroupKFold zero-leakage splitter & Albumentations
│   │   └── preprocess.py      # DynamicFaceCropper & GPU Batched MTCNN
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt + 2D FFT PyTorch architecture
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter & ONNXRuntime engine
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM heatmap generator & overlay
├── tests/                     # Automated integration test suite
│   ├── test_config.py         # YAML configuration parsing tests
│   ├── test_data_leakage.py   # GroupKFold zero-leakage assertions
│   ├── test_gradcam.py       # Grad-CAM spatial layer targeting & import tests
│   ├── test_model_forward.py  # Model forward pass tensor shape tests
│   ├── test_preprocess.py     # Dynamic cropper boundary safety tests
│   └── test_onnx.py           # PyTorch vs ONNX prediction parity tests
├── deepfake_detection_v2_pytorch.ipynb # Kaggle / Colab 2x T4 GPU Training Notebook
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production container deployment definition
└── README.md                  # Project documentation
```

---

## Author & License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
