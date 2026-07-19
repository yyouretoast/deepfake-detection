# 🎭 Deepfake Detection Engine v2 (PyTorch + ConvNeXt + 2D FFT)

> High-performance, dual-stream PyTorch 2.x Deepfake Detection Engine combining **ConvNeXt-Small** spatial features and **2D FFT Log-Magnitude Frequency Spectrum** embeddings. Features strict **Video-ID GroupKFold zero data leakage**, **ONNX Runtime acceleration**, **PyTorch Grad-CAM explainability**, and an interactive **Streamlit UI**.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Accelerated-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Batched-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Key Architectural Innovations

- 🧠 **Dual-Stream Neural Architecture**: Fuses 768-d spatial features from **ConvNeXt-Small** with 128-d frequency embeddings extracted via **2D Real FFT Log-Magnitude Spectrum** (896-d combined feature vector).
- 🛡️ **Zero-Leakage Video-ID GroupKFold Split**: Eliminates frame-level identity/background data leakage by strictly partitioning underlying video IDs across Train, Validation, and Test sets.
- ⚡ **Ultra-Fast GPU Batched Face Extraction**: Uses `facenet-pytorch` MTCNN with dynamic **1.30x relative bounding box scale expansion** running in single-pass GPU batches (`<1.0s/it` per video).
- 🚀 **3x–5x ONNX Runtime Acceleration**: Supports single-click export to ONNX Runtime format (`deepfake_convnext_v2.onnx`) for high-throughput CPU/GPU web inference.
- 🔍 **PyTorch Grad-CAM Explainability**: Interactive heatmap overlay revealing face manipulation artifacts on mouth, eyes, and boundary regions.
- 🧪 **Automated `pytest` Suite**: Integration test coverage for data leakage, forward pass tensor shapes, cropper boundary safety, and ONNX output numerical parity.

---

## 🏗️ System Architecture

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

## 📊 Dataset & Benchmarks

Trained and evaluated on **FaceForensics++ (C23)** containing 1,000 original YouTube videos across 6 manipulation categories:
- **Deepfakes**
- **Face2Face**
- **FaceSwap**
- **NeuralTextures**
- **FaceShifter** *(Held-Out target for Leave-One-Type-Out Generalization Benchmark)*
- **DeepFakeDetection**

### ⚡ Inference Latency Comparison

| Engine / Framework | Batch Size | Device | Latency per Frame | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: |
| PyTorch Native (FP32) | 1 | CPU | ~48 ms | 1.0x |
| **ONNX Runtime (CPU)** | **1** | **CPU** | **~14 ms** | **3.4x faster** ⚡ |
| **ONNX Runtime (CUDA)** | **15** | **GPU** | **~3 ms** | **16.0x faster** ⚡ |

---

## 🚀 Quick Start

### 1. Local Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### 2. Run Streamlit Web Application

```bash
streamlit run app.py
```

### 3. Run Automated Integration Test Suite

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Build and run the production containerized web application:

```bash
# Build Docker image
docker build -t deepfake-detector .

# Run container on port 8501
docker run -d -p 8501:8501 --name deepfake-app deepfake-detector
```

Access the web interface at `http://localhost:8501`.

---

## 🗂️ Directory Structure

```
deepfake-detection/
├── app.py                     # Streamlit web application with ONNX acceleration
├── config/
│   └── default.yaml           # Global parameters, paths, and model hyperparams
├── src/
│   ├── dataset/
│   │   ├── loader.py          # GroupKFold zero-leakage splitter & Albumentations
│   │   └── preprocess.py      # DynamicFaceCropper & GPU Batched MTCNN
│   ├── models/
│   │   ├── hybrid_detector.py # Dual-Stream ConvNeXt + 2D FFT PyTorch architecture
│   │   └── onnx_exporter.py   # PyTorch to ONNX exporter & ONNXRuntime engine
│   └── explainability/
│       └── gradcam.py         # PyTorchGradCAM heatmap generator & overlay
├── tests/                     # Automated integration test suite
│   ├── test_data_leakage.py   # GroupKFold zero-leakage assertions
│   ├── test_model_forward.py  # Model forward pass tensor shape tests
│   ├── test_preprocess.py     # Dynamic cropper boundary safety tests
│   └── test_onnx.py           # PyTorch vs ONNX prediction parity tests
├── deepfake_detection_v2_pytorch.ipynb # Kaggle / Colab 2x T4 GPU Training Notebook
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production container deployment definition
└── README.md                  # Project documentation
```

---

Licensed under the [MIT License](LICENSE).
