# 🎭 Dual-Stream Deepfake Detection Engine: Master Technical Guide

---

## 📖 Complete Curriculum Index

This repository contains an exhaustive, zero-assumptions technical curriculum covering every physical, mathematical, architectural, and operational aspect of the Dual-Stream Deepfake Detection Engine.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CURRICULUM DIRECTORY                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 📄 SECTION 1: Foundations & First Principles from Zero (Image Physics to Dual-Domain)  │
│ 📄 SECTION 2: The Neural Network Architecture (ConvNeXt + SRM + Bayar + 2D FFT + Gate) │
│ 📄 SECTION 3: Face Preprocessing, Alignment & Zero-Leakage Graph Partitioning          │
│ 📄 SECTION 4: Training Engine & Optimization (DDP, SyncBatchNorm, Loss Masking, EMA)   │
│ 📄 SECTION 5: Calibration, Temporal Video Aggregation & Inference Engine (L-BFGS-B)    │
│ 📄 SECTION 6: Forensic Explainability & Interpretability Engine (Grad-CAM, SRM, FFT)   │
│ 📄 SECTION 7: Quantitative Benchmarks, LOTO Generalization, Robustness & Latency       │
│ 📄 SECTION 8: Interactive Web Application & Production Deployment (Streamlit)          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📑 Direct Section Links

1. **[Section 1: Foundations & First Principles](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_1_FOUNDATIONS.md)**
   - Hardware image capture, CMOS sensors, PRNU noise physics.
   - The 5 generative manipulation technologies (*Deepfakes, Face2Face, FaceSwap, NeuralTextures, Celeb-DF v2*).
   - Forensic flaws (PRNU disruption, checkerboard grids, boundary resolution mismatches).
   - 2D Discrete Fourier Transform mathematics, Euler's formula, and steganographic principles.

2. **[Section 2: Neural Network Architecture](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_2_MODEL_ARCHITECTURE.md)**
   - Spatial stream with `ConvNeXt-Small` and pre-classifier `LayerNorm2d(768)`.
   - Frequency stream with 9-channel `SRMConv2d`, 1-channel `BayarConv2d`, and 20-channel `RealFFT2DModule`.
   - Symmetric Gated Residual Fusion ($g \in [0, 1]^{512}$, $\mathbf{f}_{\text{fused}} \in \mathbb{R}^{1024}$).
   - 5D sequence chunked processing (`forward_sequence`).

3. **[Section 3: Face Preprocessing & Alignment](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_3_DATA_PREPROCESSING_ALIGNMENT.md)**
   - Multi-engine face detection cascade (OpenCV YuNet $\to$ MTCNN $\to$ Haar Cascade $\to$ Center Crop).
   - $1.50\times$ bounding box scaling and 5-point LMEDS affine landmark alignment.
   - 2D Cosine window edge tapering eliminating FFT boundary cross-spikes.
   - Zero-leakage graph connected-component partitioning (`networkx.Graph`).

4. **[Section 4: Training Engine & Optimization](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_4_TRAINING_ENGINE_OPTIMIZATION.md)**
   - Multi-GPU Distributed Data Parallel (DDP) with Hugging Face `Accelerate` and `SyncBatchNorm`.
   - Per-sample loss masking (`valid_flags`) and dynamic `pos_weight` calculation.
   - 4-way differential parameter grouping (`lr_backbone = 1e-4`, `lr_head = 1e-3`).
   - Exponential Moving Average (EMA) shadow weight updates ($\beta = 0.999$).

5. **[Section 5: Calibration & Temporal Aggregation](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_5_CALIBRATION_TEMPORAL_AGGREGATION.md)**
   - SciPy L-BFGS-B log-temperature scaling ($T^* = 1.4788$, ECE reduction: $0.0122 \to 0.0093$).
   - Operational confidence score normalization ($[50.0\%, 100.0\%]$ anchored at threshold $\theta = 0.01$).
   - 4 temporal aggregation algorithms: Softmax-Weighted ($\tau = 0.10$), Top-$k$, EMA, and Mean.
   - Sequential video decoding with Test-Time Augmentation (TTA).

6. **[Section 6: Forensic Explainability & Interpretability](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_6_EXPLAINABILITY_INTERPRETABILITY.md)**
   - ConvNeXt Stage 4 Grad-CAM hooks and gradient-weighted activation mapping.
   - 9-channel SRM noise residual maps rendered with OpenCV Viridis.
   - Centered 2D Real FFT log-magnitude spectrums rendered with OpenCV Magma.
   - Publication-ready 4-panel diagnostic visualizer (`visualize_attention_maps.py`).

7. **[Section 7: Benchmarks, Robustness & Latency](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_7_BENCHMARKS_ROBUSTNESS_LATENCY.md)**
   - Held-out test set performance: **ROC AUC `0.9988`**, **F1 `0.9830`**, **Recall `99.79%`**.
   - Leave-One-Target-Out (LOTO) cross-generator zero-shot generalization ($0.9662 - 0.9783$ AUC).
   - Degradation stress tests (JPEG, Blur, Noise, Downscaling) and failure mode analysis.
   - Hardware latency benchmarks on NVIDIA Tesla T4 GPU ($18.62\text{ ms}$) and Intel Xeon CPU ($4.77\text{ ms}$).

8. **[Section 8: Web Application & Production Deployment](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/SECTION_8_WEB_APPLICATION_DEPLOYMENT.md)**
   - Full-stack Streamlit web app with singleton model caching and CUDA warm-up.
   - Dark glassmorphism UI/UX styling and mobile responsive breakpoints.
   - Interactive temporal anomaly sequence timeline plotting in Matplotlib.
   - Production Docker containerization and Hugging Face Spaces cloud serving.

---

## 🏛️ Master Architecture Diagram

```
[ Input Video / Image ] ──► [ OpenCV YuNet Alignment ] ──► [ 512x512 Face Crops (1.50x Scale) ]
                                                                │
                        ┌───────────────────────────────────────┴───────────────────────────────────────┐
                        ▼                                                                               ▼
          [ Spatial Stream (ConvNeXt-Small) ]                             [ Frequency Stream (SRM + Bayar + 2D FFT) ]
          • 256x256 RGB Image Input                                       • 3 Fixed SRM Filters (9-ch) + 1 Bayar Conv (1-ch)
          • LayerNorm2d Spatial Normalization                             • 20-Channel 2D FFT (10 Mag + 10 Phase)
          • 512-d Feature Embedding f_s                                   • 512-d Frequency Embedding f_f
                        │                                                                               │
                        └───────────────────────────────┬───────────────────────────────────────────────┘
                                                        ▼
                                         [ Symmetric Gated Residual Fusion ]
                                         • Gating Vector g = Sigmoid(Linear(f_s || f_f)) in R^512
                                         • Fused Feature f_fused = [f_s * (1 - g) || f_f * g] in R^1024
                                                        │
                                                        ▼
                                             [ Binary Classifier Head ]
                                             • Linear(1024, 256) + Dropout(0.3) + Linear(256, 1)
                                             • Raw Logit Output z
                                                        │
                        ┌───────────────────────────────┴───────────────────────────────┐
                        ▼                                                               ▼
      [ SciPy L-BFGS-B Temperature Calibration ]                  [ 4-Panel Interpretability Engine ]
      • Calibrated Logit: z / T* (T* = 1.4788)                   • (a) RGB Input Face Crop
      • Probability: p = Sigmoid(z / T*)                          • (b) SRM Noise Residual Map (Viridis)
      • Expected Calibration Error: 0.0122 -> 0.0093              • (c) 2D FFT Magnitude Spectrum (Magma)
                                                                  • (d) ConvNeXt Grad-CAM Attention Heatmap
```

---

## 📐 Master Mathematical Formula Sheet

### 1. 2D Discrete Fourier Transform (Spectral Decomposition)
$$F(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} f(x, y) \cdot e^{-j 2\pi \left( \frac{ux}{H} + \frac{vy}{W} \right)}$$
$$\mathcal{M}(u, v) = \ln\Big( \big| \text{fftshift}(F(u, v)) \big| + 1 \Big), \quad \Phi(u, v) = \frac{\text{angle}(\text{fftshift}(F(u, v)))}{\pi}$$

### 2. Bayar-Stamm Adaptive Constrained Convolution
$$\mathbf{W}_{\text{constrained}}(i, j) = \begin{cases} -1.0, & \text{if } (i, j) = (0, 0) \\ \frac{\mathbf{W}(i, j)}{\sum_{(u,v) \neq (0,0)} \mathbf{W}(u, v)}, & \text{if } (i, j) \neq (0, 0) \end{cases}$$

### 3. Symmetric Gated Residual Fusion
$$g = \sigma\Big( \mathbf{W}_g [\mathbf{f}_{\text{spatial}} \;\|\; \mathbf{f}_{\text{freq}}] + \mathbf{b}_g \Big) \in [0, 1]^{512}$$
$$\mathbf{f}_{\text{fused}} = \Big[ \mathbf{f}_{\text{spatial}} \odot (1 - g) \;\|\; \mathbf{f}_{\text{freq}} \odot g \Big] \in \mathbb{R}^{1024}$$

### 4. Per-Sample Masked Binary Cross-Entropy Loss
$$\mathcal{L} = \frac{\sum_{i=1}^B \mathcal{L}_{\text{BCE}}(z_i, y_i) \cdot v_i}{\max\left(1, \sum_{i=1}^B v_i\right)}, \quad v_i \in \{0.0, 1.0\}$$

### 5. Temperature Scaling Calibration ($T^* = 1.4788$)
$$p_{\text{calibrated}} = \sigma\left( \frac{z}{T^*} \right) = \frac{1}{1 + e^{-z / T^*}}$$

### 6. Softmax-Weighted Video Temporal Aggregation ($\tau = 0.10$)
$$S_{\text{video}} = \sum_{k=1}^K w_k \cdot p_k, \quad \text{where } w_k = \frac{\exp\left(\frac{p_k}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{p_j}{\tau}\right)}$$

---

## ⚡ Quickstart Commands

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt

# 2. Run the full automated test suite (66/66 passing tests)
pytest tests/ -v

# 3. Launch the interactive Streamlit Web Application
streamlit run app.py

# 4. Run Multi-GPU DDP Training
accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_dual_stream_ddp.py

# 5. Generate Publication-Ready Benchmark Figures
python scripts/generate_benchmark_plots.py
```

---

*Authored by Yassin Yasser. Licensed under the MIT License.*
