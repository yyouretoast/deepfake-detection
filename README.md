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

# 🎭 Dual-Stream Deepfake Detection Engine

A PyTorch 2.x dual-stream deepfake detection pipeline combining ConvNeXt-Small spatial representations with Steganographic Rich Model (SRM) + Bayar-Stamm 2D Real FFT frequency spectrum embeddings, SciPy L-BFGS-B temperature calibration, and thread-safe OpenCV YuNet face detection.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-DDP-005CED?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
[![pytest](https://img.shields.io/badge/pytest-54%2F54%20Passing-2EA44F?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Notebook](https://img.shields.io/badge/Notebook-Master%20Pipeline-blue?logo=jupyter)](notebooks/master_pipeline.ipynb)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live Interactive Space**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)  
**GitHub Repository**: [https://github.com/yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)

---

## System Architecture

```
[ Input Video Stream ] ──► [ Thread-Local YuNet ] ──► [ 512x512 Face Crops (1.50x Scale Expansion) ]
                                                              │
                       ┌──────────────────────────────────────┴──────────────────────────────────────┐
                       ▼                                                                             ▼
         [ Spatial Stream (ConvNeXt-Small) ]                             [ Frequency Stream (SRM + Bayar + 2D FFT) ]
         • 256x256 RGB Image Input                                        • 3 SRM Filters + 1 Bayar-Stamm Conv
         • 512-d Feature Embeddings                                       • 20-Channel 2D FFT (10 Mag + 10 Phase)
                       │                                                  • 512-d Frequency Embeddings
                       │                                                                             │
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
                                           • Raw Logit Output z
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
     [ SciPy L-BFGS-B Temperature Calibration ]                 [ 4-Panel Interpretability Engine ]
     • Log-Temperature Scaling: z / T*                          • (a) RGB Input Face Crop
     • Calibrated Probability: p = Sigmoid(z / T*)              • (b) SRM Noise Residual Map
     • ECE Minimization: 0.0122 ──► 0.0093                      • (c) 2D FFT Magnitude Spectrum
                                                                • (d) Grad-CAM ConvNeXt Attention
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

- **Identity-Disjoint Data Partitioning**: Graph connected-component partitioning (`networkx.Graph`) segregates actor IDs (`id0_id16`) across train, val, and test splits to guarantee 0% identity leakage.
- **Steganographic SRM + Bayar Noise Residuals**: Combines 3 fixed SRM high-pass kernels with 1 learnable Bayar-Stamm constrained convolution to isolate spatial noise residuals before 2D FFT extraction.
- **Probability Calibration via SciPy L-BFGS-B**: Minimizes NLL loss over log-temperatures ($\text{logit} / \exp(\log T)$) to reduce Expected Calibration Error (ECE) from `0.0122` down to `0.0093`.
- **4-Panel Interpretability Diagnostics**: Generates Grad-CAM spatial heatmaps, SRM noise residual maps, and 2D Real FFT magnitude spectrums on demand.
- **Multi-GPU DDP Engine**: Hugging Face `Accelerate` DistributedDataParallel with `SyncBatchNorm` and OpenCV C++ binary loader.
- **Per-Sample Loss Masking**: Excludes corrupt or invalid image frames from backpropagation gradient updates.

---

## Benchmark & Experimental Results

*Evaluated on 2x NVIDIA T4 GPUs at 512x512 crop resolution using PyTorch 2.1 FP16 mixed precision.*

### 1. Held-Out Test Set Metrics (10,528 Crops)
- **Test AUC**: `0.9987`
- **Test F1-Score**: `0.9830`
- **Precision (Fake)**: `0.9686`
- **Recall (Fake)**: `0.9979`
- **Optimal Temperature ($T^*$)**: `1.4788`
- **Expected Calibration Error (ECE)**: `0.0122` (Raw) $\rightarrow$ `0.0093` (Calibrated)

### 2. Per-Generator Sub-Domain Evaluation (2-Class AUC vs Real Faces)

| Generator Sub-Domain | Sample Count | AUC | F1-Score* | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Celeb-DF v2 Synthesis** | 6,639 | `0.9992` | `0.9630` | `99.97%` |
| **FF++ Deepfakes (Pairs 0-199)** | 200 | `0.9963` | `0.4405` | `100.00%` |
| **FF++ Face2Face (Pairs 200-399)** | 200 | `0.9967` | `0.4405` | `100.00%` |
| **FF++ FaceSwap (Pairs 400-599)** | 200 | `0.9961` | `0.4405` | `100.00%` |
| **FF++ NeuralTextures (Pairs 600-799)** | 200 | `0.9940` | `0.4405` | `100.00%` |

*\*Note: FF++ sub-domain F1 scores reflect heavy class imbalance (200 Fakes paired against 2,889 Real test faces) at the operating threshold = 0.01 optimized for 99.8% Fake Recall.*

### 3. True LOTO (Leave-One-Type-Out) Cross-Generator Generalization

| Experiment Fold | Held-Out Target Domain | Category Type | Test Samples | Zero-Shot AUC | Zero-Shot F1 | Precision | Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `FF++ NeuralTextures` | Within-Dataset Cross-Generator | 5,289 | `0.9783` | `0.9230` | `0.9244` | `0.9217` |
| **Fold 2** | `Celeb-DF v2` | Cross-Dataset Zero-Shot Transfer | 82,549 | `0.3234` | `0.1202` | `0.9542` | `0.0641` |

**Note on Fold 2**: Holding out Celeb-DF v2 removes 88% of all fake training crops (66,382 samples), leaving only compressed FaceForensics++ fakes for training. The AUC collapse (0.9783 → 0.3234) is consistent with the cross-dataset domain gap between low-resolution H.264-compressed FF++ synthesis and pristine high-resolution Celeb-DF v2 celebrity synthesis, as documented in Li et al. (CVPR 2020) and Rossler et al. (ICCV 2019). High Precision (0.9542) with near-zero Recall (0.0641) confirms the model's decision boundary is inverted on the unseen domain rather than random.

![LOTO Zero-Shot Generalization](figures/loto_generalization.png)

### 4. Robustness Under Image Degradation

Evaluated on full held-out test split (10,528 crops) at 256×256 resolution using calibrated checkpoint (T\*=1.4788, threshold=0.01).

![Robustness Degradation Sweeps](figures/robustness_degradation.png)

**JPEG Compression**

| Quality | AUC | F1 | ΔAUC |
| :---: | :---: | :---: | :---: |
| Clean (baseline) | `0.9988` | `0.9677` | — |
| Q=100 | `0.9985` | `0.9540` | −0.03% |
| Q=90 | `0.9971` | `0.9693` | −0.17% |
| Q=70 | `0.9852` | `0.9626` | −1.36% |
| Q=50 | `0.9685` | `0.9279` | −3.03% |
| Q=30 | `0.9335` | `0.8528` | −6.53% |

**Gaussian Blur**

| Sigma (σ) | AUC | F1 | ΔAUC |
| :---: | :---: | :---: | :---: |
| Clean (baseline) | `0.9988` | `0.9677` | — |
| σ=0.5 | `0.9981` | `0.9554` | −0.07% |
| σ=1.5 | `0.9748` | `0.8675` | −2.40% |
| σ=3.0 | `0.7375` | `0.8411` | **−26.13%** |

**Gaussian Noise**

| Sigma (σ) | AUC | F1 | ΔAUC |
| :---: | :---: | :---: | :---: |
| Clean (baseline) | `0.9988` | `0.9677` | — |
| σ=5 | `0.9777` | `0.9291` | −2.11% |
| σ=15 | `0.8844` | `0.8732` | −11.44% |
| σ=30 | `0.7544` | `0.8479` | **−24.44%** |

**Resolution Downscaling**

| Scale | AUC | F1 | ΔAUC |
| :---: | :---: | :---: | :---: |
| Clean (baseline) | `0.9988` | `0.9677` | — |
| 0.75× | `0.9952` | `0.9300` | −0.36% |
| 0.50× | `0.9910` | `0.9059` | −0.78% |
| 0.25× | `0.9518` | `0.8631` | −4.70% |

**Note**: AUC collapse under heavy Gaussian blur (σ=3.0, −26%) and noise (σ=30, −24%) is an inherent consequence of the SRM+Bayar+2D FFT frequency branch's dependence on high-frequency manipulation artifacts. Low-pass filtering (blur) and wideband noise physically destroy the spectral signal the frequency stream relies on.

### 5. Benchmark Performance & Visual Interpretability

![4-Panel Interpretability Diagnostics](figures/attention_maps/attention_map_05_fake.png)

*Figure: 4-Panel diagnostic interpretability on a Celeb-DF v2 fake face crop (p = 1.0000, logit = +26.72). (a) Input RGB face crop, (b) SRM 9-filter noise residual map highlighting boundary truncation, (c) 2D FFT magnitude spectrum (Gate = 0.207), and (d) ConvNeXt-Small Grad-CAM spatial heatmap overlay demonstrating precise attention localization on the inner facial mask.*

![ROC Curve](figures/roc_curve.png)

![ECE Reliability Diagram](figures/ece_reliability.png)

![Per-Generator Sub-Domain AUC](figures/per_generator_auc.png)

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/yyouretoast/deepfake-detection.git
cd deepfake-detection
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
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
*The app will automatically launch in your browser at `http://localhost:8501`.*

### 4. Visual Interpretability & Attention Maps *(Requires dataset crops in data/cropped)*

```bash
python scripts/visualize_attention_maps.py --n_samples 6 --output_dir figures/attention_maps
```

### 5. Multi-GPU Training

```bash
accelerate launch --mixed_precision fp16 --num_processes 2 --multi_gpu scripts/train_dual_stream_ddp.py
```

---

## Repository Structure

```
deepfake-detection/
├── app.py                         # Streamlit web interface (Video player + 4-panel diagnostics)
├── config/
│   └── default.yaml               # Configuration parameters (img_size: 512, scale_factor: 1.50)
├── figures/                       # Publication-ready 300 DPI benchmark & interpretability plots
│   ├── attention_maps/            # 4-panel diagnostic visual interpretability figures (RGB, SRM, FFT, Grad-CAM)
│   ├── roc_curve.png              # Test set ROC curve (AUC = 0.9988)
│   ├── ece_reliability.png        # ECE reliability diagram (0.0122 -> 0.0093)
│   ├── robustness_degradation.png # 2x2 grid tracking JPEG, Blur, Noise, Downscaling
│   ├── loto_generalization.png    # Leave-One-Type-Out zero-shot AUC bar chart
│   └── per_generator_auc.png      # Horizontal bar chart of per-generator AUC
├── notebooks/
│   └── master_pipeline.ipynb      # End-to-end master research pipeline notebook
├── src/
│   ├── config.py                  # Configuration parser
│   ├── dataset/
│   │   ├── loader.py              # Identity-safe graph splitter & loader
│   │   └── preprocess.py          # DynamicFaceCropper & 5-point similarity alignment
│   ├── models/
│   │   └── hybrid_detector.py     # ConvNeXt-Small + SRM/Bayar 2D FFT architecture
│   └── utils/
│       ├── checkpoint.py          # Central state-dict cleaning, L-BFGS-B temp fitting & ECE calculation
│       └── temporal_aggregation.py# Frame score pooling (soft-max, EMA, top-K, mean)
├── scripts/                       # Training, evaluation, export & plotting scripts
│   ├── extract_face_crops.py      # Thread-pool multi-threaded face cropper
│   ├── train_dual_stream_ddp.py   # Multi-GPU DDP training pipeline
│   ├── train_loto_experiment.py   # LOTO cross-generator evaluation script
│   ├── evaluate_robustness.py     # Degradation stress-testing script
│   ├── export_test_predictions.py # Raw/calibrated probability exporter
│   ├── generate_benchmark_plots.py# 300 DPI visualization rendering script
│   └── visualize_attention_maps.py# 4-panel SRM + Grad-CAM interpretability engine
├── tests/                         # 54/54 passing unit tests
└── README.md
```

---

## References

1. **Celeb-DF Dataset**: Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. *IEEE/CVF CVPR*.
2. **FaceForensics++ Dataset**: Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *IEEE/CVF ICCV*.
3. **SRM (Steganographic Rich Model)**: Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. *IEEE Transactions on Information Forensics and Security*.
4. **Bayar-Stamm Constrained Conv**: Bayar, B., & Stamm, M. C. (2016). A deep learning approach to universal image manipulation detection. *IEEE IH&MMSec*.
5. **ConvNeXt Architecture**: Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *IEEE/CVF CVPR*.

---

## License

Developed by **Yassin Yasser**. Licensed under the [MIT License](LICENSE).
- **LinkedIn**: [Yassin Yasser](https://www.linkedin.com/in/yassinyasser/)
- **Email**: [yyasso2005@gmail.com](mailto:yyasso2005@gmail.com)
