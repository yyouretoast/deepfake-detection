# Model Card for Dual-Stream Deepfake Detection Engine

## Model Details

- **Model Name**: Dual-Stream Deepfake Detection Engine
- **Model Type**: Hybrid Spatial-Spectral Binary Classifier
- **Architecture**: ConvNeXt-Small (Spatial Stream) + SRM/Bayar-Stamm 2D Real FFT (Spectral Stream) fused via Sigmoid Residual Gating.
- **Model Version**: 1.0.0
- **License**: MIT
- **Framework**: PyTorch 2.1+, Accelerate, ONNX Runtime
- **Repository**: [https://github.com/yyouretoast/deepfake-detection](https://github.com/yyouretoast/deepfake-detection)
- **Live Space**: [https://huggingface.co/spaces/yyouretoast/deepfake-detector](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

## Intended Use

- **Primary Intended Use**: Detection of facial manipulation, deepfakes, and synthetic face swapping in digital images and video frames.
- **Out-of-Scope Use Cases**:
  - Full-body deepfake synthesis detection without visible faces.
  - Audio-only deepfake or voice clone detection.
  - Autonomous automated content moderation without human review.

## Quantitative Metrics

### In-Distribution Benchmark (FF++ c23 & Celeb-DF Test Split - 10,528 Crops)

- **Test ROC AUC**: `0.9988` **[95% CI: 0.9985 – 0.9991]**
- **Test F1-Score**: `0.9830` **[95% CI: 0.9809 – 0.9850]**
- **Precision (Fake)**: `0.9686` **[95% CI: 0.9647 – 0.9725]**
- **Recall (Fake)**: `0.9979` **[95% CI: 0.9966 – 0.9987]**
- **Temperature Calibration ($T^*$)**: `1.4788` (Log-Temperature Scaling via SciPy L-BFGS-B)
- **Expected Calibration Error (ECE)**: `0.0122` (Raw) $\rightarrow$ `0.0093` (Calibrated)

### Known Failure Modes & Robustness Vulnerabilities

- **Gaussian Blur Sensitivity ($\sigma=3.0$)**: **−26.13% AUC drop** (`0.7375`). Spatial low-pass filtering erases high-frequency spectral noise residuals extracted by SRM and 2D FFT streams.
- **Gaussian Noise Sensitivity ($\sigma=30$)**: **−24.44% AUC drop** (`0.7544`). Wideband additive noise swamps steganographic noise signatures isolated by Bayar-Stamm convolutions.
- **Cross-Dataset Zero-Shot Transfer (Celeb-DF v2 Held-Out LOTO)**: `0.3234` AUC (Inverted AUC `0.6766`), exhibiting anti-correlated ranking under extreme cross-dataset domain shifts.

## Ethical Considerations, Dual-Use & Safety

- **Adversarial Oracle Risk**: As a publicly queryable deepfake detector, this system could theoretically be probed by malicious actors to iteratively refine synthetic generation or evasion techniques (e.g., tuning blur or compression artifacts until detection is bypassed). To mitigate this risk:
  - Raw network feature embeddings are not exposed; only high-level decision probabilities and 4-panel visual interpretability maps are rendered.
  - Rate-limiting and dynamic thresholding should be enforced in production deployments.
- **Demographic Representation**: Training data is derived from FaceForensics++ and Celeb-DF v2 actor splits. Performance may vary across skin tones, extreme facial angles, heavy cosmetics, or low-light conditions.
- **Human-in-the-Loop Adjudication**: Calibrated probabilities should serve strictly as decision-support signals rather than sole automated adjudicators in legal, forensic, or content-moderation settings.

## Data Provenance & Academic Citation

Training datasets are used under non-commercial academic research agreements:
- **FaceForensics++**: Rössler et al. (ICCV 2019).
- **Celeb-DF v2**: Li et al. (CVPR 2020).
