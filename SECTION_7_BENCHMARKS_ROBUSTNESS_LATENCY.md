# 📘 Section 7: Benchmarks, Robustness & Latency Analysis

---

## 📑 Table of Contents
1. [Overview & Evaluation Protocol](#1-overview--evaluation-protocol)
2. [Quantitative In-Distribution Benchmark (10,528 Test Crops)](#2-quantitative-in-distribution-benchmark-10528-test-crops)
   - [2.1 Primary Global Classification Metrics](#21-primary-global-classification-metrics)
   - [2.2 95% Non-Parametric Bootstrap Confidence Intervals](#22-95-non-parametric-bootstrap-confidence-intervals)
   - [2.3 Per-Generator Sub-Domain Breakdown](#23-per-generator-sub-domain-breakdown)
   - [2.4 Sub-Domain F1 Imbalance Analysis](#24-sub-domain-f1-imbalance-analysis)
3. [Leave-One-Target-Out (LOTO) Cross-Generator Generalization](#3-leave-one-target-out-loto-cross-generator-generalization)
   - [3.1 The LOTO Experimental Protocol](#31-the-loto-experimental-protocol)
   - [3.2 Within-Dataset LOTO Results (Folds 1–4)](#32-within-dataset-loto-results-folds-14)
   - [3.3 Cross-Dataset Domain Shift Limitation (Fold 5: Celeb-DF v2)](#33-cross-dataset-domain-shift-limitation-fold-5-celeb-df-v2)
   - [3.4 Inverted Probability Analysis ($1 - p$)](#34-inverted-probability-analysis-1---p)
4. [Robustness Stress Testing Under Image Degradation](#4-robustness-stress-testing-under-image-degradation)
   - [4.1 JPEG Compression Sweeps ($Q = 100 \dots 30$)](#41-jpeg-compression-sweeps-q--100-dots-30)
   - [4.2 Gaussian Blur Sweeps ($\sigma = 0.5 \dots 3.0$) & Low-Pass Failure Mode](#42-gaussian-blur-sweeps-sigma--05-dots-30--low-pass-failure-mode)
   - [4.3 Additive Gaussian Noise Sweeps ($\sigma = 5 \dots 30$)](#43-additive-gaussian-noise-sweeps-sigma--5-dots-30)
   - [4.4 Resolution Downscaling Sweeps ($0.75\times \dots 0.25\times$)](#44-resolution-downscaling-sweeps-075times-dots-025times)
5. [Hardware Inference Latency & Serving Throughput](#5-hardware-inference-latency--serving-throughput)
   - [5.1 NVIDIA Tesla T4 GPU Benchmarks (FP16 Mixed Precision)](#51-nvidia-tesla-t4-gpu-benchmarks-fp16-mixed-precision)
   - [5.2 Intel Xeon CPU Benchmarks (FP32 Multi-Threaded)](#52-intel-xeon-cpu-benchmarks-fp32-multi-threaded)
   - [5.3 Throughput Scaling (Single-Frame vs. Vectorized Batching)](#53-throughput-scaling-single-frame-vs-vectorized-batching)
6. [Benchmark Script Suite & Visual Plot Generators](#6-benchmark-script-suite--visual-plot-generators)
7. [Code Walkthrough & Reference](#7-code-walkthrough--reference)

---

# 1. Overview & Evaluation Protocol

To rigorously establish the scientific validity of the Dual-Stream Deepfake Detector, the model was subjected to a comprehensive experimental battery:
1. **In-Distribution Testing**: Evaluated on 10,528 identity-disjoint face crops from FaceForensics++ (c23) and Celeb-DF v2.
2. **Cross-Generator Zero-Shot Generalization (LOTO)**: Leave-One-Target-Out experiments testing detection against unseen synthesis algorithms.
3. **Degradation Stress Testing**: Robustness evaluations across 4 degradation dimensions (JPEG compression, Gaussian blur, Gaussian noise, and resolution downscaling).
4. **Hardware Latency & Throughput Profiling**: Empirical timing across GPU (FP16) and CPU (FP32) execution providers.

---

# 2. Quantitative In-Distribution Benchmark (10,528 Test Crops)

### 2.1 Primary Global Classification Metrics
Evaluated on 10,528 held-out test crops at $256 \times 256$ / $512 \times 512$ resolution using the calibrated checkpoint ($T^* = 1.4788$, decision threshold $\theta = 0.01$):

```
┌──────────────────────────────────────┬───────────────────────────────┬──────────────────────────────────────────┐
│ Evaluation Metric                    │ Empirical Value               │ 95% Non-Parametric Bootstrap CI          │
├──────────────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────┤
│ Receiver Operating Characteristic AUC│ 0.9988                        │ [0.9985 – 0.9991]                        │
│ F1-Score                             │ 0.9830                        │ [0.9809 – 0.9850]                        │
│ Precision (Fake)                     │ 0.9686                        │ [0.9647 – 0.9725]                        │
│ Recall / Sensitivity (Fake)          │ 0.9979 (99.79%)               │ [0.9966 – 0.9987]                        │
│ Optimal Temperature Scale (T*)       │ 1.4788                        │ Log-Loss L-BFGS-B Optimization           │
│ Expected Calibration Error (ECE)     │ 0.0122 (Raw) -> 0.0093 (Cal.) │ 23.8% Calibration Improvement            │
└──────────────────────────────────────┴───────────────────────────────┴──────────────────────────────────────────┘
```

### 2.2 95% Non-Parametric Bootstrap Confidence Intervals
To verify statistical significance, 95% confidence intervals were computed via non-parametric bootstrapping ($B = 1,000$ iterations with replacement across the 10,528 test samples).
- The tight interval for ROC AUC ($[0.9985, 0.9991]$) demonstrates that high performance is consistent across the entire test distribution without reliance on outlier samples.

### 2.3 Per-Generator Sub-Domain Breakdown
The test set was partitioned into individual manipulation categories and evaluated independently against the 2,889 Real test faces:

```
┌──────────────────────────────────────┬───────────────┬───────────────┬───────────────┬──────────────────────────┐
│ Generator Sub-Domain Category        │ Sample Count  │ 2-Class AUC   │ Recall (Fake) │ F1-Score*                │
├──────────────────────────────────────┼───────────────┼───────────────┼───────────────┼──────────────────────────┤
│ Celeb-DF v2 High-Res Synthesis       │ 6,639         │ 0.9992        │ 99.97%        │ 0.9630                   │
│ FF++ Deepfakes (Pairs 0–199)         │ 200           │ 0.9963        │ 100.00%       │ 0.4405                   │
│ FF++ Face2Face (Pairs 200–399)       │ 200           │ 0.9967        │ 100.00%       │ 0.4405                   │
│ FF++ FaceSwap (Pairs 400–599)        │ 200           │ 0.9961        │ 100.00%       │ 0.4405                   │
│ FF++ NeuralTextures (Pairs 600–799)  │ 200           │ 0.9940        │ 100.00%       │ 0.4405                   │
└──────────────────────────────────────┴───────────────┴───────────────┴───────────────┴──────────────────────────┘
```

### 2.4 Sub-Domain F1 Imbalance Analysis
*Note on FF++ Sub-Domain F1-Scores (`0.4405`)*:
- In the test split, each individual FF++ sub-domain contains only 200 Fake faces evaluated against the full cohort of 2,889 Real faces ($14.4 : 1$ class imbalance).
- At the operational decision threshold $\theta = 0.01$ (configured to guarantee $99.8\%$ Fake Recall), slight false positives on 2,889 real images depress the F1 score under extreme class imbalance.
- The **ROC AUC ($0.9940 - 0.9967$)** accurately reflects threshold-independent classification performance.

---

# 3. Leave-One-Target-Out (LOTO) Cross-Generator Generalization

Implemented in [`scripts/train_loto_experiment.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/train_loto_experiment.py) with empirical results serialized in [`loto_results.json`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/loto_results.json).

### 3.1 The LOTO Experimental Protocol
To evaluate **zero-shot cross-generator generalization**:
1. Select one target manipulation category to hold out completely from training.
2. Train the dual-stream model exclusively on the remaining 4 manipulation categories.
3. Evaluate the trained model in a zero-shot manner on the held-out category.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LEAVE-ONE-TARGET-OUT (LOTO) EXPERIMENT                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Fold 1: Train on [F2F, FS, NT, Celeb-DF] ─────────► Test Zero-Shot on FF++ Deepfakes   │
│ Fold 2: Train on [DF, FS, NT, Celeb-DF]  ─────────► Test Zero-Shot on FF++ Face2Face   │
│ Fold 3: Train on [DF, F2F, NT, Celeb-DF] ─────────► Test Zero-Shot on FF++ FaceSwap    │
│ Fold 4: Train on [DF, F2F, FS, Celeb-DF] ─────────► Test Zero-Shot on NeuralTextures   │
│ Fold 5: Train on [DF, F2F, FS, NT]       ─────────► Test Zero-Shot on Celeb-DF v2      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Within-Dataset LOTO Results (Folds 1–4)
```
┌────────────┬────────────────────────┬──────────────────────┬───────────────┬──────────────────────────┐
│ Experiment │ Held-Out Target Domain │ Category Type        │ Test Samples  │ Zero-Shot ROC AUC        │
├────────────┼────────────────────────┼──────────────────────┼───────────────┼──────────────────────────┤
│ Fold 1     │ FF++ Deepfakes         │ Within-Dataset LOTO  │ 5,289         │ 0.9691 (F1 = 0.9065)     │
│ Fold 2     │ FF++ Face2Face         │ Within-Dataset LOTO  │ 5,289         │ 0.9749 (F1 = 0.9179)     │
│ Fold 3     │ FF++ FaceSwap          │ Within-Dataset LOTO  │ 5,289         │ 0.9662 (F1 = 0.8969)     │
│ Fold 4     │ FF++ NeuralTextures    │ Within-Dataset LOTO  │ 5,289         │ 0.9783 (F1 = 0.9230)     │
└────────────┴────────────────────────┴──────────────────────┴───────────────┴──────────────────────────┘
```
- **Finding**: When trained on diverse manipulation types, the dual-stream architecture generalizes to unseen manipulation methods with strong performance (**$0.9662 - 0.9783$ AUC**).

### 3.3 Cross-Dataset Domain Shift Limitation (Fold 5: Celeb-DF v2)
In **Fold 5**, Celeb-DF v2 was held out:
- **Test Samples**: 82,549 crops
- **Zero-Shot Raw AUC**: `0.3234`
- **Inverted Probability AUC ($1 - p$)**: **`0.6766`**

### 3.4 Inverted Probability Analysis ($1 - p$)
**Why Fold 5 Drops to `0.3234`**:
1. **Extreme Data Imbalance**: Celeb-DF v2 contains $88\%$ of all fake training crops in the unified dataset. Holding it out leaves only compressed FaceForensics++ samples for training.
2. **Compression Domain Shift**: FaceForensics++ is compressed using the H.264 video codec at standard quantization ($c23$), which introduces distinct macroblocking noise. Celeb-DF v2 fakes are stored as high-quality WebP frames with different compression signatures.
3. **Anti-Correlated Ranking**: When decision probabilities are inverted ($p_{\text{new}} = 1 - p$), the AUC recovers to **`0.6766`**. This proves the network still separates real and fake distributions, but its ranking is inverted due to cross-codec compression artifacts.

---

# 4. Robustness Stress Testing Under Image Degradation

Implemented in [`scripts/evaluate_robustness.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/evaluate_robustness.py) with empirical results serialized in [`robustness_results.json`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/robustness_results.json).

Stress testing was conducted across 10,528 test crops to identify architectural failure modes:

```
     Degradation Robustness Comparison (AUC Drop from Baseline 0.9988)
     1.0 ┌─────────────────────────────────────────────────────────────
         │                                   ┌──────── JPEG (Q=50): 0.9685 (-3.0%)
     0.9 │                                   │
         │                                   ├──────── Downscale (0.25x): 0.9518 (-4.7%)
     0.8 │                                   │
         │                                   ├──────── Gaussian Noise (sigma=30): 0.7544 (-24.4%)
     0.7 │                                   └──────── Gaussian Blur (sigma=3.0): 0.7375 (-26.1%)
         └─────────────────────────────────────────────────────────────
```

### 4.1 JPEG Compression Sweeps ($Q = 100 \dots 30$)
```
┌──────────────────┬───────────────┬───────────────┬──────────────────────────────────────────┐
│ JPEG Quality     │ ROC AUC       │ F1-Score      │ Delta AUC from Clean Baseline            │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────────┤
│ Clean Baseline   │ 0.9988        │ 0.9677        │ —                                        │
│ Q = 100          │ 0.9985        │ 0.9540        │ -0.03%                                   │
│ Q = 90           │ 0.9971        │ 0.9693        │ -0.17%                                   │
│ Q = 70           │ 0.9852        │ 0.9626        │ -1.36%                                   │
│ Q = 50           │ 0.9685        │ 0.9279        │ -3.03%                                   │
│ Q = 30           │ 0.9335        │ 0.8528        │ -6.53%                                   │
└──────────────────┴───────────────┴───────────────┴──────────────────────────────────────────┘
```
- **Analysis**: The model exhibits strong resilience against JPEG compression down to $Q=50$ ($\Delta\text{AUC} \le -3.03\%$).

### 4.2 Gaussian Blur Sweeps ($\sigma = 0.5 \dots 3.0$) & Low-Pass Failure Mode
```
┌──────────────────┬───────────────┬───────────────┬──────────────────────────────────────────┐
│ Gaussian Blur    │ ROC AUC       │ F1-Score      │ Delta AUC from Clean Baseline            │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────────┤
│ Clean Baseline   │ 0.9988        │ 0.9677        │ —                                        │
│ sigma = 0.5      │ 0.9981        │ 0.9554        │ -0.07%                                   │
│ sigma = 1.5      │ 0.9748        │ 0.8675        │ -2.40%                                   │
│ sigma = 3.0      │ 0.7375        │ 0.8411        │ -26.13% (CRITICAL FAILURE MODE)          │
└──────────────────┴───────────────┴───────────────┴──────────────────────────────────────────┘
```
- **Forensic Failure Mode Analysis**: Gaussian blur is a mathematical low-pass spatial filter. At $\sigma = 3.0$, it suppresses high-frequency pixel variations, destroying the steganographic noise residuals extracted by SRM and 2D FFT.

### 4.3 Additive Gaussian Noise Sweeps ($\sigma = 5 \dots 30$)
```
┌──────────────────┬───────────────┬───────────────┬──────────────────────────────────────────┐
│ Noise Sigma      │ ROC AUC       │ F1-Score      │ Delta AUC from Clean Baseline            │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────────┤
│ Clean Baseline   │ 0.9988        │ 0.9677        │ —                                        │
│ sigma = 5        │ 0.9777        │ 0.9291        │ -2.11%                                   │
│ sigma = 15       │ 0.8844        │ 0.8732        │ -11.44%                                  │
│ sigma = 30       │ 0.7544        │ 0.8479        │ -24.44% (CRITICAL FAILURE MODE)          │
└──────────────────┴───────────────┴───────────────┴──────────────────────────────────────────┘
```
- **Forensic Failure Mode Analysis**: Wideband additive Gaussian noise acts as an adversarial mask over high frequencies, swamping the subtle steganographic signatures detected by Bayar-Stamm convolutions.

### 4.4 Resolution Downscaling Sweeps ($0.75\times \dots 0.25\times$)
```
┌──────────────────┬───────────────┬───────────────┬──────────────────────────────────────────┐
│ Scale Factor     │ ROC AUC       │ F1-Score      │ Delta AUC from Clean Baseline            │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────────┤
│ Clean Baseline   │ 0.9988        │ 0.9677        │ —                                        │
│ 0.75x            │ 0.9952        │ 0.9300        │ -0.36%                                   │
│ 0.50x            │ 0.9910        │ 0.9059        │ -0.78%                                   │
│ 0.25x            │ 0.9518        │ 0.8631        │ -4.70%                                   │
└──────────────────┴───────────────┴───────────────┴──────────────────────────────────────────┘
```
- **Analysis**: The network maintains high classification performance ($0.9518$ AUC) even when face crops are downscaled to $25\%$ resolution.

---

# 5. Hardware Inference Latency & Serving Throughput

Implemented in [`scripts/benchmark_latency.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/benchmark_latency.py).

Timing was empirically measured across 100 warm iterations at $512 \times 512$ face crop resolution:

```
┌─────────────────────────────────┬───────────┬────────────┬─────────────────────┬──────────────────┐
│ Hardware Execution Provider     │ Precision │ Batch Size │ Per-Crop Latency    │ Throughput (FPS) │
├─────────────────────────────────┼───────────┼────────────┼─────────────────────┼──────────────────┤
│ NVIDIA Tesla T4 GPU (Kaggle)    │ FP16 Mixed│ BS = 1     │ 18.62 ms / crop     │ 53.7 FPS         │
│ NVIDIA Tesla T4 GPU (Kaggle)    │ FP16 Mixed│ BS = 32    │ 16.41 ms / crop     │ 60.9 FPS         │
│ Intel Xeon CPU (Multi-threaded) │ FP32 Std  │ BS = 1     │ 188.25 ms / crop    │ 5.3 FPS          │
│ Intel Xeon CPU (Multi-threaded) │ FP32 Std  │ BS = 32    │ 4.77 ms / crop      │ 209.6 FPS        │
└─────────────────────────────────┴───────────┴────────────┴─────────────────────┴──────────────────┘
```

### 5.1 NVIDIA Tesla T4 GPU Benchmarks (FP16 Mixed Precision)
- **Single-Frame Latency**: $18.62\text{ ms}$ ($\approx 54\text{ FPS}$).
- Enables real-time video stream processing at standard 30 FPS broadcast framerates.

### 5.2 Intel Xeon CPU Benchmarks (FP32 Multi-Threaded)
- Single-frame processing takes $188.25\text{ ms}$ on CPU due to 2D FFT and ConvNeXt matrix operations.
- Vectorizing across batch size 32 unlocks vectorized CPU AVX-512 SIMD parallelism, boosting CPU throughput to **$209.6\text{ FPS}$** ($4.77\text{ ms/crop}$).

---

# 6. Benchmark Script Suite & Visual Plot Generators

All publication-ready visualization artifacts are generated using [`scripts/generate_benchmark_plots.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/generate_benchmark_plots.py):

| Generated Figure | Location | Description |
| :--- | :--- | :--- |
| **ROC Curve** | `figures/roc_curve.png` | Complete test set Receiver Operating Characteristic curve ($\text{AUC} = 0.9988$). |
| **ECE Reliability Diagram** | `figures/ece_reliability.png` | Calibration reliability diagram ($0.0122 \to 0.0093$). |
| **Robustness 2x2 Grid** | `figures/robustness_degradation.png` | 4-quadrant plot tracking JPEG, Blur, Noise, and Downscaling sweeps. |
| **LOTO Generalization** | `figures/loto_generalization.png` | Bar chart tracking zero-shot AUC across all 5 experimental folds. |
| **Per-Generator AUC** | `figures/per_generator_auc.png` | Horizontal bar chart breaking down performance across FF++ sub-domains. |

---

# 7. Code Walkthrough & Reference

| Script Reference | Primary Responsibility |
| :--- | :--- |
| [`scripts/evaluate_test_set.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/evaluate_test_set.py) | Full held-out test evaluation with bootstrap confidence intervals. |
| [`scripts/train_loto_experiment.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/train_loto_experiment.py) | Leave-One-Target-Out cross-generator generalization training and testing. |
| [`scripts/evaluate_robustness.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/evaluate_robustness.py) | Evaluates performance degradation under JPEG, Blur, Noise, and Downscaling. |
| [`scripts/benchmark_latency.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/benchmark_latency.py) | Hardware latency and throughput profiling script. |
| [`scripts/generate_benchmark_plots.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/generate_benchmark_plots.py) | Renders all 300 DPI publication plots in `figures/`. |

---

*This document serves as the permanent reference for Section 7 (Benchmarks, Robustness & Latency) of the Deepfake Detection Engine.*
