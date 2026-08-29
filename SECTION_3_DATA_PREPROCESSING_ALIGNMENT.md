# 📘 Section 3: Face Detection, Alignment & Data Engineering

---

## 📑 Table of Contents
1. [Overview & Engineering Objectives](#1-overview--engineering-objectives)
2. [Multi-Engine Face Detection Cascade](#2-multi-engine-face-detection-cascade)
   - [2.1 Primary Engine: OpenCV YuNet ONNX (`cv2.FaceDetectorYN`)](#21-primary-engine-opencv-yunet-onnx-cv2facedetectoryn)
   - [2.2 Thread-Local Isolation for Multi-Worker DataLoaders](#22-thread-local-isolation-for-multi-worker-dataloaders)
   - [2.3 Fallback Engine 1: MTCNN Multi-Task Cascaded CNN](#23-fallback-engine-1-mtcnn-multi-task-cascaded-cnn)
   - [2.4 Fallback Engine 2: Haar Feature Cascade Classifier](#24-fallback-engine-2-haar-feature-cascade-classifier)
   - [2.5 Fallback Engine 3: Safe Center Square Crop](#25-fallback-engine-3-safe-center-square-crop)
3. [Bounding Box Expansion: The $1.50\times$ Rule](#3-bounding-box-expansion-the-150times-rule)
   - [3.1 Why Tight Face Crops Destroy Deepfake Detection](#31-why-tight-face-crops-destroy-deepfake-detection)
   - [3.2 Mathematical Formulation of Bounding Box Padding](#32-mathematical-formulation-of-bounding-box-padding)
   - [3.3 Replicated Border Handling (`cv2.BORDER_REPLICATE`)](#33-replicated-border-handling-cv2border_replicate)
4. [5-Point Landmark Similarity Transform & Canonical Alignment](#4-5-point-landmark-similarity-transform--canonical-alignment)
   - [4.1 Anatomical Landmarks (Eyes, Nose, Mouth Corners)](#41-anatomical-landmarks-eyes-nose-mouth-corners)
   - [4.2 Scale-Compensated Canonical Landmark Coordinates](#42-scale-compensated-canonical-landmark-coordinates)
   - [4.3 Least-Median-of-Squares (LMEDS) Affine Estimation](#43-least-median-of-squares-lmeds-affine-estimation)
   - [4.4 Determinant Verification to Prevent Degenerate Warping](#44-determinant-verification-to-prevent-degenerate-warping)
   - [4.5 Affine Image Warping (`cv2.warpAffine`)](#45-affine-image-warping-cv2warpaffine)
5. [2D Cosine Window Edge Tapering](#5-2d-cosine-window-edge-tapering)
   - [5.1 The Problem of Boundary Step-Discontinuities](#51-the-problem-of-boundary-step-discontinuities)
   - [5.2 Mathematical Formulation of 2D Cosine Taper](#52-mathematical-formulation-of-2d-cosine-taper)
   - [5.3 Proof of Eliminating FFT Cross-Spikes](#53-proof-of-eliminating-fft-cross-spikes)
6. [Graph-Based Zero-Leakage Dataset Splitting](#6-graph-based-zero-leakage-dataset-splitting)
   - [6.1 The Identity Leakage Vulnerability](#61-the-identity-leakage-vulnerability)
   - [6.2 Regex Identity Parsing (`idA_idB`)](#62-regex-identity-parsing-ida_idb)
   - [6.3 NetworkX Graph Construction & Connected Components](#63-networkx-graph-construction--connected-components)
   - [6.4 Partitioning Algorithm into Train, Val, and Test Splits](#64-partitioning-algorithm-into-train-val-and-test-splits)
7. [Data Augmentation & Forensic Signal Preservation](#7-data-augmentation--forensic-signal-preservation)
   - [7.1 Geometric & Photometric Transformations](#71-geometric--photometric-transformations)
   - [7.2 Why Gaussian Blur & JPEG Compression Are Excluded From Training](#72-why-gaussian-blur--jpeg-compression-are-excluded-from-training)
8. [PyTorch Dataset & DataLoader Architecture](#8-pytorch-dataset--dataloader-architecture)
   - [8.1 Single-Frame `DeepfakeDataset`](#81-single-frame-deepfakedataset)
   - [8.2 Video Sequence Dataset with `A.ReplayCompose`](#82-video-sequence-dataset-with-areplaycompose)
   - [8.3 Multiprocessing Safety & Thread Contention Elimination](#83-multiprocessing-safety--thread-contention-elimination)
9. [Code Walkthrough & Reference](#9-code-walkthrough--reference)

---

# 1. Overview & Engineering Objectives

The data pipeline in [`src/dataset/`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/) is responsible for turning raw video files and uncropped images into standardized, forensic-ready facial tensors.

The data pipeline must satisfy three strict requirements:
1. **Geometric Standardization**: Faces must be aligned to canonical anatomical landmarks so the spatial stream can focus on facial features rather than head tilt or pose variations.
2. **Boundary Preservation**: The bounding box must capture forehead, chin, and jawline areas where face-swapping blending seams reside.
3. **Forensic Integrity & Zero Identity Leakage**: Data splitting must prevent the model from memorizing individual human faces, and data augmentation must not destroy high-frequency steganographic noise signals.

```
[ Raw Video Frame / Input Image ]
               │
               ▼
[ 1. Multi-Engine Face Detector ] ──► YuNet ONNX ──► Fallback: MTCNN ──► Fallback: Haar
               │
               ├──────────────────────────────────────────────┐
               ▼                                              ▼
     [ Bounding Box (x1,y1,x2,y2) ]            [ 5-Point Landmarks ]
               │                                      (Eyes, Nose, Mouth)
               ▼                                              │
  [ 1.50x Bbox Scale Expansion ]                              ▼
  • Captures jawline & forehead seams           [ Canonical Landmark Warp ]
  • Border replication padding                  • LMEDS Affine Estimation
  • 2D Cosine window edge tapering              • Scale-compensated target points
               │                                              │
               └──────────────────────┬───────────────────────┘
                                      ▼
                      [ Final 512x512 Aligned Crop ]
                                      │
                                      ▼
                  [ Graph-Based Identity-Disjoint Split ]
                  • networkx.Graph connected components
                  • 0% Identity Leakage across Train / Val / Test
```

---

# 2. Multi-Engine Face Detection Cascade

Implemented in [`DynamicFaceCropper`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L71-L457).

### 2.1 Primary Engine: OpenCV YuNet ONNX (`cv2.FaceDetectorYN`)
The primary face detector is **YuNet**, an ultra-lightweight ($~300\text{ KB}$), highly accurate convolutional face detection network developed for OpenCV.
- **ONNX Model File**: `face_detection_yunet_2023mar.onnx` (automatically downloaded if missing by [`get_yunet_model_path`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L26-L58)).
- **Configuration**:
  - `score_threshold = 0.6` (confidence threshold for face detection)
  - `nms_threshold = 0.3` (Non-Maximum Suppression threshold to eliminate duplicate boxes)
  - `top_k = 5000`
- **Outputs**: For each detected face, YuNet outputs a 14-element vector:
  $$\mathbf{f} = [x, y, w, h, x_{\text{reye}}, y_{\text{reye}}, x_{\text{leye}}, y_{\text{leye}}, x_{\text{nose}}, y_{\text{nose}}, x_{\text{rmouth}}, y_{\text{rmouth}}, x_{\text{lmouth}}, y_{\text{lmouth}}]$$

### 2.2 Thread-Local Isolation for Multi-Worker DataLoaders
When PyTorch's `DataLoader` loads data with multiple worker processes (`num_workers=4`), multiple threads and subprocesses run concurrently.
- Standard OpenCV C++ objects are not thread-safe. If multiple threads call `.detect()` on a shared `cv2.FaceDetectorYN` instance, memory corruption or segmentation faults occur.
- In [`DynamicFaceCropper._get_thread_yunet`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L107-L126), detector instances are isolated per-thread using Python's `threading.local()`:
  ```python
  def _get_thread_yunet(self) -> Optional[Any]:
      if not hasattr(self._local, "yunet"):
          cached_path = get_cached_yunet_path()
          self._local.yunet = cv2.FaceDetectorYN.create(
              model=cached_path, config="", input_size=(300, 300),
              score_threshold=0.6, nms_threshold=0.3, top_k=5000
          )
      return self._local.yunet
  ```

### 2.3 Fallback Engine 1: MTCNN Multi-Task Cascaded CNN
If YuNet fails to detect a face (e.g., extreme low light or profile angle), the pipeline falls back to **MTCNN** ([`src/dataset/preprocess.py#L364-L378`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L364-L378)).
- Runs a 3-stage convolutional pyramid (P-Net $\to$ R-Net $\to$ O-Net) to extract bounding boxes and facial landmark coordinates on the GPU/CPU.

### 2.4 Fallback Engine 2: Haar Feature Cascade Classifier
If MTCNN is unavailable or fails, the pipeline falls back to classical CPU **Haar Feature Cascades** (`haarcascade_frontalface_default.xml` in [`src/dataset/preprocess.py#L161-L176`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L161-L176)).

### 2.5 Fallback Engine 3: Safe Center Square Crop
If all detection engines fail to locate a face, [`_center_crop`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L332-L341) extracts the central square region of the image, ensuring that data loading never crashes due to missing detections.

---

# 3. Bounding Box Expansion: The $1.50\times$ Rule

### 3.1 Why Tight Face Crops Destroy Deepfake Detection
Standard face recognition models (e.g., FaceNet, ArcFace) use tight bounding boxes ($1.0\times$ scale) cropped directly across the eyebrows and chin.
- **In deepfake detection, tight crops are fatal**: The neural synthesis mask is blended with the background along the **outer perimeter** (jawline, hairline, forehead, neck).
- A $1.0\times$ tight crop cuts off the blending seam, removing the primary forensic signal and reducing detector AUC by over $15\%$.

### 3.2 Mathematical Formulation of Bounding Box Padding
Given a detected bounding box $[x_1, y_1, x_2, y_2]$:
1. Compute width $w = x_2 - x_1$, height $h = y_2 - y_1$.
2. Compute bounding box center:
   $$c_x = x_1 + \frac{w}{2}, \quad c_y = y_1 + \frac{h}{2}$$
3. Compute expanded square crop dimension with scale factor $S = 1.50$:
   $$\text{side} = \max(w, h) \times 1.50$$
4. Compute expanded crop coordinates:
   $$x_1' = \text{round}\left( c_x - \frac{\text{side}}{2} \right), \quad x_2' = \text{round}\left( c_x + \frac{\text{side}}{2} \right)$$
   $$y_1' = \text{round}\left( c_y - \frac{\text{side}}{2} \right), \quad y_2' = \text{round}\left( c_y + \frac{\text{side}}{2} \right)$$

```
┌─────────────────────────────────────────────────────────────┐
│ Expanded 1.50x Crop Boundary                                │
│    ┌───────────────────────────────────────────────────┐    │
│    │ Forehead & Hairline Blending Seam                 │    │
│    │       ┌───────────────────────────────────┐       │    │
│    │       │ 1.0x Tight Crop (Eyes, Nose)      │       │    │
│    │       └───────────────────────────────────┘       │    │
│    │ Jawline & Neck Blending Boundary                  │    │
│    └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Replicated Border Handling (`cv2.BORDER_REPLICATE`)
If the expanded crop extends beyond the image boundaries (e.g., $x_1' < 0$ or $y_2' > H_{\text{img}}$):
- The image is padded using `cv2.copyMakeBorder` with `cv2.BORDER_REPLICATE`:
  $$\text{pad}_{\text{left}} = \max(0, -x_1'), \quad \text{pad}_{\text{top}} = \max(0, -y_1')$$
  $$\text{pad}_{\text{right}} = \max(0, x_2' - W_{\text{img}}), \quad \text{pad}_{\text{bottom}} = \max(0, y_2' - H_{\text{img}})$$
- Edge pixels are replicated outward rather than padded with black zeros, preventing extreme dark edge contrasts.

---

# 4. 5-Point Landmark Similarity Transform & Canonical Alignment

Implemented in [`DynamicFaceCropper._crop_single_box`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L197-L304).

### 4.1 Anatomical Landmarks (Eyes, Nose, Mouth Corners)
YuNet and MTCNN extract 5 specific anatomical landmark coordinates:
$$\mathbf{P}_{\text{src}} = \begin{bmatrix} x_{\text{left\_eye}} & y_{\text{left\_eye}} \\ x_{\text{right\_eye}} & y_{\text{right\_eye}} \\ x_{\text{nose}} & y_{\text{nose}} \\ x_{\text{left\_mouth}} & y_{\text{left\_mouth}} \\ x_{\text{right\_mouth}} & y_{\text{right\_mouth}} \end{bmatrix} \in \mathbb{R}^{5 \times 2}$$

### 4.2 Scale-Compensated Canonical Landmark Coordinates
Standard landmark alignment defines canonical facial positions within a normalized $[0.0, 1.0]$ square:
$$\mathbf{P}_{\text{base}} = \begin{bmatrix} 0.30 & 0.35 \\ 0.70 & 0.35 \\ 0.50 & 0.50 \\ 0.35 & 0.70 \\ 0.65 & 0.70 \end{bmatrix}$$
**The Scale Mismatch Bug**:
If we warp $\mathbf{P}_{\text{src}}$ directly to $\mathbf{P}_{\text{base}} \times \text{target\_size}$, the affine warp zooms in tightly on the face ($1.0\times$ zoom), neutralizing the $1.50\times$ bounding box expansion.

**The Solution**:
In [`src/dataset/preprocess.py#L269-L284`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L269-L284), canonical landmarks are shifted inward by a margin fraction $\gamma$:
$$\gamma = \frac{1.0 - \frac{1.0}{S}}{2.0}, \quad \text{where } S = 1.50 \implies \gamma = \frac{1 - 1/1.5}{2} = 0.1667$$
$$\mathbf{P}_{\text{canonical}} = \Big( \mathbf{P}_{\text{base}} \cdot (1.0 - 2\gamma) + \gamma \Big) \times \text{target\_size}$$
This shifts canonical target points toward the center, guaranteeing that the affine warped face retains the exact same $1.50\times$ expanded zoom.

### 4.3 Least-Median-of-Squares (LMEDS) Affine Estimation
To align $\mathbf{P}_{\text{src}}$ to $\mathbf{P}_{\text{canonical}}$, we compute a 2D affine transformation matrix $\mathbf{M} \in \mathbb{R}^{2 \times 3}$:
$$\mathbf{M} = \begin{bmatrix} a & b & t_x \\ -b & a & t_y \end{bmatrix}$$
where:
- Scale $s = \sqrt{a^2 + b^2}$
- Rotation angle $\theta = \text{atan2}(b, a)$
- Translation vector $\mathbf{t} = [t_x, t_y]^T$

Estimated via OpenCV's Least Median of Squares (`cv2.LMEDS`):
```python
M, inliers = cv2.estimateAffinePartial2D(
    np.array(landmarks), canonical_landmarks, method=cv2.LMEDS
)
```
`estimateAffinePartial2D` restricts the transformation to rotation, uniform scaling, and translation (rigid similarity transform), **prohibiting shear distortion**.

### 4.4 Determinant Verification to Prevent Degenerate Warping
If facial landmarks are noisy (e.g., face partially occluded), $\mathbf{M}$ can collapse or invert.
In [`src/dataset/preprocess.py#L291-L292`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L291-L292), the determinant of the transformation is checked:
$$\det(\mathbf{M}) = |a \cdot a - (-b \cdot b)| = a^2 + b^2 = s^2$$
$$\text{Condition: } 0.2 < \det(\mathbf{M}) < 5.0$$
If $\det(\mathbf{M})$ falls outside $[0.2, 5.0]$, the warp is rejected, and the unwarped bounding box crop is used instead.

### 4.5 Affine Image Warping (`cv2.warpAffine`)
When the matrix $\mathbf{M}$ is validated, the image is resampled into target dimensions ($512 \times 512$ or $256 \times 256$):
```python
aligned_warped_crop = cv2.warpAffine(
    image_rgb, M, (out_size, out_size),
    flags=cv2.INTER_AREA,
    borderMode=cv2.BORDER_REPLICATE
)
```

---

# 5. 2D Cosine Window Edge Tapering

Implemented in [`DynamicFaceCropper._apply_cosine_edge_taper`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L178-L196).

### 5.1 The Problem of Boundary Step-Discontinuities
When an expanded face crop touches the edge of a video frame and requires padding, the transition between original pixels and padded border pixels creates a sharp rectangular step edge.
- In the spatial domain, this boundary is minor.
- In the **2D Fourier frequency domain**, a sharp rectangular step acts as a heavy step function:
  $$\mathcal{F}\{\text{rect}(t)\} = \text{sinc}(f)$$
- This introduces severe horizontal and vertical **cross-spikes** throughout the 2D FFT spectrum that drown out subtle steganographic noise artifacts.

### 5.2 Mathematical Formulation of 2D Cosine Taper
To eliminate rectangular boundary spikes, a 2D Cosine window (Hann-style edge taper) smoothly attenuates the outer $5\%$ border of padded crops to zero:
For an image of height $H$ and width $W$, with border ratio $\beta = 0.05$:
$$T_h = \max(1, \lfloor H \cdot \beta \rfloor), \quad T_w = \max(1, \lfloor W \cdot \beta \rfloor)$$

1. **Vertical 1D Window $w_y(y)$**:
   $$w_y(y) = \begin{cases} 0.5 \cdot \left(1 - \cos\left(\frac{\pi y}{T_h}\right)\right), & 0 \le y < T_h \\ 1.0, & T_h \le y \le H - T_h \\ 0.5 \cdot \left(1 - \cos\left(\frac{\pi (H - y)}{T_h}\right)\right), & H - T_h < y < H \end{cases}$$

2. **Horizontal 1D Window $w_x(x)$**:
   $$w_x(x) = \begin{cases} 0.5 \cdot \left(1 - \cos\left(\frac{\pi x}{T_w}\right)\right), & 0 \le x < T_w \\ 1.0, & T_w \le x \le W - T_w \\ 0.5 \cdot \left(1 - \cos\left(\frac{\pi (W - x)}{T_w}\right)\right), & W - T_w < x < W \end{cases}$$

3. **2D Outer Product Window**:
   $$\mathbf{W}_{\text{2D}}(y, x) = w_y(y) \otimes w_x(x) \in [0.0, 1.0]^{H \times W}$$
4. **Tapered Crop Application**:
   $$I_{\text{tapered}}(y, x) = \text{clip}\Big( I(y, x) \odot \mathbf{W}_{\text{2D}}(y, x), \; 0, \; 255 \Big)$$

```
     1D Cosine Window Profile across Image Width:
     1.0 ────────┌────────────────────────────────┐────────
                 │                                │
     0.5 ───────/                                  \───────
               /                                    \
     0.0 ─────┴──────────────────────────────────────┴─────
             0   T_w                                W-T_w  W
```

### 5.3 Proof of Eliminating FFT Cross-Spikes
By forcing the boundary intensity and its first derivative smoothly to zero ($I(0) = 0, \frac{\partial I}{\partial x}|_{x=0} = 0$), the circular boundary condition of the Discrete Fourier Transform is satisfied:
$$I(0, y) \approx I(W-1, y) \approx 0$$
This eliminates artificial boundary frequency cross-lines in the 2D FFT spectrum.

---

# 6. Graph-Based Zero-Leakage Dataset Splitting

Implemented in [`perform_graph_split`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L133-L213).

### 6.1 The Identity Leakage Vulnerability
In manipulated datasets (FaceForensics++, Celeb-DF v2), fake videos are created by swapping a source actor's face onto a target actor's face.
- Example: Video `042_189.mp4` swaps Actor `042` onto Actor `189`.
- **The Leakage Danger**: If Video `042_189.mp4` is placed in the Training set, and Video `042_310.mp4` or real Video `042.mp4` is placed in the Test set, the neural network learns to recognize the biometric features of Actor `042` and predicts "Fake" on that basis.
- This creates **artificially inflated test scores** that fail completely when deployed in the real world.

### 6.2 Regex Identity Parsing (`idA_idB`)
In [`extract_identities`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L38-L81), filenames are parsed using regular expressions to extract both actor identifiers:
```python
# Priority 1: id{N}_id{M} (Celeb-DF format: "id0_id16_0001.webp")
match_alpha = re.search(r"(id\d+)_(id\d+)", clean_base)
if match_alpha:
    return match_alpha.group(1), match_alpha.group(2)

# Priority 2: {N}_{M} (FaceForensics++ format: "042_189.mp4")
match_num = re.search(r"(\d+)_(\d+)", clean_base)
if match_num and len(g1) <= 3 and len(g2) <= 3:
    return g1, g2

# Priority 3: Single numeric ID (Real videos: "042.mp4")
match_single = re.search(r"(\d+)", clean_base)
if match_single:
    return match_single.group(1), match_single.group(1)
```

### 6.3 NetworkX Graph Construction & Connected Components
To guarantee zero leakage, we model all dataset samples as an **Undirected Graph** $G = (V, E)$:
- **Vertices $V$**: Individual human identities ($\text{Actor } 042, \text{Actor } 189, \dots$).
- **Edges $E$**: Any video connecting two identities (e.g., edge between $042$ and $189$).

```
                      Graph Connected Component Subgraph:
                           [ Actor 042 ]
                            /         \
                           / (042_189) \ (042_310)
                          ▼             ▼
                    [ Actor 189 ]  [ Actor 310 ]
                          \             /
                           \ (189_310) /
                            ▼         ▼
                           [ Actor 505 ]
       ─────────────────────────────────────────────────────────────
       RULE: All 4 actors (042, 189, 310, 505) and ALL their videos
             MUST be assigned to the SAME split (e.g. 100% Training).
```

### 6.4 Partitioning Algorithm into Train, Val, and Test Splits
1. Extract all connected components:
   $$\mathcal{C} = \{ C_1, C_2, \dots, C_K \} = \text{nx.connected\_components}(G)$$
2. Compute sample volume for each component:
   $$N(C_k) = \text{number of image crops belonging to identities in } C_k$$
3. Sort components in descending order of size: $N(C_1) \ge N(C_2) \ge \dots \ge N(C_K)$.
4. Bin components into **Validation Split ($15\%$)**, **Test Split ($15\%$)**, and **Train Split ($70\%$)**:
   - Accumulate largest components into Validation set until $N_{\text{val}} \approx 0.15 \times N_{\text{total}}$.
   - Accumulate subsequent components into Test set until $N_{\text{test}} \approx 0.15 \times N_{\text{total}}$.
   - Place all remaining components into Train set ($N_{\text{train}} \approx 0.70 \times N_{\text{total}}$).

**Guaranteed Mathematical Property**:
$$\text{Identities}(\text{Train}) \cap \text{Identities}(\text{Val}) = \emptyset$$
$$\text{Identities}(\text{Train}) \cap \text{Identities}(\text{Test}) = \emptyset$$
$$\text{Identities}(\text{Val}) \cap \text{Identities}(\text{Test}) = \emptyset$$

---

# 7. Data Augmentation & Forensic Signal Preservation

Implemented in [`get_transforms`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L215-L243).

### 7.1 Geometric & Photometric Transformations
```python
train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

### 7.2 Why Gaussian Blur & JPEG Compression Are Excluded From Training
In typical computer vision tasks (e.g., ImageNet classification), aggressive data augmentations like heavy Gaussian Blur (`A.GaussianBlur`) and JPEG Compression (`A.ImageCompression`) are applied to improve robustness.

**Why they are omitted from deepfake training**:
- The frequency stream relies on **subtle high-frequency steganographic noise residuals** ($R(x, y)$) extracted by SRM filters and 2D FFT.
- Applying Gaussian blur or heavy JPEG compression during training acts as an aggressive low-pass filter, erasing the high-frequency forensic signal from the training images.
- Augmenting with blur forces the network to ignore the frequency branch, causing the gating vector to collapse to zero ($g \to 0$).
- Therefore, training is performed on **clean, high-frequency-preserving crops**. Robustness to blur and compression is evaluated separately post-training via [`scripts/evaluate_robustness.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/evaluate_robustness.py).

---

# 8. PyTorch Dataset & DataLoader Architecture

Implemented in [`src/dataset/loader.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py).

### 8.1 Single-Frame `DeepfakeDataset`
[`DeepfakeDataset`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L254-L285) inherits from `torch.utils.data.Dataset`:
- Input: List of tuples `(image_path, integer_label)`.
- `__getitem__(idx)`: Loads image via OpenCV, converts BGR to RGB, applies Albumentations transformation pipeline, and returns `(tensor_3xHxW, label_int)`.

### 8.2 Video Sequence Dataset with `A.ReplayCompose`
For temporal video models, [`SequenceVideoDataset`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L287-L365) loads sequences of $T$ consecutive frames:
- **`A.ReplayCompose` Synchronization**: If random augmentations (e.g., horizontal flip or rotation) are applied to frame $t_1$, the exact same random parameters must be replayed across frames $t_2, \dots, t_T$ to maintain temporal continuity.
- Returns `(seq_tensor_Tx3xHxW, label_tensor, padding_mask_T)`.

### 8.3 Multiprocessing Safety & Thread Contention Elimination
In [`_worker_init_fn`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L367-L370):
```python
def _worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
```
- By default, OpenCV spawns internal thread pools for image operations.
- When combined with PyTorch DataLoader multiprocessing (`num_workers=4`), hundreds of threads compete for CPU cores, causing severe CPU thrashing and slowdowns.
- Calling `cv2.setNumThreads(0)` disables internal OpenCV multi-threading, letting PyTorch manage worker processes efficiently.

---

# 9. Code Walkthrough & Reference

| File / Component | Primary Responsibility |
| :--- | :--- |
| [`src/dataset/preprocess.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py) | Contains [`DynamicFaceCropper`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/preprocess.py#L71-L457), YuNet detector management, 5-point affine alignment, and 2D Cosine window edge tapering. |
| [`src/dataset/loader.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py) | Contains [`perform_graph_split`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/src/dataset/loader.py#L133-L213), regex identity extractors, Albumentations pipelines, `DeepfakeDataset`, and `SequenceVideoDataset`. |
| [`scripts/extract_face_crops.py`](file:///c:/Users/Yassin/Desktop/code/deepfake-detection-main/scripts/extract_face_crops.py) | Standalone multi-threaded script for offline batch cropping of video datasets into lossless WebP face crops. |

---

*This document serves as the permanent reference for Section 3 (Data Preprocessing & Alignment) of the Deepfake Detection Engine.*
