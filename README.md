# 🎭 Deepfake Detection System

> CNN-based deepfake detector trained on FaceForensics++ achieving **90.4% accuracy** and **0.96 AUC**.

**[🔴 Live Demo](https://huggingface.co/spaces/yyouretoast/deepfake-detector)**

---

## 📌 Overview
This project builds an end-to-end deepfake detection pipeline that:
- Extracts faces from videos using **RetinaFace**
- Classifies them as Real or Fake using **EfficientNetB0**
- Deploys as an interactive **Streamlit** web app

---

## 🏗️ Pipeline
Video → Extract Frames → RetinaFace Face Detection → EfficientNetB0 → Real/Fake

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **90.4%** |
| AUC | **0.96** |
| Fake Precision | 93% |
| Real Recall | 94% |

---

## 🗂️ Dataset
- **FaceForensics++** — 1,000 original YouTube videos
- 6 manipulation types: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, DeepFakeDetection
- 2,400 extracted face images, balanced 50/50

---

## 🧠 Model
- **Base:** EfficientNetB0 pretrained on ImageNet
- **Head:** Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.2) → Sigmoid
- **Training:** Two-phase — freeze base → fine-tune last 50 layers
- **Face Detection:** RetinaFace with 20px padding

---

## 🔍 Grad-CAM
Model focuses on mouth and chin regions — where deepfake artifacts typically appear.

---

## 🚀 Run Locally
```bash
git clone https://github.com/yyouretoast/deepfake-detection
cd deepfake-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Structure

deepfake-detection/

├── app.py               # Streamlit web app

├── requirements.txt     # Dependencies

├── deepfake-detection1.ipynb       # Training notebook

└── README.md

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

---

## 👤 Author
**Yassin Mohamed**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yassinyasser/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:yyasso2005@gmail.com)
