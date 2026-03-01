<![CDATA[# 👁️ Computer Vision — University Assignments

A collection of three assignments for my **Computer Vision** university course, exploring classical and deep learning approaches to face analysis.

Each assignment builds upon the previous one, progressively moving from traditional image processing techniques to modern deep learning architectures.

---

## 📂 Repository Structure

```
├── assignment_1_face_morphing/          # Face Morphing & Blending
│   ├── src/
│   │   ├── face_morphing.py
│   │   └── haarcascade_frontalface_alt.xml
│   └── results/
│       ├── morphing.gif
│       └── blending.gif
│
├── assignment_2_face_classification_LBP/ # Real vs AI Face Classification (ML)
│   ├── src/
│   │   ├── train_lbp_classifiers.py
│   │   ├── classify_image.py
│   │   └── haarcascade_frontalface_alt.xml
│   └── models/
│
├── assignment_3_face_classification_CNN/ # Real vs AI Face Classification (DL)
│   ├── src/
│   │   ├── train_cnn.py
│   │   └── predict.py
│   └── models/
│
└── README.md
```

---

## 🎭 Assignment 1 — Face Morphing

**Goal:** Create a smooth face morphing animation between two face images, including an intermediate blending video/GIF.

### Approach

| Step | Description |
|------|-------------|
| **1. Face Detection** | Faces are detected using OpenCV's Haar Cascade Classifier to locate the region of interest |
| **2. Face Alignment** | An affine transformation aligns the second face to match the first face's position and scale |
| **3. Landmark Detection** | 68 facial landmarks are extracted using dlib's shape predictor |
| **4. Delaunay Triangulation** | A triangulation mesh is computed over the landmarks using OpenCV's `Subdiv2D` |
| **5. Piecewise Affine Warping** | For each intermediate frame, landmarks are linearly interpolated and each triangle is affinely warped (backward mapping) |
| **6. Alpha Blending** | Forward and reverse warped frames are cross-dissolved using `cv2.addWeighted` to produce the final morphing |

### Tech Stack
`Python` · `OpenCV` · `dlib` · `NumPy` · `imageio`

### Results

<p align="center">
  <img src="assignment_1_face_morphing/results/morphing.gif" width="350" alt="Face Morphing"/>
  &nbsp;&nbsp;&nbsp;
  <img src="assignment_1_face_morphing/results/blending.gif" width="350" alt="Face Blending"/>
</p>
<p align="center"><i>Left: Morphing animation — Right: Blending animation</i></p>

---

## 🔍 Assignment 2 — Real vs AI-Generated Face Classification (Machine Learning)

**Goal:** Classify whether a face image is **real** or **AI-generated** using traditional computer vision features and classic ML classifiers.

### Approach

| Step | Description |
|------|-------------|
| **1. Face Detection & Cropping** | Haar Cascade detects faces; real images are cropped and resized to 150×150, AI-generated images are already pre-cropped |
| **2. Feature Extraction (LBP)** | Two Local Binary Pattern (LBP) configurations are tested: **P=8** (uniform) and **P=256** (default), each producing a normalized histogram |
| **3. Subject-Aware Split** | The dataset is split by subject identity (60/20/20) to avoid data leakage between train/val/test |
| **4. Model Comparison** | Three classifiers are trained and compared, each with and without StandardScaler normalization |
| **5. Best Model Selection** | The model with the highest validation accuracy is selected, retrained, and evaluated on the test set |

### Models Compared

| Classifier | Description |
|-----------|-------------|
| **Random Forest** | Ensemble of decision trees |
| **Logistic Regression** | Linear model with sigmoid activation |
| **Linear SVC** | Support Vector Classifier with linear kernel |

Each model is evaluated in **4 configurations**: `{LBP P=8, LBP P=256}` × `{with standardization, without standardization}`

### Datasets
- **Real faces:** [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- **Fake faces:** AI-generated cropped face images

### Tech Stack
`Python` · `OpenCV` · `scikit-image` · `scikit-learn` · `NumPy` · `pandas` · `matplotlib`

---

## 🧠 Assignment 3 — Real vs AI-Generated Face Classification (Deep Learning)

**Goal:** Improve upon Assignment 2 by replacing handcrafted features with a **custom CNN architecture enhanced with a Self-Attention mechanism**.

### Approach

| Step | Description |
|------|-------------|
| **1. Data Preparation** | ~24K face images (real + AI-generated from two sources) are resized to 224×224 and normalized to [0, 1] |
| **2. Stratified Split** | The dataset is split with stratification to maintain class balance across train/val/test |
| **3. CNN Architecture** | A 5-block convolutional network with ReLU, MaxPooling, and L2 regularization |
| **4. Self-Attention Layer** | A custom `SelfAttention` layer (Query-Key-Value with scaled dot-product) is inserted after the last conv block to capture long-range spatial dependencies |
| **5. Training** | Binary cross-entropy loss, Adam optimizer, EarlyStopping with best-weight restoration |
| **6. Evaluation** | Classification report (precision, recall, F1) and confusion matrix on the test set |

### Architecture Overview

```
Input (224×224×3)
  ├── Conv2D(16) → ReLU → MaxPool
  ├── Conv2D(32) → ReLU → MaxPool
  ├── Conv2D(64) → ReLU → MaxPool
  ├── Conv2D(64, L2) → ReLU → MaxPool
  ├── Conv2D(128, L2) → ReLU → MaxPool
  ├── Reshape → Self-Attention (embed_dim=128)
  ├── Flatten
  ├── Dense(64, L2) → Dropout(0.3)
  ├── Dense(16)
  └── Dense(1, sigmoid) → Output
```

### Self-Attention Mechanism

The custom `SelfAttention` layer implements the standard **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This allows the model to learn **which spatial regions of the feature map are most relevant** for distinguishing real from AI-generated faces.

### Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~92% |

### Datasets
- **AI-generated:** Two separate sources of AI-generated face images (~12K total)
- **Real:** LFW dataset + additional real face images (~12K total)

### Tech Stack
`Python` · `TensorFlow/Keras` · `OpenCV` · `scikit-learn` · `NumPy` · `matplotlib` · `seaborn`

---

## ⚙️ Setup & Requirements

### Prerequisites
- Python 3.8+
- [dlib's shape predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (for Assignment 1)

### Installation

```bash
pip install opencv-python numpy dlib imageio scikit-image scikit-learn pandas matplotlib tensorflow seaborn tqdm
```

> **Note:** The datasets and pre-trained model weights (`.pkl`, `.keras`) are not included in this repository due to their size. You can find the datasets linked in each assignment section above.

---

## 📊 Assignments Summary

| # | Assignment | Approach | Key Technique |
|---|-----------|----------|---------------|
| 1 | Face Morphing | Classical CV | Delaunay Triangulation + Affine Warping |
| 2 | Real vs Fake (ML) | Traditional ML | LBP Features + Random Forest / SVM |
| 3 | Real vs Fake (DL) | Deep Learning | Custom CNN + Self-Attention |

---

## 👤 Author

University Computer Vision Course Assignments

---

<p align="center">
  <i>⭐ If you found this useful, feel free to star the repo!</i>
</p>
]]>
