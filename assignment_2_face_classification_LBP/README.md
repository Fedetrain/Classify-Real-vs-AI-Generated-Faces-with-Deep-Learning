# Assignment 2 — Real vs AI-Generated Face Classification (LBP + ML)

Binary classification of real vs AI-generated faces using **Local Binary Patterns (LBP)** feature extraction and classical machine learning models.

## Pipeline

1. **Face Detection & Cropping** — Haar Cascade detects faces, crops and resizes to 150×150
2. **Feature Extraction** — Two LBP configurations:
   - **P=8, R=1** (`uniform` method) → 10-bin histogram
   - **P=256, R=1** (`default` method) → 256-bin histogram
3. **Subject-Aware Splitting** — 60% train / 20% validation / 20% test split by subject identity
4. **Model Training** — Each of 3 classifiers is trained with & without `StandardScaler`:
   - Random Forest
   - Logistic Regression
   - Linear SVC
5. **Best Model Selection** — Highest validation accuracy → retrain & evaluate on test set

## Usage

**Train & evaluate all models:**
```bash
python train_lbp_classifiers.py
```

**Classify a single image:**
```bash
python classify_image.py
```

## Dependencies
`opencv-python` · `scikit-image` · `scikit-learn` · `numpy` · `pandas` · `matplotlib`
