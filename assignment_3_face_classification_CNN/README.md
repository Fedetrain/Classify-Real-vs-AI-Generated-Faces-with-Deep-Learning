# Assignment 3 — Real vs AI-Generated Face Classification (CNN + Self-Attention)

Deep learning approach to classify real vs AI-generated faces using a **custom CNN with a Self-Attention layer**.

## Architecture

```
Input (224×224×3)
  ├── Conv2D(16) → ReLU → MaxPool
  ├── Conv2D(32) → ReLU → MaxPool
  ├── Conv2D(64) → ReLU → MaxPool
  ├── Conv2D(64, L2) → ReLU → MaxPool
  ├── Conv2D(128, L2) → ReLU → MaxPool
  ├── Reshape → Self-Attention (embed_dim=128)
  ├── Flatten → Dense(64, L2) → Dropout(0.3)
  ├── Dense(16)
  └── Dense(1, sigmoid) → Output
```

### Self-Attention

Custom `SelfAttention` layer implementing scaled dot-product attention (Q, K, V)  to capture long-range spatial dependencies in the feature maps.

## Dataset

- ~24K images total (balanced real/fake)
- **Real:** LFW dataset + additional real face images
- **Fake:** Two separate AI-generated face image sources
- Images resized to 224×224 and normalized to [0, 1]

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~92% |

## Usage

**Train the model:**
```bash
python train_cnn.py
```

**Predict on a single image:**
```bash
python predict.py
```

## Dependencies
`tensorflow` · `opencv-python` · `scikit-learn` · `numpy` · `matplotlib` · `seaborn`
