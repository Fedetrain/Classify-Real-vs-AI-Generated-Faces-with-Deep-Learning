# Assignment 1 — Face Morphing

Smooth face morphing animation between two face images using Delaunay triangulation and piecewise affine warping.

## Pipeline

1. **Face Detection** — Haar Cascade Classifier (`haarcascade_frontalface_alt.xml`)
2. **Alignment** — Affine transform to align face 2 onto face 1
3. **Landmarks** — 68 facial landmarks via dlib's shape predictor
4. **Triangulation** — Delaunay triangulation over landmarks + corner points (OpenCV `Subdiv2D`)
5. **Warping** — Piecewise affine backward mapping for each interpolated frame
6. **Blending** — Cross-dissolve via `cv2.addWeighted`

## Usage

```bash
python face_morphing.py
```

> **Note:** Requires `shape_predictor_68_face_landmarks.dat` — download from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place in `shape_predictor_68_face_landmarks/`.

## Dependencies
`opencv-python` · `dlib` · `numpy` · `imageio`
