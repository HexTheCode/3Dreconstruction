# 3D Reconstruction from Stereo Images

Implementation of a complete **3D reconstruction pipeline from two images**, developed as part of a Computer Vision and Multiple View Geometry study project.

The repository includes both **projective** and **Euclidean reconstruction** methods, together with camera calibration, feature matching, robust estimation using RANSAC, and 3D triangulation.

---

## Features

* Camera calibration using calibration images.
* Feature detection and matching.
* Robust estimation of:

  * Fundamental Matrix
  * Essential Matrix
* Epipolar geometry visualization.
* Projective reconstruction from uncalibrated cameras.
* Euclidean reconstruction from calibrated cameras.
* Linear triangulation of 3D points.
* Visualization of reconstructed point clouds.
* Educational Jupyter notebooks explaining the underlying theory.

---

## Reconstruction Pipeline

The implemented workflow follows the classical Structure from Motion (SfM) pipeline:

1. Camera calibration (optional).
2. Feature extraction.
3. Feature matching between stereo images.
4. Outlier rejection using RANSAC.
5. Estimation of the Fundamental or Essential Matrix.
6. Recovery of camera motion.
7. Triangulation of 3D points.
8. Projective or Euclidean reconstruction.
9. Visualization and analysis of results.

---

## Repository Structure

```text
.
├── calibration.py              # Camera calibration
├── projective_recon.py         # Projective reconstruction pipeline
├── euclid_recon.py             # Euclidean reconstruction pipeline
├── requirements.txt
│
├── src/
│   ├── features.py             # Feature extraction and matching
│   ├── ransac_methods.py       # Robust estimation methods
│   └── sfm.py                  # Structure from Motion utilities
│
├── dataset/
│   ├── calibration_imgs/       # Calibration images
│   ├── imgs/                   # Stereo image pairs
│   ├── keypoints/
│   └── matrices/
│
├── notebooks/
│   ├── Ejemplo Práctico Matriz Fundamental.ipynb
│   ├── Ejemplo Cálculo Matriz Esencial.ipynb
│   ├── Reconstrucción Proyectiva.ipynb
│   ├── Ejemplo de Reconstrucción.ipynb
│   └── SIFT Personalizado.ipynb
│
├── plots/                      # Generated figures and results
│
└── old/                        # Legacy implementations and experiments
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/HexTheCode/3Dreconstruction.git
cd 3Dreconstruction
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Camera Calibration

```bash
python calibration.py
```

Computes the intrinsic calibration matrix using the images stored in:

```text
dataset/calibration_imgs/
```

---

### Projective Reconstruction

```bash
python projective_recon.py
```

Performs reconstruction without requiring camera calibration.

---

### Euclidean Reconstruction

```bash
python euclid_recon.py
```

Performs metric reconstruction using calibrated camera parameters.

---

## Example Outputs

The repository generates several visualizations, including:

* Feature correspondences.
* Epipolar lines.
* Estimated camera geometry.
* Reconstructed 3D point clouds.

Examples can be found in the `plots/` directory.

---

## Educational Notebooks

Several Jupyter notebooks are included to illustrate the mathematical foundations of the implemented algorithms:

* Fundamental Matrix estimation.
* Essential Matrix estimation.
* SIFT feature extraction.
* Projective reconstruction.
* Complete reconstruction examples.

These notebooks are intended for learning and experimentation.

---

## References

* Richard Hartley and Andrew Zisserman, *Multiple View Geometry in Computer Vision*.
* David Forsyth and Jean Ponce, *Computer Vision: A Modern Approach*.
* OpenCV Documentation.
* Multiple View Geometry course material from Oxford VGG.

---

## License

This project is distributed under the MIT License.
