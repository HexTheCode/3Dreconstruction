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

### Camera Calibration (`calibration.py`)

This script estimates the camera's intrinsic parameters and distortion coefficients using a standard chessboard pattern ($7 \times 7$ inner corners), based on Zhang's calibration method.

#### Usage

Ensure your calibration images are located in `dataset/calibratio_imgs/` and the output folder `dataset/matrices/` exists, then run:

```bash
python calibration.py
```

Computes the intrinsic calibration matrix using the images stored in:

```text
dataset/calibration_imgs/
```

---

### Projective Reconstruction (`projective_recon.py`)

This script performs a 3D projective reconstruction from an uncalibrated pair of 2D images. It computes the epipolar geometry, derives a valid projective camera pair up to an unknown projective ambiguity, and reconstructs a sparse 3D point cloud.

#### Pipeline

1. **Feature Correspondence:** Loads or extracts matched 2D keypoints between image pairs via `find_correspondence_points`.
2. **Robust Epipolar Estimation:** Computes the Fundamental Matrix ($F$) using a custom RANSAC implementation (`ransac_fundamental_matrix`) to handle noisy matching outliers.
3. **Epipole Extraction:** Computes the left and right epipoles by solving the null spaces of $F$ and $F^T$ using `compute_epipole`.
4. **Projective Camera Matrices:** Determines a compatible canonical camera pair where $P_1 = [I \mid 0]$ and $P_2 = [[e_2]_\times F \mid e_2]$ (factorized via skew-symmetric matrix configurations).
5. **Triangulation:** Runs `linear_triangulation` with the uncalibrated projection matrices ($P_1, P_2$) to project the 2D coordinates into a 3D projective space ($M$).
6. **Visualization:** Plots the 2D feature points over the image pair and generates the resulting 3D projective point cloud.

#### Usage

Execute the script by passing the targeted image pair names (without extension):

```bash
python projective_recon.py <img1_name> <img2_name> --iters 2000 --threshold 1.9
```

Arguments: 
- `img1`, `img2`: Image filenames (stored in `dataset/imgs/`).
- `--iters`: Number of RANSAC iterations (default: `2000`).
- `threshold`: Error tolerance threshold for RANSAC (default: `1.9`).

---

### Euclidean Reconstruction (`euclid_recon.py`)

This script performs a 3D Euclidean reconstruction from a pair of calibrated 2D images. It computes the relative camera pose, extracts structure via triangulation, and visualizes the resulting 3D point cloud.

#### Pipeline

1. **Feature Correspondence:** Loads or computes matched 2D keypoints between image pairs utilizing `find_correspondence_points`.
2. **Robust Estimation:** Computes the Essential Matrix ($E$) using a custom RANSAC implementation (`ransac_essential_matrix`) to filter out outlier point matches.
3. **Pose Factorization:** Decomposes $E$ into rotation ($R$) and translation ($t$) matrices via `essential_factorization`, enforcing the correct chirality (points must be in front of both cameras).
4. **Triangulation:** Constructs camera projection matrices ($P_1, P_2$) and performs `linear_triangulation` to estimate the 3D coordinates ($M$) of the inlier features.
5. **Visualization:** Plots the 2D feature correspondences using Matplotlib and renders the final 3D sparse point cloud.

#### Usage

Run the script by passing the target image names (without extension) as arguments:

```bash
python euclid_recon.py <img1_name> <img2_name> --Kmatrix <matrix_name> --iters 2000 --threshold 1e-3
```

Arguments: 
- `img1`, `img2`: Image filenames (stored in `dataset/imgs/`).
- `--Kmatrix`: Name of the pre-computed intrinsic matrix file (stored in `dataset/matrices/`).
- `--iters`: Number of RANSAC iterations (default: `2000`).
- `threshold`: Error tolerance threshold for RANSAC (default: `1e-3`).

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
