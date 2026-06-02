import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse

from src.features import find_correspondence_points
from src.sfm import linear_triangulation, compute_epipole, skew
from src.ransac_methods import ransac_fundamental_matrix


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", help="Nombre imagen 1 (sin extensión)")
    parser.add_argument("img2", help="Nombre imagen 2 (sin extensión)")
    parser.add_argument("--iters", default=2000, help="Número de iteraciones del RANSAC")
    parser.add_argument("--threshold", default=1.9, help="Threshold del error")
    parser.add_argument("--imgdir", default="dataset/imgs", help="Directorio de imágenes")
    parser.add_argument("--ext", default="JPG", help="Extensión de imagen")
    args = parser.parse_args()

    img1_path = args.img1
    img2_path = args.img2

    full_img1 = f"{args.imgdir}/{img1_path}.{args.ext}"
    full_img2 = f"{args.imgdir}/{img2_path}.{args.ext}"

    img1 = load_image(full_img1)
    img2 = load_image(full_img2)

    kp1_file = f"dataset/keypoints/pts1-{img1_path}.npy"
    kp2_file = f"dataset/keypoints/pts2-{img2_path}.npy"

    if os.path.exists(kp1_file) and os.path.exists(kp2_file):
        print(f"Cargando los keypoints de las imágenes {img1_path}.{args.ext} y {img2_path}.{args.ext}")
        pts1 = np.load(kp1_file)
        pts2 = np.load(kp2_file)
    else:
        print(f"Calculando y guardando los keypoints de las imágenes {img1_path}.{args.ext} y {img2_path}.{args.ext}")
        pts1, pts2 = find_correspondence_points(img1, img2)
        os.makedirs("keypoints", exist_ok=True)
        np.save(kp1_file, pts1)
        np.save(kp2_file, pts2)

    # Asegurar formato (2, N)
    if pts1.shape[0] != 2:
        pts1 = pts1.T
        pts2 = pts2.T

    N = pts1.shape[1]

    if N < 8:
        raise ValueError("No hay suficientes puntos para estimar la matriz esencial")
    

    F = None
    epipole1, epipole2 = None, None

    print("Calculando Matriz Fundamental y epipolos")
    
    F, idx = ransac_fundamental_matrix(pts1, pts2, num_iters=int(args.iters),threshold=float(args.threshold))

    epipole1 = compute_epipole(F)
    epipole2 = compute_epipole(F.T)

    print(f"Matriz Fundamental calculada con {len(idx)} puntos")

    print(F)
    
    print("Epipolos: Izquierda y Derecha")
    print(epipole1)
    print(epipole2)

    print("Calculando Matriz Compatible con F")


    A = (1/(np.linalg.norm(epipole2)**2))*(skew(epipole2) @ F)

    print(A)

    print("Calculando descomposición de la matriz fundamental para extraer las matrices de cámara (F = S·M)")

    U, D, V = np.linalg.svd(F)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,-1,0],[1,0,0],[0,0,0]])

    D = np.diag([D[0],D[1], (D[0]+D[1])/2])

    S = U @ Z @ U.T 

    M_matrix = U @ W.T @ D @ V

    print(S)
    print(M_matrix)
    
    start = 0
    end = pts1.shape[1]    

    pts1_sub = pts1[:, start:end]
    pts2_sub = pts2[:, start:end]

    N = pts1_sub.shape[1]

    # Colores vivos
    colors = plt.cm.hsv(np.linspace(0, 1, N))

    # Convertir BGR (OpenCV) -> RGB (matplotlib)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 9))

    # LEFT IMAGE
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.scatter(
        pts1_sub[0],
        pts1_sub[1],
        s=8,
        c=colors,
        edgecolors='none'
    )
    plt.title("Izquierda")
    plt.axis("off")

    # RIGHT IMAGE
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.scatter(
        pts2_sub[0],
        pts2_sub[1],
        s=8,
        c=colors,
        edgecolors='none'
    )
    plt.title("Derecha")
    plt.axis("off")

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/keypoints_{img1_path}_and_{img2_path}.png", dpi=200)

    plt.show()


    # Matrices de proyección
    print("Matrices de Proyección P1 y P2 con S y M")
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((A, epipole2.reshape(3, 1)))
    print(P1)
    print(P2)

    pts1_in = pts1[:,idx]
    pts2_in = pts2[:,idx]

    M = linear_triangulation(pts1, pts2, P1, P2)

    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(M[0, :], M[1, :], M[2, :], c='k', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    mid_x = sum(x_limits) / 2
    mid_y = sum(y_limits) / 2
    mid_z = sum(z_limits) / 2

    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

    plt.show()


if __name__ == "__main__":
    main()
