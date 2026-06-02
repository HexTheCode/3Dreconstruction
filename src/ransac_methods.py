import numpy as np
import random
from src.sfm import essential_factorization, compute_essential_matrix_from, compute_fundamental_matrix_from

def is_degenerate_sample(pts, eps=1e-3):
    """
    Detecta degeneración comprobando si los puntos son casi colineales.
    pts: (2, N)
    """
    if pts.shape[1] < 3:
        return True

    # Centrar los puntos
    mean = np.mean(pts, axis=1, keepdims=True)
    pts_centered = pts - mean

    # SVD de la matriz 2xN
    U, S, Vt = np.linalg.svd(pts_centered)

    # Si el segundo valor singular es muy pequeño → colinealidad
    return S[1] < eps * S[0]

def ransac_fundamental_matrix(
    pts1, pts2,
    num_iters=2000,
    threshold=1.0,
    sample_number=8
):
    best_F = None
    best_inliers = np.array([], dtype=int)
    best_error = np.inf

    n = pts1.shape[1]

    pts1_h = np.vstack((pts1, np.ones((1, n))))
    pts2_h = np.vstack((pts2, np.ones((1, n))))

    iter_count = 0
    max_iters = num_iters

    while iter_count < max_iters:

        # 🔹 1. Muestreo con control de degeneración (en ambas imágenes)
        valid_sample = False
        for _ in range(10):
            indices = random.sample(range(n), sample_number)
            sample_pts1 = pts1[:, indices]
            sample_pts2 = pts2[:, indices]

            if (not is_degenerate_sample(sample_pts1) and
                not is_degenerate_sample(sample_pts2)):
                valid_sample = True
                break

        if not valid_sample:
            iter_count += 1
            continue

        # 🔹 2. Estimar F
        F = compute_fundamental_matrix_from(sample_pts1, sample_pts2)

        if F is None:
            iter_count += 1
            continue

        # 🔹 3. Error de Sampson (vectorizado y eficiente)
        Fx1 = F @ pts1_h
        Ftx2 = F.T @ pts2_h

        numerador = np.sum(pts2_h * Fx1, axis=0) ** 2
        denominador = (
            Fx1[0]**2 + Fx1[1]**2 +
            Ftx2[0]**2 + Ftx2[1]**2
        )

        valid = denominador > 1e-8
        errors = np.full(n, np.inf)
        errors[valid] = numerador[valid] / denominador[valid]

        inliers = np.where(errors < threshold)[0]

        # 🔹 4. Selección del mejor modelo (inliers + error medio)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F
            best_error = np.mean(errors[inliers]) if len(inliers) > 0 else np.inf

        elif len(inliers) == len(best_inliers) and len(inliers) > 0:
            current_error = np.mean(errors[inliers])
            if current_error < best_error:
                best_inliers = inliers
                best_F = F
                best_error = current_error

        iter_count += 1

    # 🔹 6. Refinamiento final con todos los inliers
    if best_F is not None and len(best_inliers) >= sample_number:
        best_F = compute_fundamental_matrix_from(
            pts1[:, best_inliers],
            pts2[:, best_inliers]
        )
    
    '''
    Fx1 = best_F @ pts1_h
    Ftx2 = best_F.T @ pts2_h

    numerador = np.sum(pts2_h * Fx1, axis=0) ** 2
    denominador = (
    Fx1[0]**2 + Fx1[1]**2 +
    Ftx2[0]**2 + Ftx2[1]**2
    )

    valid = denominador > 1e-8
    errors = np.full(n, np.inf)
    errors[valid] = numerador[valid] / denominador[valid]

    best_inliers = np.where(errors < threshold)[0]
    '''

    return best_F, best_inliers

def ransac_essential_matrix(
    pts1, pts2, K1, K2,
    verbose=False,
    num_iters=2000,
    threshold=1e-3,
    sample_number=8
):
    best_E = None
    best_inliers = np.array([], dtype=int)
    best_error = np.inf

    n = pts1.shape[1]

    pts1_h = np.vstack((pts1, np.ones((1, n))))
    pts2_h = np.vstack((pts2, np.ones((1, n))))

    # 🔹 Normalizar puntos con K (clave para E)
    q1 = np.linalg.solve(K1, pts1_h)
    q2 = np.linalg.solve(K2, pts2_h)

    iter_count = 0

    while iter_count < num_iters:

        # 🔹 1. Muestreo con control de degeneración
        valid_sample = False
        for _ in range(10):
            indices = random.sample(range(n), sample_number)
            sample_pts1 = pts1[:, indices]
            sample_pts2 = pts2[:, indices]

            if (not is_degenerate_sample(sample_pts1) and
                not is_degenerate_sample(sample_pts2)):
                valid_sample = True
                break

        if not valid_sample:
            iter_count += 1
            continue

        # 🔹 2. Estimar E
        E = compute_essential_matrix_from(
            pts1[:, indices],
            pts2[:, indices],
            K1, K2
        )

        if E is None:
            iter_count += 1
            continue

        # 🔹 3. Comprobar orientación (quiralidad)
        try:
            R, t, z = essential_factorization(
                pts1[:, indices],
                pts2[:, indices],
                K1, K2,
                E, verbose=verbose
            )
            if not ((z[0] > 0) and (z[1] > 0)):
                iter_count += 1
                continue
        except:
            iter_count += 1
            continue

        # 🔹 4. Error de Sampson en coordenadas normalizadas
        Eq1 = E @ q1
        Etq2 = E.T @ q2

        numerador = np.sum(q2 * Eq1, axis=0) ** 2
        denominador = (
            Eq1[0]**2 + Eq1[1]**2 +
            Etq2[0]**2 + Etq2[1]**2
        )

        valid = denominador > 1e-8
        errors = np.full(n, np.inf)
        errors[valid] = numerador[valid] / denominador[valid]

        inliers = np.where(errors < threshold)[0]

        # 🔹 5. Selección del mejor modelo
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_E = E
            best_error = np.mean(errors[inliers]) if len(inliers) > 0 else np.inf

        elif len(inliers) == len(best_inliers) and len(inliers) > 0:
            current_error = np.mean(errors[inliers])
            if current_error < best_error:
                best_inliers = inliers
                best_E = E
                best_error = current_error

        iter_count += 1

    # 🔹 6. Refinamiento final
    if best_E is not None and len(best_inliers) >= sample_number:
        best_E = compute_essential_matrix_from(
            pts1[:, best_inliers],
            pts2[:, best_inliers],
            K1, K2
        )

    # 🔹 7. Recalcular inliers finales
    Eq1 = best_E @ q1
    Etq2 = best_E.T @ q2

    numerador = np.sum(q2 * Eq1, axis=0) ** 2
    denominador = (
        Eq1[0]**2 + Eq1[1]**2 +
        Etq2[0]**2 + Etq2[1]**2
    )

    valid = denominador > 1e-8
    errors = np.full(n, np.inf)
    errors[valid] = numerador[valid] / denominador[valid]

    best_inliers = np.where(errors < threshold)[0]

    return best_E, best_inliers
