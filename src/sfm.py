import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensure_homogeneous(m):
    # m: (2, N)
    if m.shape[0] == 2:
        ones = np.ones((1, m.shape[1]))
        return np.vstack([m, ones])  # (3, N)
    
    return m

def correspondence_matrix(p1, p2):
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 have different shapes")

    N = p1.shape[1]

    rows = []
    for i in range(N):
        row = np.kron(p1[:, i].T, p2[:, i].T)  # (9,)
        rows.append(row)

    A = np.vstack(rows)  # (N, 9)
    return A

def precond(m):
    # m: (2, N)

    # 1. centroide (sobre columnas)
    avg = np.mean(m, axis=1, keepdims=True)  # (2,1)

    # 2. centrar puntos
    m_centered = m - avg  # broadcasting

    # 3. distancia media al origen
    d = np.sqrt(np.sum(m_centered**2, axis=0))  # (N,)
    scale = np.sqrt(2) / np.mean(d)

    # 4. escalar
    m_scaled = m_centered * scale

    # 5. matriz de transformación
    T = np.array([
        [scale, 0, -scale * avg[0, 0]],
        [0, scale, -scale * avg[1, 0]],
        [0, 0, 1]
    ])

    return T, m_scaled

def eight_points(p1,p2):
    p1 = ensure_homogeneous(p1);  
    p2 = ensure_homogeneous(p2);
    A = correspondence_matrix(p1,p2)

    _, _, V = np.linalg.svd(A)
    return V.T[:,-1].reshape(3,3)

def compute_fundamental_matrix_from(m1, m2):
    T1, m1 = precond(m1)
    T2, m2 = precond(m2)

    F = eight_points(m1, m2)

    U, D, V = np.linalg.svd(F)
    D[2] = 0
    F = U @ np.diag(D) @ V
    F = T2.T @ F @ T1

    return F

def compute_essential_matrix_from(m1,m2,K1,K2):
    q1 = np.linalg.solve(K1, ensure_homogeneous(m1))
    q2 = np.linalg.solve(K2, ensure_homogeneous(m2))

    # Precondicionamiento
    
    T1, q1n = precond(q1[0:2,:])
    T2, q2n = precond(q2[0:2,:])

    # Convertir a homogéneo nuevamente
    q1n = ensure_homogeneous(q1n)
    q2n = ensure_homogeneous(q2n)

    E_norm = eight_points(q1n, q2n)

    # Desnormalizar
    E = T2.T @ E_norm @ T1

    U, D, V = np.linalg.svd(E)
    s = (D[0]+D[1])/2
    E = U @ np.diag(np.array([s,s,0.])) @ V

    return E

def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.)
        :param F: Fundamental Matrix
        :return: epipole
    """
    # return null space of F (Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V.T[:,-1]
    return e / e[2]

def draw_epipolar_lines(ax, lines, img_shape):
    h, w = img_shape[:2]
    
    for line in lines:
        a, b, c = line
        
        if abs(b) > 1e-6:
            # y = (-a*x - c)/b
            x0, x1 = 0, w
            y0 = (-c - a*x0) / b
            y1 = (-c - a*x1) / b
        else:
            # línea vertical
            x0 = x1 = -c / a
            y0, y1 = 0, h
        color = np.random.rand(3,)
        ax.plot([x0, x1], [y0, y1], color)


def epipolar_lines(p1, p2, F):
    epipolar_lines2 = (F @ p1).T
    epipolar_lines1 = (F.T @ p2).T
    return epipolar_lines1, epipolar_lines2

def essential_factorization(m1,m2,K1,K2,E=None, verbose=False):
    q1 = np.linalg.solve(K1, ensure_homogeneous(m1))
    q2 = np.linalg.solve(K2, ensure_homogeneous(m2))

    S1 = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    R1 = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    if E is None:
        E = compute_essential_matrix_from(m1,m2,K1,K2)

    U, D, V = np.linalg.svd(E)

    for i in range(1,5):
        S = ((-1)**i)*(U @ S1 @ U.T)
        if i==3:
            R1 = R1.T

        t = np.array([S[2,1],S[0,2],S[1,0]]).T
        R = np.linalg.det(U @ V)*(U @ R1 @ V)

        A = np.array([q2[:,0], -R @ q1[:,0]]).T
        z, _, _, _ = np.linalg.lstsq(A, t, rcond=None)

        if (z[0]>0) and (z[1]>0):
            if verbose:
                print("Orientación correcta encontrada")
            break

    return R, t, z

def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    x = np.asarray(x).flatten()
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def linear_triangulation(p1, p2, P1, P2):
    num_points = p1.shape[1]

    p1 = ensure_homogeneous(p1)
    p2 = ensure_homogeneous(p2)

    M = np.zeros((4, num_points))

    for i in range(num_points):
        A = np.vstack([
            skew(p1[:, i]) @ P1,
            skew(p2[:, i]) @ P2
        ])

        _, _, V = np.linalg.svd(A)

        X = V[-1]              # último vector singular
        X = X / X[-1]          # homogenizar

        M[:, i] = X

    return M[:3, :]  # devolver coordenadas 3D
        
        
        

    