import numpy as np


def music(Y: np.ndarray, M: int, pilots: np.ndarray):
    Y_H = np.conj(Y.T)
    R_y = (Y @ Y_H)/M
    eigvals, Q = np.linalg.eig(R_y)
    E = np.diag(eigvals)


    return E