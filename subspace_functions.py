import numpy as np
import matplotlib.pyplot as plt

def plot_signatures(signatures):
    x_axis = np.arange(len(signatures))
    plt.title("Pilot Signatures for MUSIC")
    plt.stem(x_axis, signatures)
    plt.show()

def music(Y: np.ndarray, M: int, K: int, pilots: np.ndarray, plot: bool):
    Y_H = np.conj(Y.T)
    R_y = (Y @ Y_H)/M
    eigvals, Q = np.linalg.eig(R_y)
    eigvals = np.real(eigvals)

    sort_idx = np.argsort(eigvals)[::-1]
    sorted_eigvals = eigvals[sort_idx]
    sorted_Q = Q[:, sort_idx]

    Q_n = sorted_Q[:, -(Q.shape[1]-K):]
    Q_n_H = np.conj(Q_n.T)

    signatures = 1 / (np.linalg.norm(Q_n_H @ pilots, axis=0)**2)
    signature_idx = np.argpartition(signatures, -K)[-K:]

    if(plot):
        plot_signatures(signatures)

    return signatures, signature_idx