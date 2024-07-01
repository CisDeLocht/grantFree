import numpy as np
from system_functions import generate_gaussian_pilots


def OMP(A, y, N_active):
    r = y
    index_set = []
    for k in range(N_active):
        h_k = A.T @ r
        n_largest_indices = np.argpartition(np.abs(h_k), N_active)[-N_active:]
        missing_indices = np.setdiff1d(n_largest_indices, index_set)
        if len(missing_indices) > 0:
            index_set.append(missing_indices[0])
        else:
            return index_set





if __name__ == "__main__":
    N = 10
    A, a, column_idx = generate_gaussian_pilots(12, N , 100)
    h = np.zeros(100)
    h[column_idx] = np.random.randn(N)
    y = np.reshape(np.dot(A, h), (12,1))
    print(y)
