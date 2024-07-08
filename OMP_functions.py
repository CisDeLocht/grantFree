import numpy as np
from system_functions import *

def calculate_coherence(A):
    columns = A.shape[1]
    v = []
    for i in range(columns):
        for j in range(columns):
            if(i != j):
                x_i = A[:,i]
                x_j = A[:,j]
                T = np.abs(np.conj(x_i).T @ x_j)
                N = np.linalg.norm(x_i) * np.linalg.norm(x_j)
                v.append(T/N)
    return np.max(v)

def find_largest_idx(h_n, index_set):
    found = False
    i = 0
    idx = 100000000
    sorted_indices = np.argsort(h_n)[::-1]
    if (len(index_set)==0):
        return sorted_indices[i]
    else:
        while(not found):
            idx = sorted_indices[i]
            if(np.in1d(idx, index_set)):
                i += 1
            else:
                found = True
        return idx

def OMP(A, y, N_active):
    r = y
    x_k = np.zeros(A.shape[1], dtype=complex)
    index_set = []
    for k in range(N_active):
        h_k = np.conj(A.T) @ r
        h_n = np.linalg.norm(h_k.reshape(h_k.shape[0], 1), axis=1)
        index_set.append(find_largest_idx(h_n, index_set))
        A_idx = A[:, index_set]
        A_pinv = np.linalg.pinv(A_idx)
        v = A_pinv @ y
        x_k[index_set] = v
        b_k = A @ x_k
        r = y - b_k
    return index_set

def MMV_OMP(A, Y, N_active):
    R = Y
    X_K = np.zeros((A.shape[1], Y.shape[1]), dtype=complex)
    index_set = []
    for k in range(N_active):
        H_K = np.conj(A.T) @ R
        h_n = np.linalg.norm(H_K, axis=1)
        index_set.append(find_largest_idx(h_n, index_set))
        A_idx = A[:, index_set]
        A_pinv = np.linalg.pinv(A_idx)
        V = A_pinv @ Y
        X_K[index_set] = V
        B_K = A @ X_K
        B = A_idx @ V
        R = Y - B_K
    return index_set


if __name__ == "__main__":
    K = 4
    A, a, column_idx = get_gaussian_pilots("./pilots/gauss_12_100_set1.npy", 100,K)


    B, b, c = get_ICBP_pilots("./pilots/ICBP_12_100.mat", 100, K)
    h = np.zeros(100)
    h[column_idx] = np.random.randn(K)
    y = np.dot(A, h)
    coherence = calculate_coherence(A)
    c_B = calculate_coherence(B)
    index_set = OMP(A, y, K)
    y = np.reshape(y, (12,1))
    mmv_index_set = MMV_OMP(A, y, K)

    print(np.sort(column_idx))
    print(np.sort(index_set))
    print(np.sort(mmv_index_set))

    H = np.zeros((K, 4))
    H = np.random.randn(K,4)
    Y = A[:, column_idx] @ H
    mmv_index_set = MMV_OMP(A,Y,K)
    print(np.sort(mmv_index_set))
