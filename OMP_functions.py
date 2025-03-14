import numpy as np
from system_functions import *
import math
import scipy.special
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

    n = 100
    max_k = 16

    total_combinations = sum(scipy.special.comb(n,k) for k in range(1, max_k + 1))
    print(total_combinations)
    print(scipy.special.comb(100, 8))
