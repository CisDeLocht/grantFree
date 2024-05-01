import numpy as np
import scipy as sp

def populate_cell(grid, sparsity):
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    num_active = int(np.floor(grid_size ** 2 * sparsity))
    if (num_active == 0):
        raise Exception("Active devices is zero, please increase sparsity")

    indices = np.random.choice(grid_size ** 2, num_active, replace=False)
    i, j = np.unravel_index(indices, (grid_size, grid_size))
    grid[i, j] = 1
    grid[center, center] = -1
    indices = np.column_stack((i, j))
    indices_list = [tuple(index) for index in indices]

    return grid, indices, indices_list


def calculate_distances(grid, indices):
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    center_index = np.array([center, center])
    distances2b = np.asarray([np.linalg.norm(center_index - index) for index in indices])
    num_users = len(indices)
    distance_matrix = np.array([[np.linalg.norm(indices[i] - indices[j]) for j in range(num_users)] for i in range(num_users)])

    return distances2b, distance_matrix

def generate_correlated_shadow_fading(distance_matrix, sigma):
    eps = 1e-10
    SF_uncorrelated = np.random.normal(0, sigma, distance_matrix.shape[0])
    correlation_matrix = 16 * 2**(-distance_matrix/9)
    ret_val = sp.linalg.ldl(correlation_matrix)
    L = ret_val[0]
    D = ret_val[1]
    D[np.abs(D) < eps] = 0
    SF = L @ np.sqrt(D) @ SF_uncorrelated
    return SF
def simulate_path_loss_rayleigh(d2b, dm, f):
    UMC_PL = 36.7*np.log10(d2b) + 22.7 + 26*np.log10(f)
    SF = generate_correlated_shadow_fading(dm, 4)
    PL_dB = UMC_PL + SF                                                                                  #sigma = 4 according to UMC model in TS 36.814
    PL = 10**(-PL_dB/10)
    rayleigh_fading = np.random.normal(0, np.sqrt(0.5), size=d2b.shape[0]) + 1j * np.random.normal(0, np.sqrt(0.5), size=d2b.shape[0])
    h = np.multiply(np.sqrt(PL) ,rayleigh_fading)
    return h

def simulate_noise(snr, N):
    s_lin = 10 ** (snr / 10)
    n_std = np.sqrt(1 / (2 * s_lin))
    n = np.random.normal(0, n_std, size=N) + 1j * np.random.normal(0, n_std, size=N)
    return n
