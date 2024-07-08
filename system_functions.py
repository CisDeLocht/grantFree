import numpy as np
import scipy.linalg as linalg
import scipy.io as sio

def populate_cell(grid, K):
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    num_active = K
    if (num_active == 0):
        raise Exception("Active devices is zero, please increase sparsity")

    indices = np.random.choice(grid_size ** 2, num_active, replace=False)
    i, j = np.unravel_index(indices, (grid_size, grid_size))
    grid[i, j] = 1
    grid[center, center] = -1
    indices = np.column_stack((i, j))
    indices_list = [tuple(index) for index in indices]

    return grid, indices, indices_list, len(indices_list)


def calculate_distances(grid, indices, M, K):
    bs_height = 10
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    center_index = np.array([center, center])
    distances2b_horizontal = np.asarray([np.linalg.norm(center_index - index) for index in indices])
    distances2b = np.sqrt(distances2b_horizontal**2 + bs_height**2)                                                     #Pythagoras
    IU_distance_matrix = np.array([[np.linalg.norm(indices[i] - indices[j]) for j in range(K)] for i in range(K)])      #Inter-User distances

    return distances2b, IU_distance_matrix

def generate_correlated_shadow_fading(distance_matrix, sigma):
    eps = 1e-10
    SF_uncorrelated = np.random.normal(0, sigma, distance_matrix.shape[0])
    correlation_matrix = 16 * 2**(-distance_matrix/9)
    ret_val = linalg.ldl(correlation_matrix)
    L = ret_val[0]
    D = ret_val[1]
    D[np.abs(D) < eps] = 0
    SF = L @ np.sqrt(D) @ SF_uncorrelated
    return SF
def simulate_path_loss_rayleigh(d2b, dm, M, K, f):
    UMC_PL = 36.7*np.log10(d2b) + 22.7 + 26*np.log10(f)
    SF = generate_correlated_shadow_fading(dm, 4)
    PL_dB = UMC_PL + SF                                                                                  #sigma = 4 according to UMC model in TS 36.814
    PL = 10**(-PL_dB/10)
    rayleigh_fading = np.random.normal(0, np.sqrt(0.5), size=(d2b.shape[0], M)) + 1j * np.random.normal(0, np.sqrt(0.5), size=(d2b.shape[0], M))
    H = np.multiply(np.sqrt(PL.reshape(K,1)) , rayleigh_fading)
    return H

def generate_gaussian_pilots(L, N, K):
    std = np.sqrt(1/(2*L))
    pilots = np.random.normal(0, std, size=(L,N)) + 1j * np.random.normal(0, std, size=(L,N))
    norms = np.linalg.norm(pilots, axis=0)
    a_n = pilots/norms
    column_idx = np.random.randint(N, size=K)
    a_n_active = a_n[:, column_idx[:K]]
    return a_n, a_n_active, column_idx

def get_gaussian_pilots(filename, N, K):
    a_n = np.load(filename)
    active_idx = np.random.randint(N,size=K)
    a_n_active = a_n[:, active_idx[:K]]
    return a_n, a_n_active, active_idx

def get_ICBP_pilots(filename, N, K):
    active_idx = np.random.randint(N, size=K)
    vars = sio.loadmat(filename)
    A_ICBP = vars['X']
    coherence = vars['muX'][0][0]
    A_ICBP_active = A_ICBP[:, active_idx[:K]]
    return A_ICBP, A_ICBP_active, coherence

def simulate_noise(snr, L, M):
    s_lin = 10 ** (snr / 10)
    n_std = np.sqrt(1 / (2 * s_lin))
    n = np.random.normal(0, n_std, size=(L, M)) + 1j * np.random.normal(0, n_std, size=(L, M))
    return n
