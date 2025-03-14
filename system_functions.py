import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.io as sio
from mytest.tests import *

def populate_cell_radius(radius, margin,K):
    x, y = generate_positions(radius, margin, K)
    pos = np.column_stack((x, y))
    return pos

def populate_cell(grid, K):
    grid_size = grid.shape[0]
    num_active = K
    if (num_active == 0):
        raise Exception("Active devices is zero, please increase sparsity")


    i = np.random.randint(0, grid_size, num_active)
    j = np.random.randint(0, grid_size, num_active)
    indices = np.column_stack((i, j))
    x, y = generate_positions((grid_size-1)/2, 10, num_active)
    pos = np.column_stack((x,y))
    return grid, indices


def calculate_distances_radius(indices, K):
    bs_height = 10
    center_index = np.array([0, 0])
    distances2b_horizontal = np.asarray([np.linalg.norm(center_index - index) for index in indices])
    distances2b = np.sqrt(distances2b_horizontal**2 + bs_height**2)                                                             #Pythagoras
    IU_distance_matrix = np.zeros((K,K))
    IU_distance_matrix[np.triu_indices(K,1)] = [np.linalg.norm(indices[i] - indices[j]) for i in range(K) for j in range(i+1, K)]      #Inter-User distances
    IU_distance_matrix += IU_distance_matrix.T
    return distances2b, IU_distance_matrix

def calculate_distances(grid, indices, M, K):
    bs_height = 10
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    center_index = np.array([center, center])
    distances2b_horizontal = np.asarray([np.linalg.norm(center_index - index) for index in indices])
    distances2b = np.sqrt(distances2b_horizontal**2 + bs_height**2)                                                             #Pythagoras
    IU_distance_matrix = np.zeros((K,K))
    IU_distance_matrix[np.triu_indices(K,1)] = [np.linalg.norm(indices[i] - indices[j]) for i in range(K) for j in range(i+1, K)]      #Inter-User distances
    IU_distance_matrix += IU_distance_matrix.T
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
def simulate_path_loss_rayleigh(d2b, dm, M, K, P, f):
    UMC_PL = 36.7*np.log10(d2b) + 22.7 + 26*np.log10(f)
    SF = generate_correlated_shadow_fading(dm, 4)
    PL_dB = UMC_PL + SF                                                                                  #sigma = 4 according to UMiC model in TS 36.814
    PL = 10**(-PL_dB/10)
    rayleigh_fading = np.random.normal(0, np.sqrt(0.5), size=(d2b.shape[0], M)) + 1j * np.random.normal(0, np.sqrt(0.5), size=(d2b.shape[0], M))
    power = P * PL.reshape(K,1)
    H = np.multiply(np.sqrt(P*PL.reshape(K,1)) , rayleigh_fading)
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
    #coherence = vars['muX'][0][0]
    A_ICBP_active = A_ICBP[:, active_idx[:K]]
    return A_ICBP, A_ICBP_active, #coherence

def simulate_noise(snr, L, M):
    s_lin = 10 ** (snr / 10)
    n_std = np.sqrt(1 / (2 * s_lin))
    n = np.random.normal(0, n_std, size=(L, M)) + 1j * np.random.normal(0, n_std, size=(L, M))
    return n


def generate_positions(radius, margin, num_points):
    # Define the range for the radius
    min_radius = radius - margin
    max_radius = radius + margin

    # Generate random radii and angles
    radii = np.random.uniform(min_radius, max_radius, num_points)
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Convert polar coordinates to Cartesian coordinates
    x_positions = radii * np.cos(angles)
    y_positions = radii * np.sin(angles)

    return np.rint(x_positions).astype(np.int32), np.rint(y_positions).astype(np.int32)


def plot_positions(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=10, c='blue', alpha=0.6)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('User Positions on a Circle')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    K = 8
    grid_size = 201

    populate_cell(np.zeros((grid_size, grid_size)), K)
    # Example usage
    radius = 100
    margin = 20
    num_points = 1000
    x, y = generate_positions(radius, margin, num_points)

    # Plot the positions
    plot_positions(x, y)
    _, indices = populate_cell(np.zeros((grid_size, grid_size)), 200)
    plot_positions(indices[:,0], indices[:,1])
    pos = populate_cell_radius(radius, margin, 200)
    plot_positions(pos[:,0], pos[:,1])

    idx_r = np.asarray([[0, 100], [50, 50], [-30, 80]])
    idx = np.asarray([[100, 200], [150, 150], [70, 180]])

    d2b_r = calculate_distances_radius(idx_r, 3)
    d2b = calculate_distances(grid=np.zeros([201, 201]), M=2, indices=idx, K=3)

    print(d2b_r)
    print(d2b)