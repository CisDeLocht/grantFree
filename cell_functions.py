import numpy as np


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


def calculate_distance(grid, indices):
    grid_size = grid.shape[0]
    center = int((grid_size - 1) / 2)
    center_index = np.array([center, center])
    distances = [np.linalg.norm(center_index - index) for index in indices]

    return distances

#REMEMBER: shadow fading can be implemented by having a Gauss. addition to the dB value of PL_CI
def simulate_path_loss_rayleigh(d, f, n):
    FSPL = 20 * np.log10(d) + 20 * np.log10(f) - 147.55
    PL_CI = FSPL + 10 * n * np.log10(d)
    PL_lin = 1/(10**(PL_CI/10))         #Square root or not?
    rayleigh_fading = np.random.normal(0, np.sqrt(0.5)) + 1j * np.random.normal(0, np.sqrt(0.5))
    PL_with_fading = PL_lin * rayleigh_fading

    return PL_with_fading
