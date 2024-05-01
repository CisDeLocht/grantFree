import numpy as np
from cell_functions import * #populate_cell, calculate_distance, simulate_path_loss_rayleigh

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 4                                                         #in meters
    grid_size = 2*cell_radius + 1
    sparsity = 0.1
    grid = np.zeros((grid_size, grid_size))
    freq = 2                                                                #in GHz
    SNR = 10                                                                #in dB

    grid, indices, indices_list = populate_cell(grid, sparsity)

    distances2b, distance_matrix = calculate_distances(grid, indices)
    channel_coeff = simulate_path_loss_rayleigh(distances2b,distance_matrix, freq)
    noise_vector = simulate_noise(SNR, len(distances2b))

    #TODO create pilots somehow

    print(grid)
    print(indices_list)
    print(distances2b)
    print(distance_matrix)
    #print(channel_coeff)