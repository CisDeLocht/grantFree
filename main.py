import numpy as np
from cell_functions import populate_cell, calculate_distance, simulate_path_loss_rayleigh

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 4                                                         #in meters
    grid_size = 2*cell_radius + 1
    sparsity = 0.1
    grid = np.zeros((grid_size, grid_size))
    freq = 3.3e6                                                            #in Hertz
    PLE = 2.5                                                               #Path Loss Exponent [2 - 4]
    SNR = 10                                                                #in dB

    grid, indices, indices_list = populate_cell(grid, sparsity)

    distances = calculate_distance(grid, indices)
    channel_coeff = [simulate_path_loss_rayleigh(d, freq, PLE) for d in distances]


    print(grid)
    print(indices_list)
    print(distances)
    print(channel_coeff)