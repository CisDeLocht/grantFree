import numpy as np
from cell_functions import * #populate_cell, calculate_distance, simulate_path_loss_rayleigh

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 4                                                                     #in meters
    grid_size = 2*cell_radius + 1
    sparsity = 0.1                                                                      #N = +- (sparsity * grid_size^2) -> 100
    grid = np.zeros((grid_size, grid_size))
    freq = 2                                                                            #in GHz -> w = 15cm, spacing between antenna = 4*w
    SNR = 10                                                                            #in dB
    Lp = 3                                                                              #Pilot sequence length L << N -> 12
    M = 4                                                                               #Nr of antennas
    grid, indices, indices_list, N = populate_cell(grid, sparsity)

    distances2b, distance_matrix = calculate_distances(grid, indices, M, N)             #Distances2b for each antenna: dim = NxM
    H = simulate_path_loss_rayleigh(distances2b,distance_matrix, M, freq)               #H is channel matrix: dim = NxM
    A = generate_gaussian_pilots(Lp, N)                                                 #A is pilot sequences: dim = LxN
    No = simulate_noise(SNR, Lp, M)                                                     #No is noise matrix : dim =LxM
    Y = A @ H + No



    print(grid)
    print(indices_list)
    print(distances2b)
    print(distance_matrix)
    #print(channel_coeff)