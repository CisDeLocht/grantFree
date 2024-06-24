import numpy as np

from system_functions import *
from subspace_functions import *

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 4                                                                     #in meters
    grid_size = 2*cell_radius + 1
    sparsity = 0.1                                                                      #N = +- (sparsity * grid_size^2) -> 100
    grid = np.zeros((grid_size, grid_size))
    freq = 2                                                                            #in GHz
    SNR = 10                                                                            #in dB
    Lp = 3                                                                              #Pilot sequence length L << N -> 12
    M = 4                                                                               #Nr of antennas
    ICBP_file = "pilots.mat"
    # ------------ System setup --------------

    grid, indices, indices_list, N = populate_cell(grid, sparsity)
    A_g = generate_gaussian_pilots(Lp, N)                                               #Should be fixed over all simulation runs
    A_ICBP, coherence = get_ICBP_pilots(ICBP_file)                                      # A_x is pilot sequences: dim = LxN

    distances2b, distance_matrix = calculate_distances(grid, indices, M, N)             #Distances2b for each antenna: dim = NxM
    H = simulate_path_loss_rayleigh(distances2b,distance_matrix, M, freq)               #H is channel matrix: dim = NxM
    No = simulate_noise(SNR, Lp, M)                                                     #No is noise matrix : dim =LxM
    Y_g = A_g @ H + No
    #Y_ICBP = A_ICBP @ H + No

    music(Y_g, M, A_g)
