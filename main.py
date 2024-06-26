import numpy as np

from system_functions import *
from subspace_functions import *
from tests import *

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 4                                                                     #in meters
    grid_size = 2*cell_radius + 1
    sparsity = 0.1                                                                      #N = +- (sparsity * grid_size^2) -> 100
    grid = np.zeros((grid_size, grid_size))
    K = 100                                                                             #K total users, N active users
    freq = 2                                                                            #in GHz
    SNR = 10                                                                            #in dB
    Lp = 12                                                                             #Pilot sequence length L << N -> 12
    M = 12                                                                              #Nr of antennas
    ICBP_file = "pilots.mat"
    # ------------ System setup --------------

    grid, indices, indices_list, N = populate_cell(grid, sparsity)                      #N is active users
    A_g, A_g_active, active_idx = generate_gaussian_pilots(Lp, N, K)                    #Should be fixed over all simulation runs
    A_ICBP, A_ICBP_active, coherence = get_ICBP_pilots(ICBP_file, active_idx)                          # A_x is pilot sequences: dim = LxN

    distances2b, distance_matrix = calculate_distances(grid, indices, M, N)             #Distances2b for each antenna: dim = NxM
    H = simulate_path_loss_rayleigh(distances2b,distance_matrix, M, N, freq)               #H is channel matrix: dim = NxM
    No = simulate_noise(SNR, Lp, M)                                                     #No is noise matrix : dim =LxM
    Y_g = A_g_active @ H + No
    #Y_ICBP = A_ICBP_active @ H + No

    # ------------ Subspace method --------------

    sigs_g, sigs_idx_g = music(Y_g, M, N, A_g, False)
    #sigs_I, sigs_idx_I = music(Y_ICBP, M, N, A_ICBP, False)

    TEST_signature_match(active_idx, sigs_idx_g, "MUSIC", "Gaussian")
    #TEST_signature_match(active_idx, sigs_idx_I, "MUSIC", "ICBP")
    print(active_idx)