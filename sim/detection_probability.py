from system_functions import *
from OMP_functions import *
from subspace_functions import *
import os

def sim_detection_probability(N, K, M ,p_type, p_length, cell_radius, SNR, method, s) -> float:
    """
    Function to simulate detection probability
    :Parameters:
    N (int) : number of total users
    K (int) : number of active users
    M (int) : number of receive antennas
    p_type (int) : type of pilot
    p_length (int) : length of pilot
    cell_radius (int) : radius of cell
    SNR (int) : transmit power of user devices
    method (string) : method name
    s (int) : number of simulation runs

    :return: probability of correct detection averaged over s simulations
    """
    f = 2
    detection = 0
    root = os.path.abspath("..")
    if (p_type.value == 0):
        path = os.path.join(root, "grantFree/pilots", "gauss_12_100_set1.npy")
        A, _, _ = get_gaussian_pilots(path, N, K)
    else:
        path = os.path.join(root, "grantFree/pilots", "ICBP_12_100.mat")
        A, _, _ = get_ICBP_pilots(path, N, K)

    grid_size = 2*cell_radius+1
    grid = np.zeros((grid_size,grid_size))
    for i in range(s):
        grid, indices, indices_list, check = populate_cell(grid, K)
        distances2b, distance_matrix = calculate_distances(grid, indices, M, K)
        H = simulate_path_loss_rayleigh(distances2b, distance_matrix, M, K, f)
        No = simulate_noise(SNR, p_length, M)

        active_idx = np.random.randint(N, size=K)
        A_active = A[:, active_idx[:K]]

        Y = A_active @ H + No

        if(method == "OMP"):
            estimate_idx = MMV_OMP(A, Y, K)
        elif(method == "MUSIC"):
            _ , estimate_idx = music(Y, M, K, A, False)
        else:
            raise Exception(f"Method name: {method} not recognized!")

        if(len(np.setdiff1d(active_idx, estimate_idx)) == 0):
            detection += 1

    return detection/s * 100

def plot_detection_results(probabilities, method, p_type, x, x_name, x_unit):
    plt.plot(x, probabilities)
    plt.xlabel(x_name+x_unit)
    plt.ylabel("Detection Probability [%]")
    plt.title("Detection Probability vs. " + x_name + "[" + method + "] with " + p_type.name + " pilots")
    plt.show()