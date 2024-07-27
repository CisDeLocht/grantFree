from system_functions import *
from OMP_functions import *
from subspace_functions import *
import os
import time
from enum import Enum

class Pilot(Enum):
    GAUSSIAN = 0
    ICBP = 1
def sim_detection_probability(N, K, M , P ,p_type, p_length, cell_radius, SNR, method, s) -> tuple:
    """
    Function to simulate detection probability
    :Parameters:
    N (int) : number of total users
    K (int) : number of active users
    M (int) : number of receive antennas
    P (int) : device transmit power
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
        path = os.path.join(root, "grantFree/pilots", "gauss_" + str(p_length)+ "_100.npy")
        A, _, _ = get_gaussian_pilots(path, N, K)
    else:
        path = os.path.join(root, "grantFree/pilots", "ICBP_" +str(p_length)+"_100.mat")
        A, _ = get_ICBP_pilots(path, N, K)

    grid_size = 2*cell_radius+1
    grid = np.zeros((grid_size,grid_size))
    for i in range(s):
        grid, indices = populate_cell(grid, K)

        distances2b, distance_matrix = calculate_distances(grid, indices, M, K)

        H = simulate_path_loss_rayleigh(distances2b, distance_matrix, M, K, P, f)

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
    correct_detection = detection/s *100
    incorrect_detection = 100 - correct_detection
    return correct_detection, incorrect_detection

def plot_detection_results(probabilities, method, p_type, x, x_name, x_unit):
    plt.plot(x, probabilities)
    plt.xlabel(x_name+x_unit)
    plt.ylabel("Detection Probability [%]")
    plt.title("Detection Probability vs. " + x_name + "[" + method + "] with " + p_type.name + " pilots")
    plt.show()

def plot_4detection_results(prob1, prob2, prob1_icbp, prob2_icbp, method1, method2, x, x_name, x_unit):
    plt.plot(x, prob1, "-b", label=method1 + " [Gaussian]")
    plt.plot(x, prob2, "-r", label=method2 + " [Gaussian]")
    plt.plot(x, prob1_icbp, "--b", label=method1 + " [ICBP]")
    plt.plot(x, prob2_icbp, "--r", label=method2 + " [ICBP]")
    plt.xlabel(x_name + x_unit)
    plt.ylabel("Probability [%]")
    plt.title("Correct Detection Probability vs. " + x_name)
    plt.legend(loc="upper right")
    plt.show()

def plot_detailed_reliability(prob1, prob2, prob1_icbp, prob2_icbp, method1, method2, x, x_name, x_unit):
    plt.plot(x, prob1, "-b", label=method1 + " [Gaussian]")
    plt.plot(x, prob2, "-r", label=method2 + " [Gaussian]")
    plt.plot(x, prob1_icbp, "--b", label=method1 + " [ICBP]")
    plt.plot(x, prob2_icbp, "--r", label=method2 + " [ICBP]")
    plt.xlabel(x_name + x_unit)
    plt.ylabel("Probability [%]")
    plt.yscale("log")
    plt.ylim(0.001, 102)
    plt.title("Detailed Incorrect Detection Probability vs. " + x_name)
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    cell_radius = 500  # in meters
    N = 100
    K = 8  # N total users, K active users
    P = 1
    freq = 2  # in GHz
    SNR = 1000  # in dB
    Lp = 12  # Pilot sequence length L << N -> 12
    M = 8  # Nr of antennas
    s = 1

    p_g_m, ip_g_m = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s)
    p_g_m, ip_g_m = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s)
