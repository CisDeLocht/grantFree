from torch.utils.data import Dataset
import numpy as np
import torch
import os
from system_functions import *
import random

class GF_dataset(Dataset):
    def __init__(self, pilots: np.ndarray, K:int ,M:int, P:int, SNR:int,
                 cell_radius: int, f: int, dataset_size:int ,transform=None):
        self.data = []
        self.length = dataset_size
        self.numclasses = pilots.shape[1]

        grid_size = 2 * cell_radius + 1
        grid = np.zeros((grid_size, grid_size))
        for i in range(dataset_size):

            grid, indices = populate_cell(grid, K)

            distances2b, distance_matrix = calculate_distances(grid, indices, M, K)

            H = simulate_path_loss_rayleigh(distances2b, distance_matrix, M, K, P, f)

            No = simulate_noise(SNR, pilots.shape[0], M)

            active_idx = random.sample(range(pilots.shape[1]), K)
            A_active = pilots[:, active_idx[:K]]

            Y = A_active @ H + No
            self.data.append([Y, active_idx])

    def __getitem__(self, id):
        Y, active_idx = self.data[id]
        Y_real = Y.real
        Y_imag = Y.imag
        Y_stacked = np.stack((Y_real, Y_imag), axis=0)
        Y_tensor = torch.tensor(Y_stacked).float()

        multi_label_idx = np.zeros(self.numclasses, dtype=float)
        multi_label_idx[active_idx] = 1
        return Y_tensor, torch.tensor(multi_label_idx).float()


    def __len__(self):
        return self.length

    def getNumClasses(self):
        return self.numclasses

if __name__ == "__main__":
    cell_radius = 500  # in meters
    N = 100
    K = 8  # N total users, K active users
    P = 1
    freq = 2  # in GHz
    SNR = 170  # in dB
    Lp = 12  # Pilot sequence length L << N -> 12
    M = 8
    root = os.path.abspath("..")
    path = os.path.join(root, "pilots", "ICBP_" + str(Lp) + "_100.mat")
    A, _ = get_ICBP_pilots(path, N, K)

    dataset = GF_dataset(A, K, M, P, SNR, cell_radius, freq, 1000)
    dataset.__getitem__(5)
    print(dataset.getNumClasses())

