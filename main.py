import numpy as np

from sim.detection_probability import *

from enum import Enum
import time
import multiprocessing

def worker(func, *args):
    return func(*args)

def collect_result(result, results1, results2, idx):
    results1[idx], results2[idx] = result

class Pilot(Enum):
    GAUSSIAN = 0
    ICBP = 1

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 500                                                                       #in meters
    N = 100
    K = 8                                                                               #N total users, K active users
    P = 1
    freq = 2                                                                            #in GHz
    SNR = 170                                                                           #in dB
    Lp = 12                                                                             #Pilot sequence length L << N -> 12
    M = 8                                                                               #Nr of antennas
    s = 10000
    # ------------ Detection Probability vs. Active users --------------
    users = np.arange(15) + 1
    P_g_m = [None]*len(users)
    P_g_o = [None]*len(users)
    P_I_m = [None]*len(users)
    P_I_o = [None]*len(users)
    IP_g_m = [None]*len(users)
    IP_g_o = [None]*len(users)
    IP_I_m = [None]*len(users)
    IP_I_o = [None]*len(users)
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)
    for i, k in enumerate(users):
        pool.apply_async(worker, (sim_detection_probability, N, k, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_m, IP_g_m,idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, k, M, P, Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_m, IP_I_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, k, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_o, IP_g_o, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, k, M, P, Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_o, IP_I_o, idx))
    pool.close()
    pool.join()
    print(IP_g_m)
    print(IP_g_o)
    print(IP_I_o)
    print(IP_I_m)
    print("Execution time for users: %s seconds" % (time.time()-start_time))
    plot_4detection_results(P_g_m, P_g_o, P_I_m, P_I_o, "MUSIC", "OMP", users, "Number of Active Users", " ")
    plot_detailed_reliability(IP_g_m, IP_g_o, IP_I_m, IP_I_o, "MUSIC", "OMP", users, "Number of Active Users", " ", x_log=False)
    # ------------ Detection Probability vs. Antenna's --------------
    antennas = np.arange(24)+1
    P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o = reset_lists(P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o, len(antennas))
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)
    for i, a in enumerate(antennas):
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, a, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_m, IP_g_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, a, P, Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_m, IP_I_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, a, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_o, IP_g_o, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, a, P, Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_o, IP_I_o, idx))
    pool.close()
    pool.join()

    print("Execution time for antennas: %s seconds" % (time.time() - start_time))
    #plot_4detection_results(P_g_m, P_g_o, P_I_m, P_I_o, "MUSIC", "OMP", antennas, "Number of Antennas", " ")
    plot_detailed_reliability(IP_g_m, IP_g_o, IP_I_m, IP_I_o, "MUSIC", "OMP", antennas, "Number of Antennas", " ", x_log=False)

    # ------------ Detection Probability vs. Pilot Length --------------
    pilot_lengths = np.asarray([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 32])
    P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o = reset_lists(P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o, len(pilot_lengths))
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)
    for i, l in enumerate(pilot_lengths):
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.GAUSSIAN, l, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_m, IP_g_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.ICBP, l, cell_radius, SNR, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_m, IP_I_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.GAUSSIAN, l, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_o, IP_g_o, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.ICBP, l, cell_radius, SNR, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_o, IP_I_o, idx))
    pool.close()
    pool.join()

    print("Execution time for pilots: %s seconds" % (time.time() - start_time))
    #plot_4detection_results(P_g_m, P_g_o, P_I_m, P_I_o, "MUSIC", "OMP", pilot_lengths, "Pilot Length", " ")
    plot_detailed_reliability(IP_g_m, IP_g_o, IP_I_m, IP_I_o, "MUSIC", "OMP", pilot_lengths, "Pilot Length", " ", x_log=False)

    # ------------ Detection Probability vs. Tx Power --------------

    SNR_list = np.asarray([1, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500,600, 700, 800])
    P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o = reset_lists(P_g_m, P_g_o, P_I_m, P_I_o, IP_g_m, IP_g_o, IP_I_m, IP_I_o, len(SNR_list))
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)
    for i, snr in enumerate(SNR_list):
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_m, IP_g_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "MUSIC", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_m, IP_I_m, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_g_o, IP_g_o, idx))
        pool.apply_async(worker,
                         (sim_detection_probability, N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "OMP", s),
                         callback=lambda res, idx=i: collect_result(res, P_I_o, IP_I_o, idx))
    pool.close()
    pool.join()

    print("Execution time for SNR: %s seconds" % (time.time() - start_time))
    #plot_4detection_results(P_g_m, P_g_o, P_I_m, P_I_o, "MUSIC", "OMP", SNR_list, "Transmit SNR", "[dB]")
    plot_detailed_reliability(IP_g_m, IP_g_o, IP_I_m, IP_I_o, "MUSIC", "OMP", SNR_list, "Transmit SNR", "[dB]", x_log=True)

