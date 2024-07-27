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
    K = 8                                                                                 #N total users, K active users
    P = 1
    freq = 2                                                                            #in GHz
    SNR = 1000                                                                           #in dB
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
        # p_g_m, ip_g_m = sim_detection_probability(N, k, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s)
        # p_I_m, ip_I_m = sim_detection_probability(N, k, M,P,Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s)
        # p_g_o, ip_g_o = sim_detection_probability(N, k, M, P,Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s)
        # p_I_o, ip_I_o = sim_detection_probability(N, k, M, P,Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s)
        #
        # P_g_m[i] = p_g_m
        # P_g_o[i] = p_g_o
        # P_I_m[i] = p_I_m
        # P_I_o[i] = p_I_o
        # IP_g_m[i] = ip_g_m
        # IP_g_o[i] = ip_g_o
        # IP_I_m[i] = ip_I_m
        #IP_I_o[i] = ip_I_o

    print("Execution time: %s seconds" % (time.time()-start_time))
    plot_4detection_results(P_g_m, P_g_o, P_I_m, P_I_o, "MUSIC", "OMP", users, "Number of Active Users", " ")
    plot_detailed_reliability(IP_g_m, IP_g_o, IP_I_m, IP_I_o, "MUSIC", "OMP", users, "Number of Active Users", " ")
    # ------------ Detection Probability vs. Antenna's --------------
    # P_g_m = []
    # P_g_o = []
    # P_I_m = []
    # P_I_o = []
    # antennas = np.asarray([4, 8, 12, 16, 32, 48, 64])
    # for a in antennas:
    #     p_g_m = sim_detection_probability(N, K, a, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s)
    #     p_I_m = sim_detection_probability(N, K, a, P,Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s)
    #     p_g_o = sim_detection_probability(N, K, a, P,Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s)
    #     p_I_o = sim_detection_probability(N, K, a, P,Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s)
    #
    #     P_g_m.append(p_g_m)
    #     P_g_o.append(p_g_o)
    #     P_I_m.append(p_I_m)
    #     P_I_o.append(p_I_o)
    #
    # plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, antennas, "Number of Antenna's", " ")
    # plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, antennas, "Number of Antenna's", " ")
    # plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, antennas, "Number of Antenna's", " ")
    # plot_detection_results(P_I_o, "OMP", Pilot.ICBP, antennas, "Number of Antenna's", " ")
    #
    # plot_2detection_results(P_g_m, P_g_o, "MUSIC", "OMP", Pilot.GAUSSIAN, antennas, "Number of Antenna's", " ")
    # plot_2detection_results(P_I_m, P_I_o, "MUSIC", "OMP", Pilot.ICBP, antennas, "Number of Antenna's", " ")

    # ------------ Detection Probability vs. Pilot Length --------------
    # P_g_m = []
    # P_g_o = []
    # P_I_m = []
    # P_I_o = []
    # pilot_lengths = np.asarray([6, 12, 18, 24, 32, 48, 64])
    # for l in pilot_lengths:
    #     p_g_m = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, l, cell_radius, SNR, "MUSIC", s)
    #     p_I_m = sim_detection_probability(N, K, M, P, Pilot.ICBP, l, cell_radius, SNR, "MUSIC", s)
    #     p_g_o = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, l, cell_radius, SNR, "OMP", s)
    #     p_I_o = sim_detection_probability(N, K, M, P, Pilot.ICBP, l, cell_radius, SNR, "OMP", s)
    #
    #     P_g_m.append(p_g_m)
    #     P_g_o.append(p_g_o)
    #     P_I_m.append(p_I_m)
    #     P_I_o.append(p_I_o)
    #
    # plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, pilot_lengths, "Pilot Length", " ")
    # plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, pilot_lengths, "Pilot Length", " ")
    # plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, pilot_lengths, "Pilot Length", " ")
    # plot_detection_results(P_I_o, "OMP", Pilot.ICBP, pilot_lengths, "Pilot Length", " ")
    #
    # plot_2detection_results(P_g_m, P_g_o, "MUSIC", "OMP", Pilot.GAUSSIAN, pilot_lengths, "Pilot Length", " ")
    # plot_2detection_results(P_I_m, P_I_o, "MUSIC", "OMP", Pilot.ICBP, pilot_lengths, "Pilot Length", " ")
    # ------------ Detection Probability vs. Tx Power --------------
    # P_g_m = []
    # P_g_o = []
    # P_I_m = []
    # P_I_o = []
    # SNR = np.asarray([1, 10, 50, 100, 200, 400, 800, 1000])
    # for snr in SNR:
    #     p_g_m = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "MUSIC", s)
    #     p_I_m = sim_detection_probability(N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "MUSIC", s)
    #     p_g_o = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "OMP", s)
    #     p_I_o = sim_detection_probability(N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "OMP", s)
    #
    #     P_g_m.append(p_g_m)
    #     P_g_o.append(p_g_o)
    #     P_I_m.append(p_I_m)
    #     P_I_o.append(p_I_o)
    #
    # plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, SNR, "SNR", "")
    # plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, SNR, "SNR", "")
    # plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, SNR, "SNR", "")
    # plot_detection_results(P_I_o, "OMP", Pilot.ICBP, SNR, "SNR", "")
    #
    # plot_2detection_results(P_g_m, P_g_o, "MUSIC", "OMP", Pilot.GAUSSIAN, SNR, "SNR", "")
    # plot_2detection_results(P_I_m, P_I_o, "MUSIC", "OMP", Pilot.ICBP, SNR, "SNR", "")
