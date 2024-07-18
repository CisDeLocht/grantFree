from sim.detection_probability import *

from enum import Enum

class Pilot(Enum):
    GAUSSIAN = 0
    ICBP = 1

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 500                                                                     #in meters
    N = 100
    K = 8                                                                               #N total users, K active users
    P = 1
    freq = 2                                                                            #in GHz
    SNR = 1000                                                                           #in dB
    Lp = 12                                                                             #Pilot sequence length L << N -> 12
    M = 8                                                                               #Nr of antennas
    s = 100
    # ------------ Detection Probability vs. Active users --------------
    P_g_m = []
    P_g_o = []
    P_I_m = []
    P_I_o = []
    users = np.arange(15) + 1
    # for k in users:
    #     p_g_m = sim_detection_probability(N, k, M, P, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s)
    #     p_I_m = sim_detection_probability(N, k, M,P,Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s)
    #     p_g_o = sim_detection_probability(N, k, M, P,Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s)
    #     p_I_o = sim_detection_probability(N, k, M, P,Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s)
    #
    #     P_g_m.append(p_g_m)
    #     P_g_o.append(p_g_o)
    #     P_I_m.append(p_I_m)
    #     P_I_o.append(p_I_o)
    #
    # plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, users, "Number of Active Users", " ")
    # plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, users, "Number of Active Users", " ")
    # plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, users, "Number of Active Users", " ")
    # plot_detection_results(P_I_o, "OMP", Pilot.ICBP, users, "Number of Active Users", " ")
    #
    # plot_2detection_results(P_g_m, P_g_o, "MUSIC", "OMP", Pilot.GAUSSIAN, users, "Number of Active Users", " ")
    # plot_2detection_results(P_I_m, P_I_o, "MUSIC", "OMP", Pilot.ICBP, users, "Number of Active Users", " ")

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
    P_g_m = []
    P_g_o = []
    P_I_m = []
    P_I_o = []
    SNR = np.asarray([1, 10, 50, 100, 200, 400, 800, 1000])
    for snr in SNR:
        p_g_m = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "MUSIC", s)
        p_I_m = sim_detection_probability(N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "MUSIC", s)
        p_g_o = sim_detection_probability(N, K, M, P, Pilot.GAUSSIAN, Lp, cell_radius, snr, "OMP", s)
        p_I_o = sim_detection_probability(N, K, M, P, Pilot.ICBP, Lp, cell_radius, snr, "OMP", s)

        P_g_m.append(p_g_m)
        P_g_o.append(p_g_o)
        P_I_m.append(p_I_m)
        P_I_o.append(p_I_o)

    plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, SNR, "SNR", "")
    plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, SNR, "SNR", "")
    plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, SNR, "SNR", "")
    plot_detection_results(P_I_o, "OMP", Pilot.ICBP, SNR, "SNR", "")

    plot_2detection_results(P_g_m, P_g_o, "MUSIC", "OMP", Pilot.GAUSSIAN, SNR, "SNR", "")
    plot_2detection_results(P_I_m, P_I_o, "MUSIC", "OMP", Pilot.ICBP, SNR, "SNR", "")
