from sim.detection_probability import *

from enum import Enum

class Pilot(Enum):
    GAUSSIAN = 0
    ICBP = 1

if __name__ == '__main__':
    #------------ Simulation Parameters --------------
    cell_radius = 500                                                                     #in meters
    N = 100
    K = 5                                                                               #N total users, K active users
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
    for k in users:
        p_g_m = sim_detection_probability(N, k, M, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "MUSIC", s)
        p_I_m = sim_detection_probability(N, k, M, Pilot.ICBP, Lp, cell_radius, SNR, "MUSIC", s)
        p_g_o = sim_detection_probability(N, k, M, Pilot.GAUSSIAN, Lp, cell_radius, SNR, "OMP", s)
        p_I_o = sim_detection_probability(N, k, M, Pilot.ICBP, Lp, cell_radius, SNR, "OMP", s)

        P_g_m.append(p_g_m)
        P_g_o.append(p_g_o)
        P_I_m.append(p_I_m)
        P_I_o.append(p_I_o)

    plot_detection_results(P_g_m, "MUSIC", Pilot.GAUSSIAN, users, "Number of Active Users", " ")
    plot_detection_results(P_g_o, "OMP", Pilot.GAUSSIAN, users, "Number of Active Users", " ")
    plot_detection_results(P_I_m, "MUSIC", Pilot.ICBP, users, "Number of Active Users", " ")
    plot_detection_results(P_I_o, "OMP", Pilot.ICBP, users, "Number of Active Users", " ")



    # ------------ Subspace method --------------

