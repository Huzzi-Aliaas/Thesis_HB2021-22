import sys; sys.path.append('../../')
import numpy as np
from gnse.config import FTFREQ, FT, IFT
import matplotlib.pyplot as plt

def fetch_data(f_name):
    dat = np.load(f_name)
    return dat['z'], dat['t'], dat['w'], dat['atz'], dat['utz']

def figure_1d(t, z, It_1, It_2, It_3, It_4, It_5, It_6, It_7, It_8, It_9, tlim=None, oName=None):


    f, ((ax1, ax2,ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)

    ax1.pcolormesh(t, z, It_1)
    ax1.set_xlim(-tlim, tlim)
    ax1.set_title('N=0.40')

    ax2.pcolormesh(t, z, It_2)
    ax2.set_xlim(-tlim, tlim)
    ax2.set_title('N=0.45')

    ax3.pcolormesh(t, z, It_3)
    ax3.set_xlim(-tlim, tlim)
    ax3.set_title('N=0.50')

    ax4.pcolormesh(t, z, It_4)
    ax4.set_xlim(-tlim, tlim)
    ax4.set_title('N=0.55')

    ax5.pcolormesh(t, z, It_5)
    ax5.set_xlim(-tlim, tlim)
    ax5.set_title('N=0.60')

    ax6.pcolormesh(t, z, It_6)
    ax6.set_xlim(-tlim, tlim)
    ax6.set_title('N=0.65')

    ax7.pcolormesh(t, z, It_7)
    ax7.set_xlim(-tlim, tlim)
    ax7.set_title('N=0.70')

    ax8.pcolormesh(t, z, It_8)
    ax8.set_xlim(-tlim, tlim)
    ax8.set_title('N=0.75')

    ax9.pcolormesh(t, z, It_9)
    ax9.set_xlim(-tlim, tlim)
    ax9.set_title('N=0.80')

    f.text(0.5, 0.005, "Time delay $t$ (fs)", ha='center')
    f.text(0.04, 0.5, 'Propagation distance $z$ ($\mu$m)', va='center', rotation='vertical')

    if oName:
        plt.savefig(oName,format='png',dpi=600, bbox_inches="tight")
    else:
        plt.show()


def main_a():

    x = 0.4
    # -- READ IN DATA
    f_name_1 = '../numExp04_NLPM-750/NLPM-750_N=0.40.npz'
    z, t, w, atz_1, utz_1 = fetch_data(f_name_1)

    aw_1 = FT(atz_1)
    aw_1 = np.where(w > 0.03, 0, aw_1)
    aw_1 = np.where(w < -0.03, 0, aw_1)
    at_1 = IFT(aw_1)

    It_1 = np.abs(at_1) ** 2
    It_max_1 = np.max(It_1)
    It_1 = np.where(It_1 > x * It_max_1, It_1, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_2 = '../numExp04_NLPM-750/NLPM-750_N=0.45.npz'
    z, t, w, atz_2, utz_2 = fetch_data(f_name_2)

    aw_2 = FT(atz_2)
    aw_2 = np.where(w > 0.03, 0, aw_2)
    aw_2 = np.where(w < -0.03, 0, aw_2)
    at_2 = IFT(aw_2)

    It_2 = np.abs(at_2) ** 2
    It_max_2 = np.max(It_2)
    It_2 = np.where(It_2 > x * It_max_2, It_2, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_3 = '../numExp04_NLPM-750/NLPM-750_N=0.50.npz'
    z, t, w, atz_3, utz_3 = fetch_data(f_name_3)

    aw_3 = FT(atz_3)
    aw_3 = np.where(w > 0.03, 0, aw_3)
    aw_3 = np.where(w < -0.03, 0, aw_3)
    at_3 = IFT(aw_3)

    It_3 = np.abs(at_3) ** 2
    It_max_3 = np.max(It_3)
    It_3 = np.where(It_3 > x * It_max_3, It_3, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_4 = '../numExp04_NLPM-750/NLPM-750_N=0.55.npz'
    z, t, w, atz_4, utz_4 = fetch_data(f_name_4)

    aw_4 = FT(atz_4)
    aw_4 = np.where(w > 0.03, 0, aw_4)
    aw_4 = np.where(w < -0.03, 0, aw_4)
    at_4 = IFT(aw_4)

    It_4 = np.abs(at_4) ** 2
    It_max_4 = np.max(It_4)
    It_4 = np.where(It_4 > x * It_max_4, It_4, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_5 = '../numExp04_NLPM-750/NLPM-750_N=0.60.npz'
    z, t, w, atz_5, utz_5 = fetch_data(f_name_5)

    aw_5 = FT(atz_5)
    aw_5 = np.where(w > 0.03, 0, aw_5)
    aw_5 = np.where(w < -0.03, 0, aw_5)
    at_5 = IFT(aw_5)

    It_5 = np.abs(at_5) ** 2
    It_max_5 = np.max(It_5)
    It_5 = np.where(It_5 > x * It_max_5, It_5, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_6 = '../numExp04_NLPM-750/NLPM-750_N=0.65.npz'
    z, t, w, atz_6, utz_6 = fetch_data(f_name_6)

    aw_6 = FT(atz_6)
    aw_6 = np.where(w > 0.03, 0, aw_6)
    aw_6 = np.where(w < -0.03, 0, aw_6)
    at_6 = IFT(aw_6)

    It_6 = np.abs(at_6) ** 2
    It_max_6 = np.max(It_6)
    It_6 = np.where(It_6 > x * It_max_6, It_6, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_7 = '../numExp04_NLPM-750/NLPM-750_N=0.70.npz'
    z, t, w, atz_7, utz_7 = fetch_data(f_name_7)

    aw_7 = FT(atz_7)
    aw_7 = np.where(w > 0.03, 0, aw_7)
    aw_7 = np.where(w < -0.03, 0, aw_7)
    at_7 = IFT(aw_7)

    It_7 = np.abs(at_7) ** 2
    It_max_7 = np.max(It_7)
    It_7 = np.where(It_7 > x * It_max_7, It_7, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_8 = '../numExp04_NLPM-750/NLPM-750_N=0.75.npz'
    z, t, w, atz_8, utz_8 = fetch_data(f_name_8)

    aw_8 = FT(atz_8)
    aw_8 = np.where(w > 0.03, 0, aw_8)
    aw_8 = np.where(w < -0.03, 0, aw_8)
    at_8 = IFT(aw_8)

    It_8 = np.abs(at_8) ** 2
    It_max_8 = np.max(It_8)
    It_8 = np.where(It_8 > x * It_max_8, It_8, 0)

    ###################################################################################

    # -- READ IN DATA
    f_name_9 = '../numExp04_NLPM-750/NLPM-750_N=0.80.npz'
    z, t, w, atz_9, utz_9 = fetch_data(f_name_9)

    aw_9 = FT(atz_9)
    aw_9 = np.where(w > 0.03, 0, aw_9)
    aw_9 = np.where(w < -0.03, 0, aw_9)
    at_9 = IFT(aw_9)

    It_9 = np.abs(at_9) ** 2
    It_max_9 = np.max(It_9)
    It_9 = np.where(It_9 > x * It_max_9, It_9, 0)

    tlim = 5500
    figure_1d(t, z, It_1, It_2, It_3, It_4, It_5,It_6, It_7, It_8, It_9, tlim=tlim, oName='fig_1.png')

if __name__ == "__main__":
    main_a()