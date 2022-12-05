import sys; sys.path.append('../../')
import numpy as np
from gnse.config import FTFREQ, FT, IFT
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as col


def fetch_data(f_name):
    dat = np.load(f_name)
    return dat['z'], dat['t'], dat['w'], dat['atz'], dat['utz']


def main_a():
    # -- READ IN DATA
    f_name = ['../numExp04_NLPM-750/NLPM-750_offset_400_N=0.55.npz', '../numExp04_NLPM-750/NLPM-750_utz=0_offset_400_N=0.55.npz']

    z_1, t_1, w_1, atz_1, utz_1 = fetch_data(f_name[0])
    z_2, t_2, w_2, atz_2, utz_2 = fetch_data(f_name[1])
    z_1/=1e6; z_2/=1e6

    I_ = []

    I_l = []
    I_r = []

    for i in range(z_1.size):
        at = atz_1[i]
        aw = np.where((w_1 > 0.025) & (w_1 < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_.append(It)

        at_r = np.where(t_1 < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r.append(np.sum(t_1 * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t_1 > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l.append(np.sum(t_1 * It_l_t) / np.sum(It_l_t))

    I_l_1 = []
    I_r_1 = []
    I_ /= np.max(I_[0])


    for i in range(z_1.size):
        at = atz_2[i]
        aw = np.where((w_1 > 0.025) & (w_1 < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2

        at_r = np.where(t_1 < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_1.append(np.sum(t_1 * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t_1 > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_1.append(np.sum(t_1 * It_l_t) / np.sum(It_l_t))


    plt.plot(I_r, z_1, lw=1.5, ls='--', color='black')
    plt.plot(I_l, z_1, lw=1.5, ls='--', color='black')
    plt.plot(I_r_1, z_1, lw=1.5, ls='--', color='white')
    plt.plot(I_l_1, z_1, lw=1.5, ls='--', color='white')
    cmap = mpl.cm.get_cmap("jet")
    mpl.rcParams['axes.formatter.useoffset'] = False
    plt.pcolormesh(t_1, z_1, I_, cmap=cmap)
    plt.xlim(-2500, 2500)
    plt.xlabel("Time delay $t$ (ps)")
    plt.ylabel(r"Propagation distance $z$ (m)")
    #plt.title('Solitons $_1$ and $A_2$ trajactories')
    plt.colorbar()
    plt.savefig('trajectories', dpi=600)

if __name__ == "__main__":
    main_a()
