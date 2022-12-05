import sys; sys.path.append('../../')
import numpy as np
from gnse.config import FTFREQ, FT, IFT
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as col


def figure_1d(t, z, It_1, It_2, It_3, It_4, It_5, It_6, It_7, It_8, It_9,I_r_1, I_r_2,I_r_3,I_r_4,I_r_5,I_r_6,I_r_7,I_r_8,I_r_9,
              I_l_1,I_l_2,I_l_3,I_l_4,I_l_5,I_l_6,I_l_7,I_l_8,I_l_9,tlim=None, oName=None):

    cmap = mpl.cm.get_cmap("jet")
    mpl.rcParams['axes.formatter.useoffset'] = False

    f, ((ax1, ax2,ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)

    ax1.pcolormesh(t, z, It_1, cmap=cmap)
    ax1.set_xlim(-tlim, tlim)
    ax1.plot(I_r_1, z, lw=0.5, ls='--', color='black')
    ax1.plot(I_l_1, z, lw=0.5, ls='--', color='black')
    ax1.annotate(r'$\xi $=0.3 ns', xy=(-5,0.7), color='white')
    ax1.set_ylabel('Propagation \n distance $z$ \n (km)')

    ax2.pcolormesh(t, z, It_2, cmap=cmap)
    ax2.set_xlim(-tlim, tlim)
    ax2.plot(I_r_2, z, lw=0.5, ls='--', color='black')
    ax2.plot(I_l_2, z, lw=0.5, ls='--', color='black')
    ax2.annotate(r'$\xi $=0.325 ns', xy=(-5,0.7), color='white')


    ax3.pcolormesh(t, z, It_3, cmap=cmap)
    ax3.set_xlim(-tlim, tlim)
    ax3.plot(I_r_3, z, lw=0.5, ls='--', color='black')
    ax3.plot(I_l_3, z, lw=0.5, ls='--', color='black')
    ax3.annotate(r'$\xi $=0.35 ns', xy=(-5,0.7), color='white')


    ax4.pcolormesh(t, z, It_4, cmap=cmap)
    ax4.set_xlim(-tlim, tlim)
    ax4.plot(I_r_4, z, lw=0.5, ls='--', color='black')
    ax4.plot(I_l_4, z, lw=0.5, ls='--', color='black')
    ax4.annotate(r'$\xi $=0.375 ns', xy=(-5,0.7), color='white')
    ax4.set_ylabel('Propagation \n distance $z$ \n (km)')



    ax5.pcolormesh(t, z, It_5, cmap=cmap)
    ax5.set_xlim(-tlim, tlim)
    ax5.plot(I_r_5, z, lw=0.5, ls='--', color='black')
    ax5.plot(I_l_5, z, lw=0.5, ls='--', color='black')
    ax5.annotate(r'$\xi $=0.4 ns', xy=(-5,0.7), color='white')


    ax6.pcolormesh(t, z, It_6, cmap=cmap)
    ax6.set_xlim(-tlim, tlim)
    ax6.plot(I_r_6, z, lw=0.5, ls='--', color='black')
    ax6.plot(I_l_6, z, lw=0.5, ls='--', color='black')
    ax6.annotate(r'$\xi $=0.425 ns', xy=(-5,0.7), color='white')


    ax7.pcolormesh(t, z, It_7, cmap=cmap)
    ax7.set_xlim(-tlim, tlim)
    ax7.plot(I_r_7, z, lw=0.5, ls='--', color='black')
    ax7.plot(I_l_7, z, lw=0.5, ls='--', color='black')
    ax7.annotate(r'$\xi $=0.45 ns', xy=(-5,0.7), color='white')
    ax7.set_xlabel('Time delay $t$ (ns)')
    ax7.set_ylabel('Propagation \n distance $z$ \n (km)')



    ax8.pcolormesh(t, z, It_8, cmap=cmap)
    ax8.set_xlim(-tlim, tlim)
    ax8.plot(I_r_8, z, lw=0.5, ls='--', color='black')
    ax8.plot(I_l_8, z, lw=0.5, ls='--', color='black')
    ax8.set_xlabel('Time delay $t$ (ns)')
    ax8.annotate(r'$\xi $=0.475 ns', xy=(-5,0.7), color='white')


    ax9.pcolormesh(t, z, It_9, cmap=cmap)
    ax9.set_xlim(-tlim, tlim)
    ax9.plot(I_r_9, z, lw=0.5, ls='--', color='black')
    ax9.plot(I_l_9, z, lw=0.5, ls='--', color='black')
    ax9.annotate(r'$\xi $=0.5 ns', xy=(-5,0.7), color='white')
    ax9.set_xlabel('Time delay $t$ (ns)')




    if oName:
        plt.savefig(oName,format='png',dpi=600, bbox_inches="tight")
    else:
        plt.show()


def fetch_data(f_name):
    dat = np.load(f_name)
    return dat['z'], dat['t'], dat['w'], dat['atz'], dat['utz']

def main_a():
    f_name = ['../numExp04_NLPM-750/NLPM-750_offset_300.npz', '../numExp04_NLPM-750/NLPM-750_offset_325.npz',
              '../numExp04_NLPM-750/NLPM-750_offset_450.npz', '../numExp04_NLPM-750/NLPM-750_offset_375.npz',
              '../numExp04_NLPM-750/NLPM-750_offset_400.npz', '../numExp04_NLPM-750/NLPM-750_offset_425.npz',
              '../numExp04_NLPM-750/NLPM-750_offset_450.npz', '../numExp04_NLPM-750/NLPM-750_offset_475.npz',
              '../numExp04_NLPM-750/NLPM-750_offset_500.npz']

    z, t, w, atz_1, utz_1 = fetch_data(f_name[0])
    z, t, w, atz_2, utz_2 = fetch_data(f_name[1])
    z, t, w, atz_3, utz_3 = fetch_data(f_name[2])
    z, t, w, atz_4, utz_4 = fetch_data(f_name[3])
    z, t, w, atz_5, utz_5 = fetch_data(f_name[4])
    z, t, w, atz_6, utz_6 = fetch_data(f_name[5])
    z, t, w, atz_7, utz_7 = fetch_data(f_name[6])
    z, t, w, atz_8, utz_8 = fetch_data(f_name[7])
    z, t, w, atz_9, utz_9 = fetch_data(f_name[8])
    z/=1e9; t/=1e3

    I_1 = []; I_l_1 = []; I_r_1 = []
    I_2 = []; I_l_2 = []; I_r_2 = []
    I_3 = []; I_l_3 = []; I_r_3 = []
    I_4 = []; I_l_4 = []; I_r_4 = []
    I_5 = []; I_l_5 = []; I_r_5 = []
    I_6 = []; I_l_6 = []; I_r_6 = []
    I_7 = []; I_l_7 = []; I_r_7 = []
    I_8 = []; I_l_8 = []; I_r_8 = []
    I_9 = []; I_l_9 = []; I_r_9 = []

    for i in range(z.size):
        at = atz_1[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_1.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_1.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_1.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_2[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_2.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_2.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_2.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_3[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_3.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_3.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_3.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_4[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_4.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_4.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_4.append(np.sum(t * It_l_t) / np.sum(It_l_t))

    for i in range(z.size):
        at = atz_5[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_5.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_5.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_5.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_6[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_6.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_6.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_6.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_7[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_7.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_7.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_7.append(np.sum(t * It_l_t) / np.sum(It_l_t))

    for i in range(z.size):
        at = atz_8[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_8.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_8.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_8.append(np.sum(t * It_l_t) / np.sum(It_l_t))


    for i in range(z.size):
        at = atz_9[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = FT(aw)

        It = np.abs(at) ** 2
        I_9.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        It_r_max = np.max(It_r)
        It_r_t = np.where(It > 0.9 * It_r_max, It_r, 0)
        I_r_9.append(np.sum(t * It_r_t) / np.sum(It_r_t))

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        It_l_max = np.max(It_l)
        It_l_t = np.where(It > 0.9 * It_l_max, It_l, 0)
        I_l_9.append(np.sum(t * It_l_t) / np.sum(It_l_t))

    tlim = 5.
    figure_1d(t, z, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8, I_9,I_r_1, I_r_2,I_r_3,I_r_4,I_r_5,I_r_6,I_r_7,I_r_8,I_r_9,
              I_l_1,I_l_2,I_l_3,I_l_4,I_l_5,I_l_6,I_l_7,I_l_8,I_l_9, tlim=tlim, oName='fig_2.png')


if __name__ == "__main__":
    main_a()
