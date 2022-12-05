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
    f_name = '../numExp04_NLPM-750/NLPM-750_offset_400_N=0.55.npz'

    z, t, w, atz, utz = fetch_data(f_name)
    z /= 1e6

    I_ = []
    I_l = []
    I_r = []

    for i in range(z.size):
        at = atz[i]
        aw = np.where((w > 0.025) & (w < -0.025), 0j, FT(at))
        at = IFT(aw)

        It = np.abs(at) ** 2
        I_.append(It)

        at_r = np.where(t < 0, 0j, at)
        It_r = np.abs(at_r) ** 2
        I_r.append(It_r)

        at_l = np.where(t > 0, 0j, at)
        It_l = np.abs(at_l) ** 2
        I_l.append(It_l)

    I_ /= np.max(I_[0])
    I_l /= np.max(I_l[0])
    I_r /= np.max(I_r[0])


    j = 8050
    plt.plot(t[j:-j],I_[0][j:-j], color='black',label=r"z = 0")
    plt.plot(t[j:-j],I_[2][j:-j], color='green',label=r"z = 2 m",ls='--')
    plt.plot(t[j:-j],I_[14][j:-j], color='green',label=r"z = 15 m")
    plt.plot(t[j:-j],I_[37][j:-j], color='blue',label=r"z = 41 m",ls='--')
    plt.plot(t[j:-j],I_[62][j:-j], color='blue',label=r"z = 68 m")
    plt.plot(t[j:-j],I_[87][j:-j], color='red',label=r"z = 96 m",ls='--')
    plt.plot(t[j:-j],I_[116][j:-j], color='red',label=r"z = 128 m")
    plt.plot(t[j:-j],I_[147][j:-j], color='grey',label=r"z = 163 m",ls='--')
    plt.plot(t[j:-j],I_[179][j:-j], color='grey',label=r"z = 198 m")




    plt.legend()
    plt.xlabel("Time delay $t$ (ps)")
    plt.ylabel("$|S|^2$ (normalized)")
    plt.savefig('Intensity', dpi=600)
    #plt.show()


if __name__ == "__main__":
    main_a()





