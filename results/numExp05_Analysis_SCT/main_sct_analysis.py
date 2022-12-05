import sys; sys.path.append('../../')
import numpy as np
from gnse.config import FTFREQ, FT, IFT
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.optimize import curve_fit

def  energy(t,A):
    return np.trapz(np.abs(A)**2,dx=t[1]-t[0])

def fetch_data(f_name):
    dat = np.load(f_name)
    return dat['z'], dat['t'], dat['w'], dat['atz'], dat['utz']

def main_a():
    # -- READ IN DATA
    f_name = '../numExp04_NLPM-750/NLPM-750.npz'
    z, t, w, atz, utz = fetch_data(f_name)

    def func(x, a, b, c):
        return a * np.exp(x/b) + c

    k = 8135
    energies = np.array([energy(t, utz[i][k:-k]) for i in range(len(z))])/806.08924502
    p0 = (0.5, -152,   0.38)
    popt, pcov = curve_fit(func, (z/1e6), energies, p0)
    print(popt)
    print(pcov)
    # print(energies)
    plt.plot((z/1e6), energies, label=r"Change in energy", color='grey')
    plt.plot((z/1e6), func((z/1e6), *popt),"--", label=r"Curve fitting", color='black')
    plt.xlabel('Propagation distance $z$ (m)')
    plt.ylabel('Energy $E$ (normalized)')
    plt.title('$E(z) = \int |U (z)|^2 dt$')
    plt.legend()
    plt.savefig('energy_utz', dpi=600)
    # plt.show()



def main_b():
    # -- READ IN DATA
    f_name = '../numExp04_NLPM-750/NLPM-750.npz'
    z, t, w, atz, utz = fetch_data(f_name)
    k = 8135
    ut0 = np.abs(utz[0][k:-k]) ** 2
    ut1 = np.abs(utz[499][k:-k]) ** 2
    plt.plot(t[k:-k], ut0, color="b", label='Utz at z = 0')
    plt.plot(t[k:-k], ut1, color="g", label='Utz at z = 400 LD')
    plt.xlabel('time t $(fs)$')
    plt.ylabel('$|U(z,t)|^2$')
    plt.savefig('Utz')
    plt.show()

def main_c():
    # -- READ IN DATA
    f_name = '../numExp04_NLPM-750/NLPM-750.npz'
    z, t, w, atz, utz = fetch_data(f_name)
    k = 8192
    at0 = np.abs(atz[100][:-k]) ** 2
    at1 = np.abs(atz[200][:-k]) ** 2
    plt.plot(t[:-k], at0, color="b", label="100")
    plt.plot(t[:-k], at1, color="g", label="499")
    plt.xlabel('time t $(fs)$')
    plt.ylabel('$|A(z,t)|^2$')
    # plt.savefig('atz')
    plt.show()

def main_d():
    # -- READ IN DATA
    f_name = '../numExp04_NLPM-750/NLPM-750.npz'
    z, t, w, atz, utz = fetch_data(f_name)
    k = 8000
    energies = np.array([energy(t, atz[i][k:-k]) for i in range(len(z))])/2834.646026160649

    plt.plot((z/1e6), energies, label=r"Change in energy", color='grey')
    plt.xlabel('Propagation distance $z$ (m)')
    plt.ylabel('Energy $E$ (normalized)')
    plt.title('$E(z) = \int |S_1(z) + S_2(z)|^2 dt$')
    plt.savefig('energy_atz', dpi=600)
    plt.legend()
    # plt.show()

if __name__ == "__main__":
     # main_a()
    # main_b()
    # main_c()
     main_d()
