import sys; sys.path.append('../../')
import numpy as np
from gnse.config import FTFREQ
from gnse.propagation_constant import PropConst, define_beta_fun_NLPM750
from gnse.tools import plot_evolution_coupled, plot_evolution_1, plot_evolution_2
from gnse.coupled_solver import Symmetric_Split_Step_Solver, Propagation_constant, Interaction_picture_method


def main():
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN
    tMax = 200000.  # (ps) bound for time mesh
    Nt = 2 ** 14  # (-) number of sample points: t-axis
    zMax = 400.  # (micron) upper limit for propagation routine
    Nz = 100000  # (-) number of sample points: z-axis
    nSkip = 50  # (-) number of z-steps to keep

    # -- SET PULSE PARAMETERS
    t01 = 200 # (fs) 1st pulse duration
    t02 = 150 # (fs) 2st pulse duration

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi

    # -- SET FIBER PARAMETERS
    pc = PropConst(define_beta_fun_NLPM750())
    w0 = 2.0 # (rad/fs) location of the first field
    beta21 = pc.beta2(w0); print('beta21 =', beta21, 'ps^2/µmm')  # (fs^2/micron) GVD of the first field
    w1 = 2.777628 # (rad/fs) location of the second field
    beta22 = pc.beta2(w1); print('beta22 =', beta22, 'ps^2/mµm') # (fs^2/micron) GVD of the second field
    beta1 = Propagation_constant(w, 0, 0, beta21)
    beta2 = Propagation_constant(w, 0, 0, beta22)
    gamma = 0.095 * 1e-6 ; print('gamma =', gamma, 'W/µm')

    LD = lambda t0: t0 * t0 / np.abs(beta21); print('LD =', LD(t01), 'µm')
    _z = np.linspace(0, zMax * LD(t01), Nz, endpoint=True)

    # -- DEFINE INTIAL REAL-VALUED FIELD. INITIALLY THE PULSE AMPLITUDE IS SET
    offset = 300
    P01 = np.abs(beta21) / t01 / t01 / gamma ; print('P01 =', P01)  # (W) pulse peak power
    P02 = np.abs(beta22) / t02 / t02 / gamma ; print('P02 =', P02) # (W) pulse peak power
    A0_t = np.sqrt(P01) / np.cosh((t + offset) / t01) - np.sqrt(P01) / np.cosh((t - offset) / t01)
    N = 0.55; print('N = ', N)
    U0_t = N * np.sqrt(P02) / np.cosh(t / t02)
    # U0_t = np.zeros(Nt)

    # -- ABSORPTION AT THE BOUNDARIES
    absorb_coeff = 10**15  # coeff. absorbtion
    absorb_tstart = tMax / 1.01  # where the layer starts
    absorb_twidth = 0.001  # width of the absorb. layer
    absorb = 0 #absorb_coeff * (1 / np.cosh(absorb_twidth * (t - absorb_tstart)) ** 2 + 1 / np.cosh(absorb_twidth * (t + absorb_tstart)) ** 2)
    # absorb = absorb_coeff * (1 / np.cosh(absorb_twidth * (w - absorb_tstart)) ** 2 + 1 / np.cosh(absorb_twidth * (w + absorb_tstart)) ** 2)


    # -- INITIALIZE SOLVER
    # my_solver = Interaction_picture_method(_z, t, beta1, beta2, gamma, absorb=absorb, nSkip=nSkip)
    my_solver = Symmetric_Split_Step_Solver(_z, t, beta1, beta2, gamma, absorb=absorb, nSkip=nSkip)
    my_solver.solve(A0_t, U0_t)

    # -- SAVE DATA
    # results = {
    #     "t": my_solver.t,
    #     "w": my_solver.w,
    #     "z": my_solver.z,
    #     "utz": my_solver.utz,
    #     "atz": my_solver.atz
    # }
    # np.savez_compressed('NLPM-750_minima_offset=500', **results)


    # -- POSTPROCESS RESULTS
    figName = 'Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)
    figName1 = '1_Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)
    figName2 = '2_Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)

    # plot_evolution_coupled(
    #     my_solver.z, my_solver.t, my_solver.atz, my_solver.utz, tLim=(-10000, 10000), wLim=(-0.1, 0.1)
    # )

    plot_evolution_1(
        (my_solver.z/1e6), my_solver.t, my_solver.atz, tLim=(-10000, 10000), wLim=(-0.05, 0.05), oName=figName1
    )

    plot_evolution_2(
        (my_solver.z/1e6), my_solver.t, my_solver.utz, tLim=(-10000, 10000), wLim=(-0.07, 0.07), oName=figName2
    )

if __name__ == "__main__":
    main()
