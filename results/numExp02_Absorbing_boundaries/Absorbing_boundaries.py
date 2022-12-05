import sys; sys.path.append('../../')
import numpy as np
from gnse.coupled_solver import Symmetric_Split_Step_Solver, Propagation_constant, Interaction_picture_method
from gnse.tools import plot_evolution_coupled, plot_evolution
from gnse.config import FTFREQ

def main():

    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN
    tMax = 500.0  # (fs) bound for time mesh
    Nt = 2 ** 14  # (-) number of sample points: t-axis
    zMax = 125  # (micron) upper limit for propagation routine
    Nz = 20000  # (-) number of sample points: z-axis
    nSkip = 100  # (-) keep only every nskip-th system state

    # -- SET FIBER PARAMETERS
    beta21 = -1  # (1/micron)
    beta22 = 1  # (1/micron)
    gamma = 1  # (W/micron)

    t01 = 1.0  # (fs) pulse duration
    t02 = 0.75  # (fs) pulse duration

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
    _z = np.linspace(0, zMax, Nz, endpoint=True)
    beta1 = Propagation_constant(w, 0, 0, -1)
    beta2 = Propagation_constant(w, 0, 0, 1)

    # -- DEFINE INTIAL REAL-VALUED FIELD. INITIALLY THE PULSE AMPLITUDE IS SET
    offset = 5
    P01 = np.abs(beta21) / t01 / t01 / gamma  # (W) pulse peak power
    P02 = np.abs(beta22) / t02 / t02 / gamma  # (W) pulse peak power
    A0_t = np.sqrt(P01) / np.cosh(t + offset / t01) - np.sqrt(P01) / np.cosh(t - offset / t01)
    N = 0.05
    U0_t = N * np.sqrt(P02) / np.cosh(t / t02)

    # -- ABSORPTION AT THE BOUNDARIES
    absorb_coeff = 10  # coeff. absorbtion
    absorb_tstart = tMax / 4  # where the layer starts
    absorb_twidth = 0.001 * tMax / 2  # width of the absorb. layer
    absorb = absorb_coeff * (1 / np.cosh(absorb_twidth * (t - absorb_tstart)) ** 2 + 1 / np.cosh(absorb_twidth * (t + absorb_tstart)) ** 2)

    # -- INITIALIZE SOLVER
    my_solver = Symmetric_Split_Step_Solver(_z, t, beta1, beta2, gamma, absorb=absorb, nSkip=nSkip)
    my_solver.solve(A0_t, U0_t)

    # -- POSTPROCESS RESULTS
    figName = 'Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)
    figName1 = '1_Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)
    figName2 = '2_Figure_t01_%lf_offset_%lf_N_%lf_t02_%lf'%(t01, offset, N, t02)



    # -- SHOW RESULTS
    plot_evolution_coupled(
        my_solver.z, my_solver.t, my_solver.atz, my_solver.utz, tLim=(-200, 200), wLim=(-15, 15), oName=figName
    )
    plot_evolution(
        my_solver.z, my_solver.t, my_solver.atz, tLim=(-200, 200), wLim=(-15, 15), oName=figName1
    )
    plot_evolution(
        my_solver.z, my_solver.t, my_solver.utz, tLim=(-200, 200), wLim=(-15, 15), oName=figName2
    )
if __name__ == "__main__":
    main()
