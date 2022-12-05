import sys; sys.path.append('../../')
import numpy as np
from gnse.coupled_solver import Symmetric_Split_Step_Solver, Propagation_constant, Interaction_picture_method, Split_Step_Solver
from gnse.tools import figure_1b, plot_evolution_coupled, plot_evolution, figure_1c
from gnse.config import FTFREQ

def main():
    # -- SET PARAMETERS FOR COMPUTATIONAL DOMAIN
    tMax = 500  # (fs) bound for time mesh
    Nt = 2**14  # (-) number of sample points: t-axis
    Nz = 20000
    nSkip = 100  # (-) keep only every nskip-th system state

    # -- SET WAVEGUIDE PARAMETERS
    beta21 = -1
    beta22 = -1
    gamma = 1    # (W/m)

    # -- SET PULSE PARAMETERS
    t0 = 1     # (fs) pulse duration of the soliton

    # ... PEAK INTENSITY OF THE SOLITON
    A = 1 / np.sqrt(3)
    # ... FUNCTION IMPLEMENTING SOLITON INITIAL CONDITION
    sol = lambda t: A / np.cosh(t)

    # -- DISPERSION LENGTH OF SOLITON
    LD = lambda t0: t0 * t0 / np.abs(beta21)

    # -- INITIALIZE COMPUTATIONAL DOMAIN
    _z = np.linspace(0, LD(t0), Nz, endpoint=True)
    t = np.linspace(-tMax, tMax, Nt, endpoint=False)
    w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
    beta1 = Propagation_constant(w, 0, 0, beta21)
    beta2 = Propagation_constant(w, 0, 0, beta22)

    # -- ANONYMOUS FUNCTION: EXACT SOLITON SOLUTION
    _XExact = lambda z, t: A / np.cosh(t) * np.exp(0.5j * z)

    # -- ANONYMOUS FUNCTION: ROOT MEAN SQUARE ERROR
    _RMSError = lambda x, y: np.sqrt(np.sum(np.abs(x - y) ** 2) / x.size)

    # -- RUN SIMULATION
    res_1 = []
    res_2 = []
    for Nz in [2 ** n for n in range(7, 14)]:
        print("# Nz = %d" % (Nz))
        _z = np.linspace(0, 2 * LD(t0), Nz, endpoint=True)

        # -- simple splitting scheme
        my_solver_1 = Split_Step_Solver(_z, t, beta1, beta2, gamma, nSkip=nSkip)
        my_solver_1.solve(sol(t), sol(t))
        Azt_1 = my_solver_1.atz
        Uzt_1 = my_solver_1.utz

        # -- symmetric splitting scheme
        my_solver_2 = Symmetric_Split_Step_Solver(_z, t, beta1, beta2, gamma, nSkip=nSkip)
        my_solver_2.solve(sol(t), sol(t))
        Azt_2 = my_solver_2.atz
        Uzt_2 = my_solver_2.utz

        # -- interaction picture method
        my_solver_3 = Interaction_picture_method(_z, t, beta1, beta2, gamma, nSkip=nSkip)
        my_solver_3.solve(sol(t), sol(t))
        Azt_3 = my_solver_3.atz
        Uzt_3 = my_solver_3.utz
        z = my_solver_2.z

        AExact = _XExact(z[-1], t)
        UExact = _XExact(z[-1], t)


        # -- ACCUMULATE SIMULATION RESULTS
        res_1.append((_z[1] - _z[0],  # z-stepsize
                    _RMSError(AExact, Azt_1[-1]),  # RMSE - simple splitting scheme
                    _RMSError(AExact, Azt_2[-1]),  # RMSE - symmetric splitting scheme
                   _RMSError(AExact, Azt_3[-1])  # RMSE - interaction picture method
                    ))

        # -- ACCUMULATE SIMULATION RESULTS
        res_2.append((_z[1] - _z[0],  # z-stepsize
                      _RMSError(UExact, Uzt_1[-1]),  # RMSE - simple splitting scheme
                      _RMSError(UExact, Uzt_2[-1]),  # RMSE - symmetric splitting scheme
                     _RMSError(UExact, Uzt_3[-1])  # RMSE - interaction picture method
                      ))

     # -- POSTPROCESS RESULTS
    figure_1b(res_1, 'Quality_control_1.png')
    figure_1b(res_2, 'Quality_control_2.png')

if __name__ == "__main__":
    main()
