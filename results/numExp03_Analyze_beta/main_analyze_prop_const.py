import sys; sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np
from gnse.propagation_constant import PropConst, define_beta_fun_NLPM750
from gnse.tools import plot_details_prop_const


def main():
    pc = PropConst(define_beta_fun_NLPM750())
    c0 = 0.000299792458  # (micron/fs)
    # -- DETERMINE ZERO-DISPERSION POINT SEPARATING ANOMALOUS AND NORMAL DOMAIN OF DISPERSION
    w_min, w_max = 2., 2.5
    w_Z = pc.find_root_beta2(w_min, w_max)
    print("# w_Z = %lf" % (w_Z))

    # -- DETERMINE FREQUENCY IN DOMAIN OF NORMAL DISPERSION, GV-MATCHED TO SOLITON
    w0, w_min, w_max = 1.5, w_Z, 3.
    lam_1 = ((2*np.pi*c0*1e3) / w0); print('# lambda_1 =', lam_1, 'nm')
    w_GVM = pc.find_match_beta1(w0, w_min, w_max)
    print("# w_GVM = %lf" % (w_GVM))
    lam_2 = ((2*np.pi*c0*1e3) / w_GVM); print('# lambda_2 =', lam_2, 'nm')


    print('# GD_1 =',pc.beta1(2.))
    print('# GVD_1 =',pc.beta2(2.))
    print('# GD_2 =',pc.beta1(2.777628))
    print('# GVD_2 =',pc.beta2(2.777628))

    # -- SHOW BETA1 AND BETA2
    # w = np.linspace(-4.16, -4.10, 100)
    # w = np.linspace(17.275, 17.35, 100)
    w = np.linspace(1.2, 3.3, 400)


    plot_details_prop_const(w, pc.beta1(w), pc.beta2(w), oName='beta2')



if __name__ == '__main__':
    main()
