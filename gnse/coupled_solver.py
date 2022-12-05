"""
Implents solver base class that serves as driver for different propagation
algorithms. Currently, the following algorithms are supported:

    Interaction Picture Method
    SymmetricSplitStepSolver

DATE: 2021-04-22

"""
import numpy as np
from .config import FTFREQ, FT, IFT


def Propagation_constant(w, *betas):
    r""" Propagation constant.
    Args:
            w (:obj:`numpy.ndarray`):
                Angular frequency for which to compute propagation constant.
            *betas (:variable length argument: 'float')
                 All beta_n; where n = 0,1,2,3,... stands for nth order dispersion parameter.
    Returns:
            :obj:`numpy.ndarray`: Propagation constant.
    """
    propagation_constant = np.zeros(len(w))
    for idx, beta in enumerate(betas):
        propagation_constant += beta * w ** (idx) / np.math.factorial(idx)
    return propagation_constant



class SolverBaseClass:
    r"""Base class for solver.

    Implements solver base class that serves as driver for the implemented
    :math:`z`-propagation algorithms.

    Attributes:
        beta (:obj:`numpy.ndarray`):
           Frequency dependent propagation constant.
        gamma (:obj:`float` or :obj:`numpy.ndarray`):
           Coefficient function of nonlinear part.
        dz (:obj:`float`):
            Stepsize, i.e. :math:`z`-increment for integration.
        z_ (:obj:`numpy.ndarray`):
            :math:`z`-values used for :math:`z`-integration.
        t (:obj:`numpy.ndarray`):
            Temporal grid.
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        _z (:obj:`list`):
            :math:`z`-values for which field is stored and available after
            propagation.
        _a (:obj:`list`):
            Frequency domain representation of the 1st field at :math:`z`-values
            listed in `_z`.
        _u (:obj:`list`):
            Frequency domain representation of the 2nd field at :math:`z`-values
            listed in `_z`.
        nSkip (:obj:`int`):
            Step interval in which data is stored upon propagation (default: 1).

    Args:
        z (:obj:`numpy.ndarray`):
            :math:`z`-values used for :math:`z`-integration.
        t (:obj:`numpy.ndarray`):
            Temporal grid.
        beta1 (:obj:`numpy.ndarray`):
           Frequency dependent propagation constant of the 1st field.
        beta2 (:obj:`numpy.ndarray`):
           Frequency dependent propagation constant of the 2nd field.
        gamma (:obj:`float` or :obj:`numpy.ndarray`):
           Coefficient function of nonlinear part.
        nSkip (:obj:`int`):
            Step interval in which data is stored upon propagation (default: 1).

    """

    def __init__(self, z, t, beta1, beta2, gamma, absorb=0, nSkip=1):
        self.nSkip = nSkip
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.dz = z[1] - z[0]
        self.z_ = z
        self.t = t
        self.w = FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi
        self._z = []
        self._u = []
        self._a = []
        self.absorb = absorb


    def solve(self, a, u):
        r"""Propagate field

        Args:
            u (:obj:`numpy.ndarray`):
                Time-domain representation of initial field.
        """
        uw = FT(u)
        aw = FT(a)
        self._z.append(self.z_[0])
        self._u.append(uw)
        self._a.append(aw)
        for i in range(1, self.z_.size):
            aw, uw = self.singleStep(aw,uw)
            if i % self.nSkip == 0:
                self._a.append(aw)
                self._u.append(uw)
                self._z.append(self.z_[i])

    @property
    def atz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Time-domain representation of the 1st field"""
        return IFT(np.asarray(self._a), axis=-1)

    @property
    def utz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Time-domain representation of the 2nd field"""
        return IFT(np.asarray(self._u), axis=-1)

    @property
    def awz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Frequency-domain representation of the 1st field"""
        return np.asarray(self._a)

    @property
    def uwz(self):
        r""":obj:`numpy.ndarray`, 2-dim: Frequency-domain representation of the 2nd field"""
        return np.asarray(self._u)

    @property
    def z(self):
        r""":obj:`numpy.ndarray`, 1-dim: :math:`z`-slices at which field is
        stored"""
        return np.asarray(self._z)

    def singleStep(self, aw, uw):
        r"""Advance both the fields by a single :math:`z`-slice"""
        raise NotImplementedError


class Split_Step_Solver(SolverBaseClass):
     r"""Fixed stepsize algorithm implementing the simple split step
    method (SiSSM).

    Implements a fixed stepsize algorithm referred to as the simple split step
    Fourier method (SiSSM) [1].

    References:
        [1] T. R. Taha, M. J. Ablowitz,
        Analytical and numerical aspects of certain nonlinear evolution
        equations. II. Numerical, nonlinear Schrödinger equation,
        J. Comput. Phys. 55 (1984) 203,
        https://doi.org/10.1016/0021-9991(84)90003-2.
    """

     def singleStep(self, aw, uw):
        r"""Advance both the fields by a single :math:`z`-slice

        Implements symmetric splitting formula for split-step Fourier approach.

        Args:
            aw (:obj:`numpy.ndarray`): Frequency domain representation of the
            1st field at the current :math:`z`-position.
            uw (:obj:`numpy.ndarray`): Frequency domain representation of the
            2nd field at the current :math:`z`-position.

        Returns:
            obj:`numpy.ndarray`: Frequency domain representation of the 1st field
            and the 2nd field at :math:`z` + :math:`dz`.
        """

        # -- DECLARE CONVENIENT ABBREVIATIONS
        dz, w, beta1, beta2, gamma = self.dz, self.w, self.beta1, self.beta2, self.gamma

        # -- LINEAR STEP / FREQUENCY DOMAIN
        _lin_half = lambda xw, beta: np.exp(1j * dz * beta) * xw
        # -- NONLINEAR STEP / TIME DOMAIN
        _nlin = lambda xt, yt: np.exp(1j * gamma * (np.abs(xt) ** 2 + 2 * np.abs(yt) ** 2) * dz) * xt

        # -- ADVANCE FIELD
        aw1 = _lin_half(aw, beta1)
        uw1 = _lin_half(uw, beta2)
        aw = FT(_nlin(IFT(aw1), IFT(uw1)))
        uw = FT(_nlin(IFT(uw1), IFT(aw1)))
        return aw, uw


class Symmetric_Split_Step_Solver(SolverBaseClass):
    r"""Fixed stepsize algorithm implementing the symmetric split step
    method (SySSM).

    Implements a fixed stepsize algorithm referred to as the symmetric split
    step Fourier method (SySSM) as discussed in [1,2].

    References:
        [1] P. L. DeVries,
        Application of the Split Operator Fourier Transform method to the
        solution of the nonlinear Schrödinger equation,
        AIP Conference Proceedings 160, 269 (1987),
        https://doi.org/10.1063/1.36847.

        [2] J. Fleck, J. K. Morris, M. J. Feit,
        Time-dependent propagation of high-energy laser beams through the
        atmosphere: II,
        Appl. Phys. 10, (1976) 129,
        https://doi.org/10.1007/BF00882638.
    """

    def singleStep(self, aw, uw):
        r"""Advance both the fields by a single :math:`z`-slice
        Implements symmetric splitting formula for split-step Fourier approach.
        Args:
            aw (:obj:`numpy.ndarray`): Frequency domain representation of the
            1st field at the current :math:`z`-position.
            uw (:obj:`numpy.ndarray`): Frequency domain representation of the
            2nd field at the current :math:`z`-position.

        Returns:
            obj:`numpy.ndarray`: Frequency domain representation of the 1st field
            and the 2nd field at :math:`z` + :math:`dz`.
        """

        # -- DECLARE CONVENIENT ABBREVIATIONS
        dz, w, beta1, beta2, gamma, absorb  = self.dz, self.w, self.beta1, self.beta2, self.gamma, self.absorb

        # -- LINEAR STEP / FREQUENCY DOMAIN
        _lin_half = lambda xw, beta: np.exp(0.5j * beta * dz) * xw
        # -- NONLINEAR STEP / TIME DOMAIN
        _nlin = lambda xt, yt: np.exp(1j * gamma * (np.abs(xt) ** 2 + 2 * np.abs(yt) ** 2) * dz + 1j * absorb * dz) * xt

        # -- ADVANCE FIELD
        aw = _lin_half(aw, beta1)
        uw = _lin_half(uw, beta2)
        aw = FT(_nlin(IFT(aw), IFT(uw)))
        uw = FT(_nlin(IFT(uw), IFT(aw)))
        aw = _lin_half(aw, beta1)
        uw = _lin_half(uw, beta2)
        return aw, uw


class Interaction_picture_method(SolverBaseClass):

    r"""Fixed step size algorithm implementing the Runge-Kutta 4th
    order method.

    Implements a fixed step size algorithm referred to as the Interaction picture
    method as discussed in [1].

    References:
        [1] Johan Hult,
        A Fourth-Order Runge–Kutta in the Interaction Picture Method for Simulating
        Supercontinuum Generation in Optical Fibers,
        JOURNAL OF LIGHTWAVE TECHNOLOGY, VOL. 25, NO. 12, DECEMBER 2007,
    """

    def singleStep(self, aw, uw):
        r"""Advance field by a single :math:`z`-slice

        Implements Runge Kutta fourth order method for solving Nonlinear
        SchrÖdinger Equation.

        Args:
            aw (:obj:`numpy.ndarray`): Frequency domain representation of the
            1st field at the current :math:`z`-position.
            uw (:obj:`numpy.ndarray`): Frequency domain representation of the
            2nd field at the current :math:`z`-position.

        Returns:
            aw (:obj:`numpy.ndarray`): Frequency domain representation of the 1st
            field at :math:`z` + :math:`dz`.
            uw (:obj:`numpy.ndarray`): Frequency domain representation of the 2nd
            field at :math:`z` + :math:`dz`.
        """

        # -- DECLARE CONVENIENT ABBREVIATIONS
        dz, w, beta1, beta2, gamma, absorb = self.dz, self.w, self.beta1, self.beta2, self.gamma, self.absorb

        # -- DERIVATIVE OF THE AUXILIARY FIELD
        Non_lin = lambda xt, yt: FT(1j * gamma * (np.abs(xt)**2 + 2 * np.abs(yt)**2) * xt)
        def derivative(dz, xw, yw):
            xt = IFT(np.exp(1j * (beta1 + 1j * absorb) * dz) * xw)
            yt = IFT(np.exp(1j * (beta2 + 1j * absorb) * dz) * yw)
            return np.exp(-1j * (beta1 + 1j * absorb) * dz) * Non_lin(xt, yt), np.exp(-1j * (beta2 + 1j * absorb) * dz) * Non_lin(yt, xt)

        # -- 4TH ORDER RK METHOD FOR t-STEPPING AUX FIELD
        def Runge_Kutta_4(xw, yw):
            k1, l1 = derivative(0, xw, yw)
            k2, l2 = derivative(dz / 2, xw + dz * k1 / 2, yw + dz * l1 / 2)
            k3, l3 = derivative(dz / 2, xw + dz * k2 / 2, yw + dz * l2 / 2)
            k4, l4 = derivative(dz, xw + dz * k3, yw + dz * l3)
            return xw + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6, yw + dz * (l1 + 2 * l2 + 2 * l3 + l4) / 6

        # -- ADVANCE FIELD
        aw, uw = Runge_Kutta_4(aw, uw)

        # -- AUX. FIELD TO ORIGINAL FIELD BACK TRANSFORMATION
        return np.exp(1j * (beta1 + 1j * absorb) * dz) * aw, np.exp(1j * (beta2 + 1j * absorb) * dz) * uw
