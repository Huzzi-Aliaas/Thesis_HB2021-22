"""
This module implements functions for postprocessing of simulation data.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
from .config import FTFREQ, FT, IFT, SHIFT
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', size=14)
# rc('legend', fontsize=13)
# rc('text.latex', preamble=r'\usepackage{cmbright}')


def plot_evolution_coupled(z, t, a, u, tLim=None, wLim=None, oName=None):
    def _setColorbar(im, refPos):
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.02 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    def _truncate(I):
        I[I < 1e-6] = 1e-6
        return I

    w = SHIFT(FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi)

    if tLim == None:
        tLim = (np.min(t), np.max(t))
    if wLim == None:
        wLim = (np.min(w), np.max(w))

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, sharey=True)

    cmap = mpl.cm.get_cmap("jet")
    mpl.rcParams['axes.formatter.useoffset'] = False

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(a) ** 2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax1.pcolorfast(
        t, z, It[:-1, :-1], norm=col.LogNorm(vmin=It.min(), vmax=It.max()), cmap=cmap
    )
    cbar1 = _setColorbar(im1, ax1.get_position())
    cbar1.ax.set_title(r"$|A|^2$", color="k", y=3.5)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.0, z.max()])
    ax1.set_xlabel("Time delay \n $t$ (ps)")
    ax1.set_ylabel(r"Propagation distance $z$ (m)")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS
    Iw = np.abs(SHIFT(FT(a, axis=-1), axes=-1)) ** 2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(
        w, z, Iw[:-1, :-1], norm=col.LogNorm(vmin=Iw.min(), vmax=Iw.max()), cmap=cmap
    )
    cbar2 = _setColorbar(im2, ax2.get_position())
    cbar2.ax.set_title(r"$|A_\omega|^2$", color="k", y=3.5)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.0, z.max()])
    ax2.set_xlabel("Angular frequency \n detuning $\omega$ (rad/ps)")
    ax2.tick_params(labelleft=False)

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u) ** 2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax3.pcolorfast(
        t, z, It[:-1, :-1], norm=col.LogNorm(vmin=It.min(), vmax=It.max()), cmap=cmap
    )
    cbar1 = _setColorbar(im1, ax3.get_position())
    cbar1.ax.set_title(r"$|A|^2$", color="k", y=3.5)
    ax3.xaxis.set_ticks_position("bottom")
    ax3.yaxis.set_ticks_position("left")
    ax3.set_xlim(tLim)
    ax3.set_ylim([0.0, z.max()])
    ax3.set_xlabel("Time delay \n $t$ (ps)")
    ax3.tick_params(labelleft=False)

    # ax3.set_ylabel(r"Propagation distance $z$ (m)")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS
    Iw = np.abs(SHIFT(FT(u, axis=-1), axes=-1)) ** 2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax4.pcolorfast(
        w, z, Iw[:-1, :-1], norm=col.LogNorm(vmin=Iw.min(), vmax=Iw.max()), cmap=cmap
    )
    cbar2 = _setColorbar(im2, ax4.get_position())
    cbar2.ax.set_title(r"$|A_\omega|^2$", color="k", y=3.5)
    ax4.xaxis.set_ticks_position("bottom")
    ax4.yaxis.set_ticks_position("left")
    ax4.set_xlim(wLim)
    ax4.set_ylim([0.0, z.max()])
    ax4.set_xlabel("Angular frequency \n detuning $\omega$ (rad/ps)")
    ax4.tick_params(labelleft=False)

    if oName:
        plt.savefig(oName + ".png", format="png", bbox_inches='tight', dpi=600)
    else:
        plt.show()



def plot_details_prop_const(w, beta1, beta2, oName=None):
    """Generate a figure of the group-delay and group-velocity dispersion.

    Generates a plot showing the grop-delay (top subplot) and group-velocity
    dispersion (bottom subplot).

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta1 (:obj:`numpy.ndarray`):
            Group-delay profile.
        beta2 (:obj:`numpy.ndarray`):
            Group-velocity dispersion profile.
    """

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 4))
    plt.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.96, hspace=0.1)
    plt.tight_layout()

    l1 = ax1.plot(w, beta1, color="k", linewidth=1)
    ax1.axvspan(2.39, 3.3, color='lightgray', alpha=0.5)
    ax1.axvspan(1.2, 1.492, color='lightgray', alpha=0.5)


    ax1.set_xlim(np.min(w), np.max(w))
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.tick_params(axis="y", length=2.0)
    ax1.tick_params(axis="x", length=2.0, labelbottom=False)
    ax1.set_ylabel(r"GD $\beta_1$ (fs/$\mu$m)")
    ax1.title.set_text('(a)')
    ax1.text(1.30, 4.975, '$\Omega_{ND}$', fontsize=10)
    ax1.text(1.90, 4.975, '$\Omega_{AD}$', fontsize=10)
    ax1.text(3.0, 4.975, '$\Omega_{ND}$', fontsize=10)

    l2 = ax2.plot(w, beta2, color="k", linewidth=1)
    ax2.axhline(0, color="k", lw=0.75, ls=":")
    ax2.axvspan(2.39, 3.3, color='lightgray', alpha=0.5)
    ax2.axvspan(1.2, 1.492, color='lightgray', alpha=0.5)



    ax2.set_xlim(np.min(w), np.max(w))
    ax2.ticklabel_format(useOffset=False, style="plain")
    ax2.tick_params(axis="y", length=2.0)
    ax2.tick_params(axis="x", length=2.0)
    ax2.set_ylabel(r"GVD $\beta_2$  (fs$^2$/$\mu$m)")

    ax2.set_xlabel(r"Angular frequency $\omega~\mathrm{(rad/fs)}$")
    ax2.title.set_text('(b)')

    if oName:
        plt.savefig(oName + ".png", format="png", dpi=600, bbox_inches="tight")
    else:
        plt.show()


def plot_evolution_1(z, t, u, tLim=None, wLim=None, oName=None):
    def _setColorbar(im, refPos):
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.02 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    def _truncate(I):
        I[I < 1e-6] = 1e-6
        return I

    w = SHIFT(FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi)

    if tLim == None:
        tLim = (np.min(t), np.max(t))
    if wLim == None:
        wLim = (np.min(w), np.max(w))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    cmap = mpl.cm.get_cmap("jet")

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u) ** 2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax1.pcolorfast(
        t, z, It[:-1, :-1], norm=col.LogNorm(vmin=It.min(), vmax=It.max()), cmap=cmap
    )
    cbar1 = _setColorbar(im1, ax1.get_position())
    cbar1.ax.set_title("$|S|^2 = |S_1 + S_2|^2$", color="k", y=3.5)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.0, z.max()])
    ax1.set_xlabel("Time delay $t$ (ps) \n (a)")
    ax1.set_ylabel(r"Propagation distance $z$ (m)")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS
    Iw = np.abs(SHIFT(FT(u, axis=-1), axes=-1)) ** 2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(
        w, z, Iw[:-1, :-1], norm=col.LogNorm(vmin=Iw.min(), vmax=Iw.max()), cmap=cmap
    )
    cbar2 = _setColorbar(im2, ax2.get_position())
    cbar2.ax.set_title("$|S_{\omega}|^2 = |S_{\omega 1} + S_{\omega2}|^2$", color="k", y=3.5)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.0, z.max()])
    ax2.set_xlabel("Angular frequency detuning $\omega$ (rad/ps) \n (b)")
    ax2.tick_params(labelleft=False)

    if oName:
        plt.savefig(oName + ".png", format="png", dpi=600,bbox_inches="tight")
    else:
        plt.show()
        
        
   
def plot_evolution_2(z, t, u, tLim=None, wLim=None, oName=None):
    def _setColorbar(im, refPos):
        x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
        cax = f.add_axes([x0, y0 + 1.02 * h, w, 0.02 * h])
        cbar = f.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(
            color="k",
            labelcolor="k",
            bottom=False,
            direction="out",
            labelbottom=False,
            labeltop=True,
            top=True,
            size=4,
            pad=0,
        )

        cbar.ax.tick_params(which="minor", bottom=False, top=False)
        return cbar

    def _truncate(I):
        I[I < 1e-6] = 1e-6
        return I

    w = SHIFT(FTFREQ(t.size, d=t[1] - t[0]) * 2 * np.pi)

    if tLim == None:
        tLim = (np.min(t), np.max(t))
    if wLim == None:
        wLim = (np.min(w), np.max(w))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    cmap = mpl.cm.get_cmap("jet")

    # -- LEFT SUB-FIGURE: TIME-DOMAIN PROPAGATION CHARACTERISTICS
    It = np.abs(u) ** 2
    It /= np.max(It[0])
    It = _truncate(It)
    im1 = ax1.pcolorfast(
        t, z, It[:-1, :-1], norm=col.LogNorm(vmin=It.min(), vmax=It.max()), cmap=cmap
    )
    cbar1 = _setColorbar(im1, ax1.get_position())
    cbar1.ax.set_title("$|U|^2$", color="k", y=3.5)
    ax1.xaxis.set_ticks_position("bottom")
    ax1.yaxis.set_ticks_position("left")
    ax1.set_xlim(tLim)
    ax1.set_ylim([0.0, z.max()])
    ax1.set_xlabel("Time delay $t$ (ps) \n (a)")
    ax1.set_ylabel(r"Propagation distance $z$ (m)")

    # -- RIGHT SUB-FIGURE: ANGULAR FREQUENCY-DOMAIN PROPAGATION CHARACTERISTICS
    Iw = np.abs(SHIFT(FT(u, axis=-1), axes=-1)) ** 2
    Iw /= np.max(Iw[0])
    Iw = _truncate(Iw)
    im2 = ax2.pcolorfast(
        w, z, Iw[:-1, :-1], norm=col.LogNorm(vmin=Iw.min(), vmax=Iw.max()), cmap=cmap
    )
    cbar2 = _setColorbar(im2, ax2.get_position())
    cbar2.ax.set_title("$|U_\omega|^2$", color="k", y=3.5)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.set_xlim(wLim)
    ax2.set_ylim([0.0, z.max()])
    ax2.set_xlabel("Angular frequency detuning $\omega$ (rad/ps) \n (b)")
    ax2.tick_params(labelleft=False)

    if oName:
        plt.savefig(oName + ".png", format="png", dpi=600,bbox_inches="tight")
    else:
        plt.show()
        
        
             

def figure_1b(res,oName=None):
    """Plot RMS error of splitting schemes

    Generates a loglog-plot showing the scaling behavior of the
    root-mean-square error at given z-stepsize for the
    simple and symmetric operator splitting schemes.

    Args:
        res (array): results of the simulation run in main_b of ex04 part 1
        oName (str): name of output figure (optional, default: None)
    """

    dz, RMSError_1, RMSError_2, RMSError_3 = zip(*res)

    f, ax = plt.subplots()
    ax.plot(dz, RMSError_1, r"o-", label=r"simple splitting")
    ax.plot(dz, RMSError_2, r"^-", label=r"symmetric splitting")
    ax.plot(dz, RMSError_3, r"^-", label=r"Interaction picture method")
    ax.set(xlabel=r"stepsize $dz$",ylabel=r"RMS error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', ls='-', color='0.65')
    ax.legend()

    if oName:
        plt.savefig(oName,format='png',dpi=600)
    else:
        plt.show()




def figure_1c(res,oName=None):
    """Plot RMS error of splitting schemes

    Generates a loglog-plot showing the scaling behavior of the
    root-mean-square error at given z-stepsize for the
    simple and symmetric operator splitting schemes.

    Args:
        res (array): results of the simulation run in main_b of ex04 part 1
        oName (str): name of output figure (optional, default: None)
    """

    dz, RMSError_1, RMSError_2= zip(*res)

    f, ax = plt.subplots()
    ax.plot(dz, RMSError_1, r"o-", label=r"simple splitting")
    ax.plot(dz, RMSError_2, r"^-", label=r"symmetric splitting")
    ax.set(xlabel=r"stepsize $dz$",ylabel=r"RMS error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which='both', ls='-', color='0.65')
    ax.legend()

    if oName:
        plt.savefig(oName,format='png',dpi=600)
    else:
        plt.show()

