import matplotlib
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pyrovelocity.logging import configure_logging


__all__ = ["set_colorbar", "set_font_size"]

logger = configure_logging(__name__)


def set_font_size(size: int):
    matplotlib.rcParams.update({"font.size": size})


def set_colorbar(
    smp,
    ax,
    orientation="vertical",
    labelsize=None,
    fig=None,
    position="right",
    rainbow=False,
):
    if position == "right" and (not rainbow):
        cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
        cb = fig.colorbar(smp, orientation=orientation, cax=cax)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="8%", pad=0.08)
        cb = fig.colorbar(smp, cax=cax, orientation=orientation, shrink=0.4)

    cb.ax.tick_params(labelsize=labelsize)

    # TODO: remove cb.draw_all()
    #
    # MatplotlibDeprecationWarning: The draw_all function was deprecated in
    # Matplotlib 3.6 and will be removed two minor releases later. Use
    # fig.draw_without_rendering() instead. cbar.draw_all()
    #
    # draw_all is not required with cb.solids.set_alpha(1)
    # https://matplotlib.org/stable/api/colorbar_api.html#matplotlib.colorbar.Colorbar.set_alpha
    cb.solids.set_alpha(1)
    # cb.set_alpha(1)
    # cb.draw_all()

    cb.locator = MaxNLocator(nbins=2, integer=True)

    if position == "left":
        cb.ax.yaxis.set_ticks_position("left")
    cb.update_ticks()
