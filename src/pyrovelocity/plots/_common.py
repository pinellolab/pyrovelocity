from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_colorbar(
    smp,
    ax,
    orientation="vertical",
    labelsize=None,
    fig=None,
    position="right",
    rainbow=False,
):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if position == "right" and (not rainbow):
        cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
        cb = fig.colorbar(smp, orientation=orientation, cax=cax)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="8%", pad=0.08)
        cb = fig.colorbar(smp, cax=cax, orientation=orientation, shrink=0.4)

    cb.ax.tick_params(labelsize=labelsize)
    cb.set_alpha(1)
    cb.draw_all()
    cb.locator = MaxNLocator(nbins=2, integer=True)

    if position == "left":
        cb.ax.yaxis.set_ticks_position("left")
    cb.update_ticks()
