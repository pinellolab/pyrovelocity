import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.gridspec import GridSpec

from pyrovelocity.styles import configure_matplotlib_style

configure_matplotlib_style()

__all__ = ["example_plot"]


def example_plot():
    """
    Create an example plot with a custom layout.

    Each subplot in the gridspec grid may be labeled with a panel label
    whose location is given in Figure-level coordinates.
    """
    fig = plt.figure(figsize=(8.5 - 1, (11 - 1) * 0.9))

    gs = GridSpec(
        figure=fig,
        nrows=3,
        height_ratios=[0.1, 0.3, 0.6],
        ncols=3,
        width_ratios=[0.5, 0.25, 0.25],
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("First Row")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Second Row, Left")
    ax2.set_aspect("equal", adjustable="box")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Second Row, Right 1")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("Second Row, Right 2")

    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Third Row")

    plt.tight_layout()

    add_panel_label(fig, "a", -0.015, 1.00)
    add_panel_label(fig, "b", -0.015, 0.87)
    add_panel_label(fig, "c", 0.45, 0.87)
    add_panel_label(fig, "d", 0.72, 0.87)
    add_panel_label(fig, "e", -0.015, 0.58)

    plt.savefig("example_plot_layout.pdf", format="pdf")


def add_panel_label(
    fig: FigureBase,
    label: str,
    x: float,
    y: float,
    fontsize: int = 14,
    fontweight: str = "bold",
    va: str = "top",
    ha: str = "left",
):
    """
    Add a panel label to the figure using global figure coordinates.

    Args:
        fig: matplotlib figure object
        label: string, the label to add (e.g., 'a', 'b', 'c')
        x: float, x-coordinate in figure coordinates (0-1)
        y: float, y-coordinate in figure coordinates (0-1)
        fontsize: int, font size for the label
        fontweight: str, font weight for the label
        va: str, vertical alignment for the label
        ha: str, horizontal alignment for the label
    """
    fig.text(
        x=x,
        y=y,
        s=label,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
    )


def add_axis_label(
    ax: Axes,
    label: str,
    x: float = -0.1,
    y: float = 1.1,
    fontsize: int = 14,
    fontweight: str = "bold",
    va: str = "top",
    ha: str = "right",
):
    """
    Add a label to the given axes.
    """
    ax.text(
        x=x,
        y=y,
        s=label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
    )
