from pathlib import Path

import dill
import matplotlib.pyplot as plt
from beartype.typing import Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.gridspec import GridSpec

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.styles import configure_matplotlib_style

configure_matplotlib_style()

__all__ = [
    "plot_main",
    "plot_subfigures",
]


def create_main_figure(
    width: float,
    height: float,
    layout: Dict[str, List[float]],
) -> Tuple[FigureBase, Dict[str, Axes]]:
    """Create the main figure with all subplots."""
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(
        figure=fig,
        nrows=len(layout["height_ratios"]),
        ncols=len(layout["width_ratios"]),
        height_ratios=layout["height_ratios"],
        width_ratios=layout["width_ratios"],
    )

    axes = {}
    axes["ax1"] = fig.add_subplot(gs[0, :])
    axes["ax2"] = fig.add_subplot(gs[1, 0])
    axes["ax3"] = fig.add_subplot(gs[1, 1])
    axes["ax4"] = fig.add_subplot(gs[1, 2])
    axes["ax5"] = fig.add_subplot(gs[2, :])

    for key, ax in axes.items():
        ax.set_label(key)

    return fig, axes


def extract_subfigures(
    buffer: Optional[bytes] = None,
    axes_to_keep: List[str] = [],
    main_fig: Optional[FigureBase] = None,
    figure_file_path: Optional[str | Path] = None,
) -> FigureBase:
    """
    Extract a subset of axes from the main figure using dill for serialization.

    Args:
        main_fig: The main Figure object
        axes_to_keep: List of axes keys to keep in the new figure

    Returns:
        A new Figure object with only the specified axes
    """

    if buffer:
        subfig = dill.loads(buffer)
    elif main_fig:
        buffer = dill.dumps(main_fig)
        subfig = dill.loads(buffer)
    else:
        raise ValueError("Either buffer or main_fig must be provided.")

    for text in subfig.texts[:]:
        subfig.texts.remove(text)

    axes_to_remove = [
        ax for ax in subfig.axes if ax.get_label() not in axes_to_keep
    ]

    for ax in axes_to_remove:
        subfig.delaxes(ax)

    if figure_file_path:
        with Path(figure_file_path).open("wb") as f:
            dill.dump(subfig, f)

    return subfig


def plot_main(
    figure_file_path: Path | str = "main_figure.dill.zst",
) -> FigureBase:
    """Create an example plot with a custom layout and demonstrate subplot extraction."""
    width = 8.5 - 1
    height = (11 - 1) * 0.9
    layout = {
        "height_ratios": [0.1, 0.3, 0.6],
        "width_ratios": [0.5, 0.25, 0.25],
    }

    fig, axes = create_main_figure(width, height, layout)

    plot_wide_row(axes["ax1"])
    plot_small_cell(axes["ax2"])
    plot_narrow_column(axes["ax3"])
    plot_narrow_column(axes["ax4"])
    plot_large_cell(axes["ax5"])

    fig.tight_layout()

    x_col1 = -0.015
    y_row2 = 0.87
    add_panel_label(fig, "a", x_col1, 1.00)
    add_panel_label(fig, "b", x_col1, y_row2)
    add_panel_label(fig, "c", 0.45, y_row2)
    add_panel_label(fig, "d", 0.72, y_row2)
    add_panel_label(fig, "e", x_col1, 0.57)

    fig.savefig("example_plot_layout.pdf", format="pdf")
    CompressedPickle.save(figure_file_path, fig)

    return fig


def plot_subfigures(figure_file_path: Path | str = "main_figure.dill.zst"):
    fig = CompressedPickle.load(figure_file_path)
    buffer = dill.dumps(fig)

    subfig1 = extract_subfigures(buffer=buffer, axes_to_keep=["ax1"])
    subfig1.savefig("extracted_ax1.pdf", format="pdf")

    subfig23 = extract_subfigures(buffer=buffer, axes_to_keep=["ax2", "ax3"])
    subfig23.savefig("extracted_ax2_ax3.pdf", format="pdf")

    subfig34 = extract_subfigures(buffer=buffer, axes_to_keep=["ax3", "ax4"])
    subfig34.savefig("extracted_ax3_ax4.pdf", format="pdf")

    subfig15 = extract_subfigures(buffer=buffer, axes_to_keep=["ax1", "ax5"])
    subfig15.savefig("extracted_ax1_ax5.pdf", format="pdf")

    subfig5 = extract_subfigures(buffer=buffer, axes_to_keep=["ax5"])
    subfig5.savefig("extracted_ax5.pdf", format="pdf")


def example_plot_manual():
    """
    Create an example plot with a custom layout.

    Each subplot in the gridspec grid may be labeled with a panel label
    whose location is given in Figure-level coordinates.
    """
    n_rows = 3
    n_cols = 3
    width = 8.5 - 1
    height = (11 - 1) * 0.9
    row_1_fraction = 0.1
    row_2_fraction = 0.3
    row_3_fraction = 0.6

    col_1_fraction = 0.5
    col_2_fraction = 0.25
    col_3_fraction = 0.25

    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(
        figure=fig,
        nrows=n_rows,
        height_ratios=[
            row_1_fraction,
            row_2_fraction,
            row_3_fraction,
        ],
        ncols=n_cols,
        width_ratios=[
            col_1_fraction,
            col_2_fraction,
            col_3_fraction,
        ],
    )

    fig_1 = plt.figure(
        figsize=(
            (col_2_fraction + col_3_fraction) * width,
            (row_2_fraction) * height,
        )
    )
    gs_1 = GridSpec(
        figure=fig,
        nrows=1,
        height_ratios=[row_2_fraction],
        ncols=2,
        width_ratios=[
            col_2_fraction,
            col_3_fraction,
        ],
    )

    ax1 = fig.add_subplot(gs[0, :])
    plot_wide_row(ax1)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    plot_wide_row(ax)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_small_cell(ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_narrow_column(ax3)
    ax3_1 = fig_1.add_subplot(gs_1[0, 0])
    plot_narrow_column(ax3_1)

    ax4 = fig.add_subplot(gs[1, 2])
    plot_narrow_column(ax4)
    ax4_1 = fig_1.add_subplot(gs_1[0, 1])
    plot_narrow_column(ax4_1)

    ax5 = fig.add_subplot(gs[2, :])
    plot_large_cell(ax5)

    fig.tight_layout()
    fig_1.tight_layout()

    x_col1 = -0.015
    y_row2 = 0.87
    add_panel_label(fig, "a", x_col1, 1.00)
    add_panel_label(fig, "b", x_col1, y_row2)
    add_panel_label(fig, "c", 0.45, y_row2)
    add_panel_label(fig, "d", 0.72, y_row2)
    add_panel_label(fig, "e", x_col1, 0.57)

    fig.savefig("example_plot_layout.pdf", format="pdf")
    fig_1.savefig("example_plot_layout_1.pdf", format="pdf")


def plot_wide_row(
    ax: Axes,
    title: str = "Wide Row",
):
    ax.set_title(title)


def plot_small_cell(
    ax: Axes,
    title: str = "Small Cell",
):
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def plot_narrow_column(
    ax: Axes,
    title: str = "Narrow Column",
):
    ax.set_title(title)


def plot_large_cell(
    ax: Axes,
    title: str = "Large Cell",
):
    ax.set_title(title)


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
