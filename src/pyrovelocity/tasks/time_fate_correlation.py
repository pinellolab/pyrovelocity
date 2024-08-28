from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import List, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from pyrovelocity.io.datasets import larry_cospar
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import plot_lineage_fate_correlation
from pyrovelocity.styles import configure_matplotlib_style
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import load_anndata_from_path
from pyrovelocity.workflows.main_configuration import (
    WorkflowConfiguration,
    larry_configuration,
    larry_mono_configuration,
    larry_multilineage_configuration,
    larry_neu_configuration,
)

__all__ = [
    "configure_time_lineage_fate_plot",
]

logger = configure_logging(__name__)

configure_matplotlib_style()


@beartype
def estimate_time_lineage_fate_correlation(
    reports_path: str | Path = "reports",
    model_identifier: str = "model2",
    configurations: List[WorkflowConfiguration] = [
        larry_mono_configuration,
        larry_neu_configuration,
        larry_multilineage_configuration,
        larry_configuration,
    ],
) -> Path:
    """
    This function is a duplicate of the `combine_time_lineage_fate_correlation`
    task function and will be removed in a future release.
    """
    n_rows = len(configurations)
    n_cols = 7
    width = 14
    height = width * (n_rows / n_cols) + 1

    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(
        n_rows + 1,
        n_cols + 1,
        width_ratios=[0.02] + [1] * n_cols,
        height_ratios=[1] * n_rows + [0.2],
    )

    adata_cospar = larry_cospar()

    all_axes = []
    for i, config in enumerate(configurations):
        data_set_name = config.download_dataset.data_set_name
        data_set_model_pairing = f"{data_set_name}_{model_identifier}"
        model_path = f"models/{data_set_model_pairing}"

        adata_pyrovelocity = load_anndata_from_path(
            f"{model_path}/postprocessed.h5ad"
        )
        posterior_samples_path = f"{model_path}/pyrovelocity.pkl.zst"
        plot_path = (
            Path(reports_path)
            / f"time_fate_correlation_{data_set_model_pairing}.pdf"
        )

        axes = [fig.add_subplot(gs[i, j + 1]) for j in range(n_cols)]
        all_axes.append(axes)

        plot_lineage_fate_correlation(
            posterior_samples_path=posterior_samples_path,
            adata_pyrovelocity=adata_pyrovelocity,
            adata_cospar=adata_cospar,
            all_axes=axes,
            fig=fig,
            state_color_dict=LARRY_CELL_TYPE_COLORS,
            lineage_fate_correlation_path=plot_path,
            save_plot=False,
            ylabel="",
            show_titles=True if i == 0 else False,
            show_colorbars=False,
            default_fontsize=12 if matplotlib.rcParams["text.usetex"] else 9,
        )

    return configure_time_lineage_fate_plot(
        fig=fig,
        gs=gs,
        all_axes=all_axes,
        row_labels=["a", "b", "c", "d"],
        vertical_texts=[
            "Monocytes",
            "Neutrophils",
            "Multilineage",
            "All lineages",
        ],
        reports_path=Path(reports_path),
        model_identifier=model_identifier,
    )


@beartype
def configure_time_lineage_fate_plot(
    fig: Figure,
    gs: GridSpec,
    all_axes: List[List[Axes]],
    row_labels: List[str],
    vertical_texts: List[str],
    reports_path: Path,
    model_identifier: str,
) -> Path:
    """
    Finalize the time lineage fate correlation plot by adding labels, legends, and colorbars.

    Args:
        fig: The main figure object.
        gs: The GridSpec object used for the subplot layout.
        all_axes: A list of lists containing all subplot Axes objects.
        row_labels: Labels for each row (e.g., "a", "b", "c", "d").
        vertical_texts: Vertical text labels for each row.
        reports_path: Path to save the final plot.
        model_identifier: Identifier for the model used.

    Returns:
        Path: The path where the final plot is saved.
    """
    _set_axes_aspect(all_axes)
    _add_row_labels(fig, gs, row_labels, vertical_texts)
    _add_legend(fig, gs, all_axes)
    _adjust_layout(fig)
    _add_colorbars(fig, gs, all_axes)
    return _save_plot(fig, reports_path, model_identifier)


def _set_axes_aspect(all_axes: List[List[Axes]]) -> None:
    for row_axes in all_axes:
        for ax in row_axes:
            ax.set_aspect("equal", adjustable="box")


def _add_row_labels(
    fig: Figure, gs: GridSpec, row_labels: List[str], vertical_texts: List[str]
) -> None:
    for i, (label, vtext) in enumerate(zip(row_labels, vertical_texts)):
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            1,
            rf"\textbf{{{label}}}"
            if matplotlib.rcParams["text.usetex"]
            else f"{label}",
            fontweight="bold",
            fontsize=12,
            ha="center",
            va="top",
        )
        label_ax.text(
            0.5, 0.5, vtext, rotation=90, fontsize=12, ha="center", va="center"
        )


def _add_legend(fig: Figure, gs: GridSpec, all_axes: List[List[Axes]]) -> None:
    legend_ax = fig.add_subplot(gs[-1, 1:3])
    legend_ax.axis("off")
    handles, labels = all_axes[-1][0].get_legend_handles_labels()
    legend_ax.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=(-0.1, -0.2),
        ncol=5,
        fancybox=True,
        prop={"size": 12},
        fontsize=12,
        frameon=False,
        markerscale=4,
        columnspacing=0.7,
        handletextpad=0.1,
    )


def _adjust_layout(fig: Figure) -> None:
    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.98, bottom=0.08, wspace=0.1, hspace=0.2
    )


def _add_colorbars(
    fig: Figure, gs: GridSpec, all_axes: List[List[Axes]]
) -> None:
    add_colorbar_axes = all_axes[-1][-3:]
    add_colorbar_artists = [ax.collections[0] for ax in add_colorbar_axes]
    for i, im in enumerate(add_colorbar_artists):
        cbar_ax = fig.add_subplot(gs[-1, -3 + i])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.locator = MaxNLocator(nbins=2)
        cbar.update_ticks()
        cbar_ax.xaxis.set_ticks_position("bottom")
        cbar_ax.xaxis.set_label_position("bottom")
        _adjust_colorbar_position(cbar_ax, add_colorbar_axes[i])


def _adjust_colorbar_position(cbar_ax: Axes, ref_ax: Axes) -> None:
    ax_pos = ref_ax.get_position()
    cbar_width = ax_pos.width * 0.6
    cbar_height = 0.01
    cbar_ax.set_position(
        [ax_pos.x0 + (ax_pos.width - cbar_width), 0.12, cbar_width, cbar_height]
    )


def _save_plot(
    fig: Figure,
    reports_path: Path,
    model_identifier: str,
) -> Path:
    combined_plot_path = (
        reports_path / f"combined_time_fate_correlation_{model_identifier}.pdf"
    )
    for ext in ["", ".png"]:
        fig.savefig(
            fname=f"{combined_plot_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
            transparent=False,
        )
    plt.close(fig)
    return combined_plot_path
