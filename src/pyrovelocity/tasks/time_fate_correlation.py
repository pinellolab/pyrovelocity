from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import List, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from pyrovelocity.analysis.trajectory import get_clone_trajectory
from pyrovelocity.io.datasets import (
    larry_cospar,
)
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
    "create_time_lineage_fate_correlation_plot",
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
def create_time_lineage_fate_correlation_plot(
    model_results: List[dict],
    vertical_texts: List[str] = [
        "Monocytes",
        "Neutrophils",
        "Multilineage",
        "All lineages",
    ],
    reports_path: Union[str, Path] = ".",
) -> Path:
    """
    Create a time lineage fate correlation plot from model results.

    This function is designed to be called from Flyte workflow or standalone Python code,
    processing model outputs to create lineage fate correlation visualizations.

    Args:
        model_results: List of dictionaries containing model outputs with the following keys:
            - data_model: String identifier for the data model
            - postprocessed_data: Path to the postprocessed AnnData file
            - pyrovelocity_data: Path to the posterior samples file
        vertical_texts: Labels for each row in the plot
        reports_path: Directory to save the plot

    Returns:
        Path: The path where the final plot is saved
    """
    n_rows = len(model_results)
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

    logger.info("Generating clone trajectories for all datasets")
    clone_trajectories = {}

    for model_output in model_results:
        data_set_model_pairing = model_output["data_model"]
        dataset_name = data_set_model_pairing.split("_model")[0]

        if dataset_name in clone_trajectories:
            continue

        postprocessed_data_path = model_output["postprocessed_data"]

        logger.info(f"Loading data for {dataset_name}")
        adata_pyrovelocity = load_anndata_from_path(postprocessed_data_path)

        if dataset_name == "larry_multilineage":
            logger.info(
                "Creating multilineage clone trajectory from mono and neu datasets"
            )

            from pyrovelocity.io.datasets import larry_mono, larry_neu

            mono_adata = larry_mono()
            neu_adata = larry_neu()

            logger.info(
                f"  - Generating mono trajectory with {mono_adata.n_obs} cells"
            )
            mono_clone = get_clone_trajectory(mono_adata)

            logger.info(
                f"  - Generating neu trajectory with {neu_adata.n_obs} cells"
            )
            neu_clone = get_clone_trajectory(neu_adata)

            logger.info("  - Concatenating mono and neu trajectories")
            clone_trajectories[dataset_name] = mono_clone.concatenate(neu_clone)
        else:
            logger.info(
                f"Generating clone trajectory for {dataset_name} with {adata_pyrovelocity.n_obs} cells"
            )
            clone_trajectories[dataset_name] = get_clone_trajectory(
                adata_pyrovelocity
            )

        logger.info(f"Completed trajectory generation for {dataset_name}")

    logger.info("Creating plots using generated trajectories")
    all_axes = []
    data_set_model_pairing = None

    for i, model_output in enumerate(model_results):
        data_set_model_pairing = model_output["data_model"]
        dataset_name = data_set_model_pairing.split("_model")[0]

        postprocessed_data_path = model_output["postprocessed_data"]
        posterior_samples_path = model_output["pyrovelocity_data"]

        plot_path = Path(f"time_fate_correlation_{data_set_model_pairing}.pdf")

        axes = [fig.add_subplot(gs[i, j + 1]) for j in range(n_cols)]
        all_axes.append(axes)

        adata_input_clone = clone_trajectories[dataset_name]
        logger.info(f"Using cached clone trajectory for {dataset_name}")

        plot_lineage_fate_correlation(
            posterior_samples_path=posterior_samples_path,
            adata_pyrovelocity=postprocessed_data_path,
            adata_cospar=adata_cospar,
            all_axes=axes,
            fig=fig,
            state_color_dict=LARRY_CELL_TYPE_COLORS,
            adata_input_clone=adata_input_clone,
            lineage_fate_correlation_path=plot_path,
            save_plot=False,
            ylabel="",
            show_titles=True if i == 0 else False,
            show_colorbars=False,
            default_fontsize=12 if matplotlib.rcParams["text.usetex"] else 9,
        )

    row_labels = ["a", "b", "c", "d"][:n_rows]
    vertical_texts = vertical_texts[:n_rows]

    return configure_time_lineage_fate_plot(
        fig=fig,
        gs=gs,
        all_axes=all_axes,
        row_labels=row_labels,
        vertical_texts=vertical_texts,
        reports_path=Path(reports_path),
        model_identifier=data_set_model_pairing or "model",
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
        ncol=4,
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
