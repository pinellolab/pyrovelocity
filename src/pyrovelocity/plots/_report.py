from pathlib import Path

import dill
import matplotlib.pyplot as plt
from anndata import AnnData
from beartype.typing import Any, Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from pandas import DataFrame

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plots._genes import plot_gene_ranking
from pyrovelocity.plots._parameters import (
    plot_parameter_posterior_distributions,
)
from pyrovelocity.plots._rainbow import rainbowplot_module as rainbowplot
from pyrovelocity.plots._vector_fields import plot_vector_field_summary
from pyrovelocity.styles import configure_matplotlib_style
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS

configure_matplotlib_style()

__all__ = [
    "plot_report",
    "save_subfigures",
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

    for key, ax in axes.items():
        ax.set_label(key)

    return fig, axes, gs


def plot_report(
    adata: AnnData,
    posterior_samples: Dict[str, NDArray[Any]],
    volcano_data: DataFrame,
    putative_marker_genes: List[str],
    selected_genes: List[str],
    vector_field_basis: str = "umap",
    cell_state: str = "clusters",
    state_color_dict: Optional[Dict[str, str]] = None,
    boxplot: bool = False,
    report_file_path: Path | str = "example_plot_report.pdf",
    figure_file_path: Path | str = "example_report_figure.dill.zst",
) -> FigureBase:
    """
    Plot a report figure with multiple subplots and serialize the Figure object.

    Args:
        figure_file_path (Path | str, optional):
            Figure object file. Defaults to "example_report_figure.dill.zst".
        adata (AnnData, optional):
            AnnData object. Defaults to adata.
        posterior_samples (Dict[str, NDArray[Any]], optional):
            Posterior samples dictionary. Defaults to posterior_samples.
        volcano_data (DataFrame, optional):
            Volcano data DataFrame. Defaults to volcano_data.
        vector_field_basis (str, optional):
            Vector field basis identifier. Defaults to "emb".
        cell_state (str, optional):
            Cell state identifier. Defaults to "state_info".
        putative_marker_genes (List[str], optional):
            List of putative marker genes. Defaults to putative_marker_genes.
        selected_genes (List[str], optional):
            List of genes to be included in report figure. Defaults to selected_genes.
        report_file_path (Path | str, optional):
            File to save report figure. Defaults to "example_plot_report.pdf".

    Returns:
        FigureBase: Figure object containing the report figure.

    Examples:
    >>> # xdoctest: +SKIP
    >>> from pyrovelocity.analysis.analyze import top_mae_genes
    >>> from pyrovelocity.plots import plot_report
    >>> from pyrovelocity.utils import load_anndata_from_path
    >>> from pyrovelocity.io.compressedpickle import CompressedPickle
    >>> from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
    ...
    >>> adata = load_anndata_from_path("models/larry_model2/postprocessed.h5ad")
    >>> posterior_samples = CompressedPickle.load(
    ...     "models/larry_model2/pyrovelocity.pkl.zst"
    ... )
    >>> volcano_data = posterior_samples["gene_ranking"]
    >>> vector_field_basis = "emb"
    >>> cell_state = "state_info"
    >>> putative_marker_genes = top_mae_genes(
    ...     volcano_data=volcano_data,
    ...     mae_top_percentile=3,
    ...     min_genes_per_bin=3,
    ... )
    >>> selected_genes = putative_marker_genes[:6]
    >>> selected_genes = ["Cyp11a1", "Csf2rb", "Osbpl8", "Lgals1", "Cmtm7", "Runx1"]
    >>> putative_marker_genes = list(set(putative_marker_genes + selected_genes))
    ...
    >>> plot_report(
    ...     adata=adata,
    ...     posterior_samples=posterior_samples,
    ...     volcano_data=volcano_data,
    ...     putative_marker_genes=putative_marker_genes,
    ...     selected_genes=selected_genes,
    ...     vector_field_basis=vector_field_basis,
    ...     cell_state=cell_state,
    ...     state_color_dict=LARRY_CELL_TYPE_COLORS,
    ...     report_file_path="example_plot_report.pdf",
    ... )
    """
    width = 8.5 - 1
    height = (11 - 1) * 0.9
    layout = {
        "height_ratios": [0.12, 0.28, 0.6],
        "width_ratios": [0.5, 0.25, 0.25],
    }

    selected_genes_in_adata = [
        gene for gene in selected_genes if gene in adata.var.index
    ]

    additional_genes = [
        gene
        for gene in putative_marker_genes
        if gene not in selected_genes_in_adata and gene in adata.var.index
    ]

    while len(selected_genes_in_adata) < 6 and additional_genes:
        selected_genes_in_adata.append(additional_genes.pop(0))

    extended_putative_marker_genes = selected_genes_in_adata.copy()
    for gene in putative_marker_genes:
        if (
            gene not in extended_putative_marker_genes
            and gene in adata.var.index
        ):
            extended_putative_marker_genes.append(gene)

    selected_genes = selected_genes_in_adata
    putative_marker_genes = extended_putative_marker_genes

    fig, axes, gs = create_main_figure(width, height, layout)
    plot_vector_field_summary(
        adata=adata,
        posterior_samples=posterior_samples,
        vector_field_basis=vector_field_basis,
        cell_state=cell_state,
        state_color_dict=state_color_dict,
        fig=fig,
        gs=gs[0, :],
        default_fontsize=7,
    )

    plot_gene_ranking(
        posterior_samples=posterior_samples,
        adata=adata,
        fig=fig,
        gs=gs[1, 0],
        putative_marker_genes=putative_marker_genes,
        selected_genes=selected_genes,
        time_correlation_with="st",
        show_marginal_histograms=False,
    )

    plot_parameter_posterior_distributions(
        posterior_samples=posterior_samples,
        adata=adata,
        geneset=selected_genes,
        fig=fig,
        gs=gs[1, 1:],
        boxplot=boxplot,
    )

    rainbowplot(
        volcano_data=volcano_data,
        adata=adata,
        posterior_samples=posterior_samples,
        genes=selected_genes,
        data=["st", "ut"],
        basis=vector_field_basis,
        cell_state=cell_state,
        fig=fig,
        gs=gs[2, :],
    )

    fig.tight_layout()

    x_col1 = -0.005
    y_row2 = 0.84
    add_panel_label(fig, "a", x_col1, 1.00)
    add_panel_label(fig, "b", x_col1, y_row2)
    add_panel_label(fig, "c", 0.47, y_row2)
    add_panel_label(fig, "d", x_col1, 0.57)

    fig.savefig(report_file_path, format="pdf")
    fig.savefig(
        f"{report_file_path}.png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
        format="png",
    )

    CompressedPickle.save(figure_file_path, fig)

    return fig


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


def save_subfigures(
    figure_file_path: Path | str = "main_figure.dill.zst",
    vector_field_summary_file_path: Path
    | str = "extracted_vector_field_summary.pdf",
    gene_selection_file_path: Path | str = "extracted_gene_selection.pdf",
    parameter_posteriors_file_path: Path
    | str = "extracted_parameter_posteriors.pdf",
    rainbow_file_path: Path | str = "extracted_rainbow.pdf",
):
    fig = CompressedPickle.load(figure_file_path)
    buffer = dill.dumps(fig)

    subfig_vector_field_summary = extract_subfigures(
        buffer=buffer,
        axes_to_keep=["vector_field"],
    )
    subfig_vector_field_summary.savefig(
        fname=vector_field_summary_file_path,
        format="pdf",
    )

    subfig_gene_selection = extract_subfigures(
        buffer=buffer,
        axes_to_keep=["gene_selection"],
    )
    subfig_gene_selection.savefig(
        fname=gene_selection_file_path,
        format="pdf",
    )

    subfig_parameter_posteriors = extract_subfigures(
        buffer=buffer,
        axes_to_keep=["parameter_posteriors"],
    )
    subfig_parameter_posteriors.savefig(
        fname=parameter_posteriors_file_path,
        format="pdf",
    )

    subfig_rainbow = extract_subfigures(
        buffer=buffer,
        axes_to_keep=["rainbow"],
    )
    subfig_rainbow.savefig(
        fname=rainbow_file_path,
        format="pdf",
    )


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
