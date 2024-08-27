from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import plot_lineage_fate_correlation
from pyrovelocity.styles import configure_matplotlib_style
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import load_anndata_from_path
from pyrovelocity.workflows.main_configuration import (
    larry_configuration,
    larry_mono_configuration,
    larry_multilineage_configuration,
    larry_neu_configuration,
)

logger = configure_logging(__name__)

configure_matplotlib_style()


def estimate_time_lineage_fate_correlation(
    reports_path: str | Path = "reports",
    model_identifier: str = "model2",
):
    configurations = [
        larry_mono_configuration,
        larry_neu_configuration,
        larry_multilineage_configuration,
        larry_configuration,
    ]

    n_rows = len(configurations)
    n_cols = 8
    width = 14
    height = width * (n_rows / n_cols) + 1

    fig = plt.figure(figsize=(width, height))

    gs = fig.add_gridspec(
        n_rows + 1,
        n_cols + 1,
        width_ratios=[0.02] + [1] * n_cols,
        height_ratios=[1] * n_rows + [0.2],
    )

    adata_cospar = load_anndata_from_path(f"data/external/larry_cospar.h5ad")

    all_axes = []
    for i, config in enumerate(configurations):
        data_set_name = config.download_dataset.data_set_name
        data_set_model_pairing = f"{data_set_name}_{model_identifier}"
        model_path = f"models/{data_set_model_pairing}"

        adata_pyrovelocity = load_anndata_from_path(
            f"{model_path}/postprocessed.h5ad"
        )
        plot_path = (
            Path(reports_path) / f"{data_set_name}_time_fate_correlation.pdf"
        )
        adata_dynamical = load_anndata_from_path(
            f"data/processed/{data_set_name}_processed.h5ad"
        )

        axes = [fig.add_subplot(gs[i, j + 1]) for j in range(n_cols)]
        all_axes.append(axes)

        plot_lineage_fate_correlation(
            posterior_samples_path=f"{model_path}/pyrovelocity.pkl.zst",
            adata_pyrovelocity=adata_pyrovelocity,
            adata_scvelo=adata_dynamical,
            adata_cospar=adata_cospar,
            ax=axes,
            fig=fig,
            state_color_dict=LARRY_CELL_TYPE_COLORS,
            lineage_fate_correlation_path=plot_path,
            ylabel="",
            show_titles=True if i == 0 else False,
            show_colorbars=False,
            default_fontsize=10 if matplotlib.rcParams["text.usetex"] else 9,
        )

    for row_axes in all_axes:
        for ax in row_axes:
            ax.set_aspect("equal", adjustable="box")

    row_labels = ["a", "b", "c", "d"]
    vertical_texts = [
        "Monocytes",
        "Neutrophils",
        "Multilineage",
        "All lineages",
    ]

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
            0.5,
            0.5,
            vtext,
            rotation=90,
            fontsize=12,
            ha="center",
            va="center",
        )

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

    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.98, bottom=0.08, wspace=0.1, hspace=0.2
    )

    combined_plot_path = (
        Path(reports_path)
        / f"combined_time_fate_correlation_{model_identifier}.pdf"
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
