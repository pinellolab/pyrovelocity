from pathlib import Path

import matplotlib.pyplot as plt

from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import plot_lineage_fate_correlation
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import load_anndata_from_path
from pyrovelocity.workflows.main_configuration import (
    larry_configuration,
    larry_mono_configuration,
    larry_multilineage_configuration,
    larry_neu_configuration,
)

logger = configure_logging(__name__)


def estimate_time_lineage_fate_correlation(
    reports_path: str | Path = "reports",
):
    configurations = [
        larry_mono_configuration,
        larry_neu_configuration,
        larry_multilineage_configuration,
        larry_configuration,
    ]

    n_rows = len(configurations)
    n_cols = 8
    width = 12
    height = width * (n_rows / n_cols)
    fig = plt.figure(
        figsize=(width, height),
        constrained_layout=True,
    )
    all_axes = fig.subplots(len(configurations), 8)
    for ax in all_axes.flat:
        ax.set_aspect("equal", adjustable="box")
    # fig.subplots_adjust(
    #     left=0.01,
    #     bottom=0.3,
    #     # bottom=0.1,
    #     right=0.99,
    #     top=0.95,
    #     # top=0.15,
    #     hspace=0.4,
    #     # hspace=0.2,
    #     wspace=0.2,
    # )

    model_name = "model2"
    adata_cospar = load_anndata_from_path(f"data/external/larry_cospar.h5ad")

    for i, config in enumerate(configurations):
        data_set_name = config.download_dataset.data_set_name
        data_set_model_pairing = f"{data_set_name}_{model_name}"
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

        axes = all_axes[i] if len(configurations) > 1 else all_axes

        plot_lineage_fate_correlation(
            posterior_samples_path=f"{model_path}/pyrovelocity.pkl.zst",
            adata_pyrovelocity=adata_pyrovelocity,
            adata_scvelo=adata_dynamical,
            adata_cospar=adata_cospar,
            ax=axes,
            fig=fig,
            state_color_dict=LARRY_CELL_TYPE_COLORS,
            lineage_fate_correlation_path=plot_path,
            ylabel=f"{data_set_name}",
            show_titles=True if i == 0 else False,
            show_colorbars=True if i == (len(configurations) - 1) else False,
        )

    combined_plot_path = (
        Path(reports_path) / "combined_time_fate_correlation.pdf"
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
