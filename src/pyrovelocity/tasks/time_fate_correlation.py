from pathlib import Path

import matplotlib.pyplot as plt

from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import plot_lineage_fate_correlation
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import load_anndata_from_path

logger = configure_logging(__name__)


def estimate_time_lineage_fate_correlation(
    reports_path: str | Path = "reports",
):
    data_set_name = "larry_mono"
    model_name = "model2"
    data_set_model_pairing = f"{data_set_name}_{model_name}"
    model_path = f"models/{data_set_model_pairing}"
    adata_pyrovelocity = load_anndata_from_path(
        f"{model_path}/postprocessed.h5ad"
    )
    larry_mono_plot_path = (
        Path(reports_path) / "larry_mono_time_fate_correlation.pdf"
    )
    adata_dynamical = load_anndata_from_path(
        f"data/processed/larry_mono_processed.h5ad"
    )
    adata_cospar = load_anndata_from_path(f"data/external/larry_cospar.h5ad")

    fig = plt.figure(
        figsize=(17, 2.75),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        hspace=0.4, wspace=0.2, left=0.01, right=0.99, top=0.95, bottom=0.3
    )
    ax = fig.subplots(1, 8)

    plot_lineage_fate_correlation(
        posterior_samples_path=f"{model_path}/pyrovelocity.pkl.zst",
        adata_pyrovelocity=adata_pyrovelocity,
        adata_scvelo=adata_dynamical,
        adata_cospar=adata_cospar,
        ax=ax,
        fig=fig,
        state_color_dict=LARRY_CELL_TYPE_COLORS,
        lineage_fate_correlation_path=larry_mono_plot_path,
    )
