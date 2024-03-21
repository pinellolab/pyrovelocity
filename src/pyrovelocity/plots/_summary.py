from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
from anndata import AnnData
from beartype import beartype
from matplotlib.figure import FigureBase
from numpy import ndarray

from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._genes import plot_gene_ranking
from pyrovelocity.plots._rainbow import rainbowplot
from pyrovelocity.plots._time import plot_posterior_time


logger = configure_logging(__name__)


__all__ = ["plot_gene_selection_summary"]


@beartype
def plot_gene_selection_summary(
    adata: AnnData,
    posterior_samples: Dict[str, ndarray],
    plot_name: str | Path = "",
    basis: str = "",
    cell_state: str = "",
    show_marginal_histograms: bool = False,
    selected_genes: Optional[List[str]] = None,
    number_of_genes: int = 4,
) -> FigureBase:
    fig = plt.figure(figsize=(9.5, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.8, 4])
    ax = subfigs[0].subplots(2, 1)

    plot_posterior_time(
        posterior_samples,
        adata,
        ax=ax[0],
        fig=subfigs[0],
        addition=False,
        basis=basis,
    )

    volcano_data, _ = plot_gene_ranking(
        posterior_samples=[posterior_samples],
        adata=[adata],
        ax=ax[1],
        time_correlation_with="st",
        selected_genes=selected_genes,
        show_marginal_histograms=show_marginal_histograms,
    )

    _ = rainbowplot(
        volcano_data,
        adata,
        posterior_samples,
        subfigs[1],
        data=["st", "ut"],
        basis=basis,
        cell_state=cell_state,
        num_genes=number_of_genes,
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{plot_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    plt.close(fig)

    return fig
