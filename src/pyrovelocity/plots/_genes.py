from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from beartype import beartype
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.patches import ArrowStyle
from matplotlib.patches import ConnectionStyle
from numpy import ndarray
from pandas import DataFrame

from pyrovelocity.analysis.analyze import compute_volcano_data
from pyrovelocity.logging import configure_logging


__all__ = ["plot_gene_ranking"]

logger = configure_logging(__name__)


@beartype
def plot_gene_ranking(
    posterior_samples: List[Dict[str, ndarray]],
    adata: List[AnnData],
    ax: Optional[Axes] = None,
    time_correlation_with: str = "s",
    selected_genes: Optional[List[str]] = None,
    assemble: bool = False,
    negative: bool = False,
    show_marginal_histograms: bool = False,
) -> Tuple[DataFrame, Optional[FigureBase]]:
    if selected_genes is not None:
        assert isinstance(selected_genes, (tuple, list))
        assert isinstance(selected_genes[0], str)
        volcano_data = posterior_samples[0]["gene_ranking"]
        genes = selected_genes
    elif "u" in posterior_samples[0]:
        volcano_data, genes = compute_volcano_data(
            posterior_samples,
            adata,
            time_correlation_with,
            selected_genes,
            negative,
        )
    else:
        volcano_data = posterior_samples[0]["gene_ranking"]
        genes = posterior_samples[0]["genes"]

    adjust_text_compatible = is_adjust_text_compatible()
    fig = None

    defaultfontsize = 7
    defaultdotsize = 3
    plot_title = "Pyro-Velocity genes"

    if show_marginal_histograms:
        time_corr_hist, time_corr_bins = np.histogram(
            volcano_data["time_correlation"], bins="auto", density=False
        )
        mean_mae_hist, mean_mae_bins = np.histogram(
            volcano_data["mean_mae"], bins="auto", density=False
        )

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(
            3, 3, width_ratios=[2, 2, 1], height_ratios=[1, 2, 2]
        )
        ax_scatter = plt.subplot(gs[1:, :2])
        ax_hist_x = plt.subplot(gs[0, :2])
        ax_hist_y = plt.subplot(gs[1:, 2])

        # time histogram
        ax_hist_x.bar(
            time_corr_bins[:-1],
            time_corr_hist,
            width=np.diff(time_corr_bins),
            align="edge",
        )

        # MAE histogram
        ax_hist_y.barh(
            mean_mae_bins[:-1],
            mean_mae_hist,
            height=np.diff(mean_mae_bins),
            align="edge",
        )
        ax_hist_x.tick_params(axis="x", labelbottom=False)
        ax_hist_y.tick_params(axis="y", labelleft=False)

        defaultfontsize = 14
        defaultdotsize = 12
        plot_title = ""
        ax = ax_scatter

    sns.scatterplot(
        x="time_correlation",
        y="mean_mae",
        hue="selected genes",
        data=volcano_data,
        s=defaultdotsize,
        linewidth=0,
        ax=ax,
        legend=False,
        alpha=0.3,
    )

    x_min, x_max = (
        volcano_data["time_correlation"].min(),
        volcano_data["time_correlation"].max(),
    )
    y_min, y_max = (
        volcano_data["mean_mae"].min(),
        volcano_data["mean_mae"].max(),
    )

    padding = 0.1
    x_range = (x_max - x_min) * padding
    y_range = (y_max - y_min) * padding

    ax.set_xlim(x_min - x_range, x_max + x_range)
    ax.set_ylim(y_min - y_range, y_max + y_range)

    ax.set_title(plot_title, fontsize=defaultfontsize)
    ax.set_xlabel(
        "shared time correlation\nwith spliced expression",
        fontsize=defaultfontsize,
    )
    ax.set_ylabel("negative mean\nabsolute error", fontsize=defaultfontsize)
    sns.despine()
    ax.tick_params(labelsize=defaultfontsize - 1)

    texts = []
    for i, g in enumerate(genes):
        ax.scatter(
            volcano_data.loc[g, :].time_correlation,
            volcano_data.loc[g, :].mean_mae,
            s=15,
            color="red",
            marker="*",
        )
        new_text = ax.text(
            volcano_data.loc[g, :].time_correlation,
            volcano_data.loc[g, :].mean_mae,
            g,
            fontsize=defaultfontsize - 1,
            color="black",
            ha="center",
            va="center",
        )
        texts.append(new_text)

    if not assemble and adjust_text_compatible:
        adjust_text(
            texts,
            expand=(1.5, 2.5),
            force_text=(0.3, 0.5),
            force_static=(0.2, 0.4),
            ax=ax,
            expand_axes=True,
            arrowprops=dict(
                arrowstyle=ArrowStyle.CurveFilledB(
                    head_length=2,
                    head_width=1.5,
                ),
                color="0.5",
                alpha=0.3,
                connectionstyle=ConnectionStyle.Arc3(rad=0.05),
                shrinkA=1,
                shrinkB=2,
            ),
        )

    return volcano_data, fig


def is_adjust_text_compatible():
    """Check if the current backend supports adjust_text."""
    try:
        plt.figure()
        test_text = plt.text(0.5, 0.5, "test")
        plt.close()
        adjust_text([test_text], autoalign="y")
        return True
    except Exception as e:
        logger.warning(
            f"adjust_text may not be compatible with the current backend: {e}"
        )
        return False
