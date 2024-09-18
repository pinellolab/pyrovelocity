from pathlib import Path
from typing import Dict, List, Optional, Tuple

import adjustText
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from beartype import beartype
from beartype.typing import Any, Callable
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.patches import ArrowStyle, ConnectionStyle
from numpy.typing import NDArray
from pandas import DataFrame

from pyrovelocity.analysis.analyze import compute_volcano_data
from pyrovelocity.logging import configure_logging

__all__ = ["plot_gene_ranking"]

logger = configure_logging(__name__)


if hasattr(adjustText, "logger"):
    ajusttext_warn = adjustText.logger.warn

    def filter_adjusttext_matplotlib_warn(message, *args, **kwargs):
        if "Looks like you are using an old matplotlib version" not in message:
            ajusttext_warn(message, *args, **kwargs)

    adjustText.logger.warn = filter_adjusttext_matplotlib_warn


@beartype
def plot_gene_ranking(
    posterior_samples: Dict[str, NDArray[Any] | DataFrame],
    adata: AnnData,
    fig: Optional[FigureBase] = None,
    ax: Optional[Axes] = None,
    gs: Optional[SubplotSpec] = None,
    time_correlation_with: str = "s",
    putative_marker_genes: Optional[List[str]] = None,
    selected_genes: List[str] = [""],
    assemble: bool = False,
    negative: bool = False,
    show_marginal_histograms: bool = False,
    save_volcano_plot: bool = False,
    volcano_plot_path: str | Path = "volcano.pdf",
    defaultfontsize=7,
    show_xy_labels: bool = False,
    truncate_lower_mae_percentile: float = 0.0,
    recompute_volcano_data: bool = False,
) -> Tuple[DataFrame, Optional[FigureBase]]:
    if putative_marker_genes is not None:
        volcano_data: DataFrame = posterior_samples["gene_ranking"]
        genes = putative_marker_genes
    elif "u" in posterior_samples or recompute_volcano_data:
        volcano_data, genes = compute_volcano_data(
            posterior_samples=posterior_samples,
            adata=adata,
            time_correlation_with=time_correlation_with,
            selected_genes=putative_marker_genes,
            negative=negative,
        )
    else:
        volcano_data: DataFrame = posterior_samples["gene_ranking"]
        genes = posterior_samples["genes"]

    volcano_data["percentile"] = volcano_data["mean_mae"].rank(
        pct=True, ascending=False
    )
    volcano_data["inverted_percentile"] = 1 - volcano_data["percentile"]
    volcano_mask = (
        volcano_data["inverted_percentile"] >= truncate_lower_mae_percentile
    )
    adjust_text_compatible = is_adjust_text_compatible()

    defaultdotsize = 3
    plot_title = (
        r"Mean absolute error vs spliced correlation with shared time"
        # r"$-$MAE vs $\rho(\hat{s},t)$"
        # if matplotlib.rcParams["text.usetex"]
        # else "-MAE vs ρ(s,t)"
    )

    if show_marginal_histograms:
        time_corr_hist, time_corr_bins = np.histogram(
            volcano_data["time_correlation"], bins="auto", density=False
        )
        mean_mae_hist, mean_mae_bins = np.histogram(
            a=volcano_data["mean_mae"],
            # a=volcano_data[volcano_mask]["inverted_percentile"],
            bins="auto",
            density=False,
        )

        if gs is None:
            fig = plt.figure(figsize=(10, 10))
            gsi = GridSpec(
                nrows=3,
                ncols=3,
                width_ratios=[2, 2, 1],
                height_ratios=[1, 2, 2],
            )
        else:
            gsi = gs.subgridspec(
                nrows=3,
                ncols=3,
                width_ratios=[2, 2, 1],
                height_ratios=[1, 2, 2],
            )
        ax_scatter = plt.subplot(gsi[1:, :2])
        ax_scatter.set_label("gene_selection")
        ax_hist_x = plt.subplot(gsi[0, :2])
        ax_hist_x.set_label("gene_selection")
        ax_hist_y = plt.subplot(gsi[1:, 2])
        ax_hist_y.set_label("gene_selection")

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
    else:
        if gs is not None:
            gsi = gs.subgridspec(
                nrows=2,
                ncols=1,
                height_ratios=[0.05, 1],
                hspace=0.05,
                wspace=0.0,
            )
            title_ax = fig.add_subplot(gsi[0, :])
            title_ax.axis("off")
            title_ax.set_xticklabels([])
            title_ax.set_label("gene_selection")
            title_ax.text(
                0.5,
                0.5,
                plot_title,
                ha="center",
                va="center",
                fontsize=defaultfontsize + 1,
                fontweight="bold",
                transform=title_ax.transAxes,
            )
            ax = fig.add_subplot(gsi[1, :])
            ax.set_label("gene_selection")
        elif ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_label("gene_selection")
        else:
            ax.set_label("gene_selection")

    ax.set_label("gene_selection")
    sns.scatterplot(
        x="time_correlation",
        y="mean_mae",
        # y="inverted_percentile",
        hue="selected genes",
        data=volcano_data[volcano_mask],
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
    # y_min, y_max = truncate_lower_mae_percentile, 1.0

    padding = 0.1
    x_range = (x_max - x_min) * padding
    y_range = (y_max - y_min) * padding

    ax.set_xlim(x_min - x_range, x_max + x_range)
    ax.set_ylim(y_min - y_range, y_max + y_range)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(labelsize=defaultfontsize - 1)
    ax.tick_params(axis="x", top=False, which="both")
    ax.tick_params(axis="y", right=False, which="both")

    if show_xy_labels:
        ax.set_xlabel(
            "shared time correlation with spliced expression",
            fontsize=defaultfontsize,
        )
        ax.set_ylabel("negative mean absolute error", fontsize=defaultfontsize)
        # ax.set_ylabel(
        #     "mean absolute error percentile", fontsize=defaultfontsize
        # )
    else:
        yticks = ax.get_yticks()
        logger.info(
            f"\nyticks: {yticks}\n",
        )
        ax.set_yticklabels([], fontsize=defaultfontsize)
        ax.tick_params(axis="y", direction="in", pad=-20, labelsize=4)
        # ax.yaxis.set_major_formatter(
        #     ticker.FuncFormatter(
        #         percentile_formatter(truncate_lower_mae_percentile)
        #     )
        # )

    sns.despine()

    texts = []
    light_orange = "#ffb343"
    dark_orange = "#ff6a14"
    for i, g in enumerate(genes):
        ax.scatter(
            volcano_data.loc[g, :].time_correlation,
            # volcano_data.loc[g, :].inverted_percentile,
            volcano_data.loc[g, :].mean_mae,
            s=15,
            color=dark_orange if g in selected_genes else light_orange,
            marker="*",
        )
        new_text = ax.text(
            volcano_data.loc[g, :].time_correlation,
            # volcano_data.loc[g, :].inverted_percentile,
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

    if save_volcano_plot:
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for ext in ["", ".png"]:
            fig.savefig(
                f"{volcano_plot_path}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)

    return volcano_data, fig


@beartype
def percentile_formatter(
    lower_percentile: float, upper_percentile: float = 1.0
) -> Callable:
    def custom_formatter(x, pos):
        if abs(x - lower_percentile) < 1e-6 or abs(x - upper_percentile) < 1e-6:
            return f"{round(abs(1-x),2):.1f}"
        else:
            return ""

    return custom_formatter


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
