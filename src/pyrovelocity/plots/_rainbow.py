from typing import Dict
from typing import List
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.figure import FigureBase
from numpy import ndarray
from pandas import DataFrame
from pandas import Index

from pyrovelocity.plots._common import set_colorbar
from pyrovelocity.plots._common import set_font_size


__all__ = ["rainbowplot", "us_rainbowplot"]


@beartype
def rainbowplot(
    volcano_data: DataFrame,
    adata: AnnData,
    posterior_samples: Dict[str, ndarray],
    fig: Optional[FigureBase] = None,
    genes: Optional[List[str]] = None,
    data: List[str] = ["st", "ut"],
    cell_state: str = "clusters",
    basis: str = "umap",
    num_genes: int = 5,
    add_line: bool = True,
    negative_correlation: bool = False,
    scvelo_colors: bool = False,
) -> FigureBase:
    set_font_size(7)

    if genes is None:
        genes = get_genes(volcano_data, num_genes, negative_correlation)
        genes = genes.tolist()
    number_of_genes = len(genes)

    subplot_height = 1
    subplot_width = subplot_height * 2.0 * 3

    if fig is None:
        fig, ax = plt.subplots(
            number_of_genes,
            3,
            figsize=(subplot_width, subplot_height * number_of_genes),
        )
    else:
        ax = fig.subplots(number_of_genes, 3)

    if scvelo_colors:
        colors = setup_scvelo_colors(adata, cell_state)
    else:
        colors = setup_colors(adata, cell_state)

    st, ut = get_posterior_samples(data, posterior_samples)

    for n, gene in enumerate(genes):
        ress = get_data(gene, st, ut, adata, cell_state, posterior_samples)
        ax1 = ax[n, 1]
        ax2 = ax[n, 0]
        ax3 = ax[n, 2]
        if n == 0:
            ax1.set_title("Rainbow plot", fontsize=7)
            ax2.set_title("Phase portrait", fontsize=7)
            ax3.set_title("Denoised spliced", fontsize=7)
        plot_gene(ax1, ress, colors, add_line)
        scatterplot(ax2, ress, colors)
        (index,) = np.where(adata.var_names == gene)
        im = ax3.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            c=st[:, index].flatten(),
            cmap="RdBu_r",
        )
        set_colorbar(im, ax3, labelsize=5, fig=fig, rainbow=True)
        ax3.axis("off")
        set_labels(ax1, ax2, ax3, gene, number_of_genes, ress, n)

    sns.despine()
    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.5,  # left=0.15, right=0.4, top=0.92, bottom=0.12
    )
    return fig


def set_subfigure_titles(ax, n):
    if n == 0:
        ax[0].set_title("Rainbow plot", fontsize=7)
        ax[1].set_title("Phase portrait", fontsize=7)


def plot_gene_data(ax, ress, colors, add_line):
    plot_gene(ax[1], ress, colors, add_line)
    scatterplot(ax[0], ress, colors)


def scatter_gene(ax, adata, st, gene, basis):
    (index,) = np.where(adata.var_names == gene)
    return ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=3,
        c=st[:, index].flatten(),
        cmap="RdBu_r",
    )


def adjust_subfigure(subfig):
    subfig[0].subplots_adjust(
        hspace=0.8, wspace=1.4, left=0.32, right=0.94, top=0.92, bottom=0.12
    )
    subfig[1].subplots_adjust(
        hspace=0.8, wspace=0.4, left=0.2, right=0.7, top=0.92, bottom=0.08
    )

    subfig[0].text(
        -0.025,
        0.58,
        "unspliced expression",
        size=7,
        rotation="vertical",
        va="center",
    )
    subfig[0].text(
        0.552,
        0.58,
        "spliced expression",
        size=7,
        rotation="vertical",
        va="center",
    )

    sns.despine()


@beartype
def get_genes(
    volcano_data: DataFrame,
    num_genes: int,
    negative_correlation: bool,
) -> Index:
    return (
        volcano_data.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=negative_correlation)
        .head(num_genes)
        .index
    )


def setup_scvelo_colors(adata, cell_state, basis):
    scv.pl.scatter(
        adata,
        basis=basis,
        fontsize=7,
        legend_loc="on data",
        legend_fontsize=7,
        color=cell_state,
        show=False,
    )
    return dict(
        zip(adata.obs.state_info.cat.categories, adata.uns["state_info_colors"])
    )


def setup_colors(adata, cell_state):
    clusters = adata.obs.loc[:, cell_state]
    return dict(
        zip(
            clusters.cat.categories,
            sns.color_palette("deep", clusters.cat.categories.shape[0]),
        )
    )


def get_posterior_samples(data, posterior_samples):
    if (data[0] in posterior_samples) and (data[1] in posterior_samples):
        st = posterior_samples[data[0]].mean(0).squeeze()
        ut = posterior_samples[data[1]].mean(0).squeeze()
    else:
        st = posterior_samples["st_mean"]
        ut = posterior_samples["ut_mean"]
    return st, ut


def get_data(gene, st, ut, adata, cell_state, posterior_samples):
    (index,) = np.where(adata.var_names == gene)
    pos_mean_time = posterior_samples["cell_time"].mean(0).flatten()
    ress = pd.DataFrame(
        {
            "cell_time": pos_mean_time / pos_mean_time.max(),
            "cell_type": adata.obs[cell_state].values,
            "spliced": st[:, index].flatten(),
            "unspliced": ut[:, index].flatten(),
        }
    )
    return ress.sort_values("cell_time")


def plot_gene(ax1, ress, colors, add_line):
    sns.scatterplot(
        x="cell_time",
        y="spliced",
        data=ress,
        alpha=0.1,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=colors,
        ax=ax1,
        marker="o",
        legend=False,
        s=5,
    )
    if add_line:
        for row in range(ress.shape[0]):
            ax1.vlines(
                x=ress.cell_time[row],
                ymin=0,
                ymax=ress.spliced[row],
                colors=colors[ress.cell_type[row]],
                alpha=0.1,
            )


def set_labels(ax1, ax2, ax3, gene, ngenes, ress, n):
    if n == 0:
        ax3.set_title("Denoised spliced", fontsize=7)
    if n == ngenes - 1:
        ax1.set_xlabel("shared time", fontsize=7)
        ax1.set_ylabel("expression", fontsize=7)
        ax2.set_xlabel("spliced", fontsize=7)
        ax2.set_ylabel("uspliced", fontsize=7)
    else:
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax1.set_ylabel("")
        ax2.set_ylabel("")
    ax2.text(
        -0.5,
        0.5,
        gene,
        fontsize=8,
        weight="normal",
        rotation=0,
        va="center",
        ha="right",
        transform=ax2.transAxes,
    )

    x_ticks = [0, np.power(10, get_closest_pow_of_10(ress["cell_time"].max()))]
    y_ticks = [0, np.power(10, get_closest_pow_of_10(ress["spliced"].max()))]
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax1.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax1.tick_params(axis="both", which="major", labelsize=7)

    x_ticks = [0, np.power(10, get_closest_pow_of_10(ress["spliced"].max()))]
    y_ticks = [0, np.power(10, get_closest_pow_of_10(ress["unspliced"].max()))]
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax2.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax2.tick_params(axis="both", which="major", labelsize=7)


def get_closest_pow_of_10(value):
    return np.floor(np.log10(value))


def construct_log_string(x):
    str_val = f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "0"
    return rf"{str_val}"


def scatterplot(ax2, ress, colors):
    sns.scatterplot(
        x="spliced",
        y="unspliced",
        data=ress,
        alpha=0.4,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=colors,
        ax=ax2,
        marker="o",
        legend=False,
        s=3,
    )


def us_rainbowplot(
    genes: pd.Index,
    adata: AnnData,
    posterior_samples: Dict[str, ndarray],
    data: List[str] = ["st", "ut"],
    cell_state: str = "clusters",
) -> Figure:
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(len(genes), 2)
    fig.set_size_inches(7, 14)
    n = 0
    if data[0] in posterior_samples:
        pos_s = posterior_samples[data[0]].mean(0).squeeze()
        pos_u = posterior_samples[data[1]].mean(0).squeeze()
    else:
        pos_u = posterior_samples["ut_mean"]
        pos_s = posterior_samples["st_mean"]

    for gene in genes:
        (index,) = np.where(adata.var_names == gene)
        ax1 = ax[n, 1]
        if n == 0:
            ax1.set_title("Rainbow plot")
        ress = pd.DataFrame(
            {
                "cell_time": posterior_samples["cell_time"].mean(0).squeeze(),
                "cell_type": adata.obs[cell_state].values,
                "spliced": pos_s[:, index].squeeze(),
                "unspliced": pos_u[:, index].squeeze(),
            }
        )
        if n == 2:
            sns.scatterplot(
                x="cell_time",
                y="spliced",
                data=ress,
                alpha=0.4,
                linewidth=0,
                edgecolor="none",
                hue="cell_type",
                ax=ax1,
                marker="o",
                legend="brief",
                palette="bright",
                s=10,
            )
        else:
            sns.scatterplot(
                x="cell_time",
                y="spliced",
                data=ress,
                alpha=0.4,
                linewidth=0,
                edgecolor="none",
                palette="bright",
                hue="cell_type",
                ax=ax1,
                marker="o",
                legend="brief",
                s=10,
            )

        ax2 = ax[n, 0]
        ax2.set_title(gene)
        ax2.set_ylabel("")
        ax2.set_xlabel("")
        sns.scatterplot(
            x="spliced",
            y="unspliced",
            data=ress,
            alpha=0.4,
            s=25,
            edgecolor="none",
            hue="cell_type",
            ax=ax2,
            legend=False,
            marker="*",
            palette="bright",
        )
        if n == 3:
            blue_star = mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=5,
                label="Spliced",
            )
            red_square = mlines.Line2D(
                [],
                [],
                color="black",
                marker="+",
                linestyle="None",
                markersize=5,
                label="Unspliced",
            )
            ax1.legend(
                handles=[blue_star, red_square], bbox_to_anchor=[2, -0.03]
            )
            ax1.set_xlabel("")
        n += 1
        ax1.legend(bbox_to_anchor=[2, 0.1])
        ax1.tick_params(labelbottom=True)
        ax2.set_xlabel("spliced")
        ax2.set_title(gene)

    plt.subplots_adjust(hspace=0.8, wspace=0.6, left=0.1, right=0.91)
    return fig
