from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Any, Tuple
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureBase
from matplotlib.gridspec import GridSpec, SubplotSpec
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Index

from pyrovelocity.plots._common import set_colorbar, set_font_size
from pyrovelocity.utils import ensure_numpy_array, setup_colors

__all__ = ["rainbowplot", "us_rainbowplot"]


def rainbowplot_module(
    volcano_data: pd.DataFrame,
    adata: AnnData,
    posterior_samples: Dict[str, NDArray[Any]],
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    gs: Optional[SubplotSpec] = None,
    genes: Optional[List[str]] = None,
    data: List[str] = ["st", "ut"],
    cell_state: str = "clusters",
    basis: str = "umap",
    num_genes: int = 5,
    add_line: bool = True,
    negative_correlation: bool = False,
    state_info_colors: bool = False,
    save_plot: bool = False,
    show_data: bool = True,
    rainbow_plot_path: str = "rainbow.pdf",
    dotsize: int = 1,
    default_fontsize: int = 7,
) -> Tuple[Figure, Dict[str, Axes]]:
    set_font_size(default_fontsize)

    if genes is None:
        genes = get_genes(volcano_data, num_genes, negative_correlation)
    number_of_genes = len(genes)

    if gs is None:
        fig, axes_dict = create_rainbow_figure(number_of_genes, show_data)
    else:
        axes_dict = create_rainbow_axes(gs, number_of_genes, show_data)

    if state_info_colors:
        colors = setup_state_info_colors(adata, cell_state, basis)
    else:
        colors = setup_colors(adata, cell_state)

    st, ut = get_posterior_samples_mean(data, posterior_samples)
    if "st_std" in posterior_samples:
        st_std = posterior_samples["st_std"]
    else:
        st_std = None

    for n, gene in enumerate(genes):
        ress = get_data(gene, st, ut, adata, cell_state, posterior_samples)
        plot_gene_data_module(axes_dict, n, ress, colors, add_line, show_data)
        plot_gene_on_embedding(
            axes_dict=axes_dict,
            n=n,
            adata=adata,
            st=st,
            gene=gene,
            basis=basis,
            show_data=show_data,
            st_std=st_std,
            dotsize=dotsize,
        )
        set_labels_module(axes_dict, n, gene, number_of_genes, ress)

    # sns.despine()
    _set_axes_aspect(axes_dict)
    fig.tight_layout()

    if save_plot:
        for ext in ["", ".png"]:
            fig.savefig(
                f"{rainbow_plot_path}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )

    return fig, axes_dict


def create_rainbow_figure(
    number_of_genes: int,
    show_data: bool,
    st_std: bool = True,
) -> Tuple[Figure, Dict[str, Axes]]:
    subplot_height = 0.9
    horizontal_panels = 6 if show_data else 5
    subplot_width = 1.5 * subplot_height * horizontal_panels

    fig = plt.figure(figsize=(subplot_width, subplot_height * number_of_genes))
    gs = GridSpec(
        nrows=number_of_genes,
        ncols=horizontal_panels,
        figure=fig,
        width_ratios=[
            0.21,
            *([1] * (horizontal_panels - 1)),
        ],
        wspace=0.2,
        hspace=0.2,
    )

    axes_dict = {}
    for n in range(number_of_genes):
        axes_dict[f"gene_{n}"] = fig.add_subplot(gs[n, 0])
        axes_dict[f"gene_{n}"].axis("off")
        axes_dict[f"phase_{n}"] = fig.add_subplot(gs[n, 1])
        axes_dict[f"dynamics_{n}"] = fig.add_subplot(gs[n, 2])
        axes_dict[f"predictive_{n}"] = fig.add_subplot(gs[n, 3])
        if show_data:
            axes_dict[f"data_{n}"] = fig.add_subplot(gs[n, 4])
        if st_std:
            axes_dict[f"cv_{n}"] = fig.add_subplot(gs[n, 5])

    return fig, axes_dict


def create_rainbow_axes(
    # fig: Figure,
    # ax: Axes,
    gs_top: SubplotSpec,
    number_of_genes: int,
    show_data: bool,
) -> Dict[str, Axes]:
    # gs = GridSpec(
    #     number_of_genes,
    #     4 if show_data else 3,
    #     figure=fig,
    #     # subplot_spec=ax,
    # )
    n_cols = 6 if show_data else 5
    boundary_column_width = 0.001
    gene_label_column_width = 0.21
    gs = gs_top.subgridspec(
        nrows=number_of_genes,
        ncols=n_cols + 2,
        # figure=fig,
        # subplot_spec=ax,
        # width_ratios=[0.1] + [0.21] + [1] * (n_cols - 1) + [0.1],
        width_ratios=[
            boundary_column_width,
            gene_label_column_width,
            *([1] * (n_cols - 1)),
            boundary_column_width,
        ],
        height_ratios=[1] * number_of_genes,
        wspace=0.2,
        hspace=0.2,
    )
    axs = gs.subplots()

    axes_dict = {}
    plot_label = "rainbow"
    for n in range(number_of_genes):
        # axes_dict[f"phase_{n}"] = fig.add_subplot(gs[n, 0])
        # axes_dict[f"dynamics_{n}"] = fig.add_subplot(gs[n, 1])
        # axes_dict[f"predictive_{n}"] = fig.add_subplot(gs[n, 2])
        # if show_data:
        #     axes_dict[f"data_{n}"] = fig.add_subplot(gs[n, 3])
        axs[n, 0].set_label("buffer_column_left")
        axs[n, 0].axis("off")
        axes_dict[f"gene_{n}"] = axs[n, 1]
        axs[n, 1].set_label(plot_label)
        axs[n, 1].axis("off")
        axes_dict[f"phase_{n}"] = axs[n, 2]
        axs[n, 2].set_label(plot_label)
        axes_dict[f"dynamics_{n}"] = axs[n, 3]
        axs[n, 3].set_label(plot_label)
        axes_dict[f"predictive_{n}"] = axs[n, 4]
        axs[n, 4].set_label(plot_label)
        if show_data:
            axes_dict[f"data_{n}"] = axs[n, 5]
            axs[n, 5].set_label(plot_label)
        axes_dict[f"cv_{n}"] = axs[n, 6]
        axs[n, 6].set_label(plot_label)
        axs[n, -1].set_label("buffer_column_right")
        axs[n, -1].axis("off")

    return axes_dict


def plot_gene_data_module(
    axes_dict: Dict[str, Axes],
    n: int,
    ress: pd.DataFrame,
    colors: Dict,
    add_line: bool,
    show_data: bool,
):
    scatterplot(axes_dict[f"phase_{n}"], ress, colors)
    plot_gene(axes_dict[f"dynamics_{n}"], ress, colors, add_line)


def plot_gene_on_embedding(
    axes_dict: Dict[str, Axes],
    n: int,
    adata: AnnData,
    st: NDArray[Any],
    gene: str,
    basis: str,
    show_data: bool,
    st_std: Optional[NDArray[Any]] = None,
    dotsize: int = 1,
    plot_individual_obs: bool = False,
    gridsize: int = 100,
):
    (index,) = np.where(adata.var_names == gene)

    if plot_individual_obs:
        im = axes_dict[f"predictive_{n}"].scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=dotsize,
            c=st[:, index].flatten(),
            cmap="cividis",
            edgecolors="none",
        )
    else:
        im = axes_dict[f"predictive_{n}"].hexbin(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            C=st[:, index].flatten(),
            gridsize=gridsize,
            cmap="cividis",
            edgecolors="none",
            reduce_C_function=np.mean,
        )

    set_colorbar(
        im,
        axes_dict[f"predictive_{n}"],
        labelsize=5,
        fig=axes_dict[f"predictive_{n}"].figure,
        rainbow=True,
        axes_label="rainbow",
    )
    axes_dict[f"predictive_{n}"].axis("off")

    if st_std is not None:
        if plot_individual_obs:
            im = axes_dict[f"cv_{n}"].scatter(
                adata.obsm[f"X_{basis}"][:, 0],
                adata.obsm[f"X_{basis}"][:, 1],
                s=dotsize,
                c=st_std[:, index].flatten() / st[:, index].flatten(),
                cmap="cividis",
                edgecolors="none",
            )
        else:
            im = axes_dict[f"cv_{n}"].hexbin(
                adata.obsm[f"X_{basis}"][:, 0],
                adata.obsm[f"X_{basis}"][:, 1],
                C=st_std[:, index].flatten() / st[:, index].flatten(),
                gridsize=gridsize,
                cmap="cividis",
                edgecolors="none",
                reduce_C_function=np.mean,
            )

        set_colorbar(
            im,
            axes_dict[f"cv_{n}"],
            labelsize=5,
            fig=axes_dict[f"cv_{n}"].figure,
            rainbow=True,
            axes_label="rainbow",
        )
        axes_dict[f"cv_{n}"].axis("off")

    if show_data:
        if plot_individual_obs:
            im = axes_dict[f"data_{n}"].scatter(
                adata.obsm[f"X_{basis}"][:, 0],
                adata.obsm[f"X_{basis}"][:, 1],
                s=dotsize,
                c=ensure_numpy_array(adata[:, index].X).flatten(),
                cmap="cividis",
                edgecolors="none",
            )
        else:
            im = axes_dict[f"data_{n}"].hexbin(
                adata.obsm[f"X_{basis}"][:, 0],
                adata.obsm[f"X_{basis}"][:, 1],
                C=ensure_numpy_array(adata[:, index].X).flatten(),
                gridsize=gridsize,
                cmap="cividis",
                edgecolors="none",
                reduce_C_function=np.mean,
            )

        set_colorbar(
            im,
            axes_dict[f"data_{n}"],
            labelsize=5,
            fig=axes_dict[f"data_{n}"].figure,
            rainbow=True,
            axes_label="rainbow",
        )
        axes_dict[f"data_{n}"].axis("off")


def set_labels_module(
    axes_dict: Dict[str, Axes],
    n: int,
    gene: str,
    number_of_genes: int,
    ress: pd.DataFrame,
    default_font_size: int | float = 7,
    small_labelpad: int | float = 0.7,
    title_background_color: str = "#F0F0F0",
):
    if n == 0:
        axes_dict[f"phase_{n}"].set_title(
            # label=r"Predictive ($\hat{\mu}(\hat{s}), \hat{\mu}(\hat{u}))$"
            # if matplotlib.rcParams["text.usetex"]
            # else "Predictive samples (μ(s), μ(u))",
            label=r"$(u, s)$ phase space",
            fontsize=default_font_size + 1,
        )
        axes_dict[f"dynamics_{n}"].set_title(
            # label=r"Predictive ($ \hat{\mu}(t), \hat{\mu}(\hat{s}))$"
            # if matplotlib.rcParams["text.usetex"]
            # else "Predictive samples (μ(t), μ(s))",
            label="Spliced dynamics",
            fontsize=default_font_size + 1,
        )
        axes_dict[f"predictive_{n}"].set_title(
            # label=r"Predictive $\hat{\mu}(\hat{s})$"
            # if matplotlib.rcParams["text.usetex"]
            # else "Predictive samples μ(s)",
            label="Predictive spliced",
            fontsize=default_font_size + 1,
            backgroundcolor=title_background_color,
        )
        axes_dict[f"cv_{n}"].set_title(
            # label=r"Predictive $\left.\hat{\sigma}(\hat{s}) \right/ \hat{\mu}(\hat{s})$"
            # if matplotlib.rcParams["text.usetex"]
            # else "Predictive samples σ/μ(s)",
            label=r"Predictive uncertainty",
            fontsize=default_font_size + 1,
            backgroundcolor=title_background_color,
        )
        if f"data_{n}" in axes_dict:
            axes_dict[f"data_{n}"].set_title(
                # label=r"$\log \hat{s}$"
                # if matplotlib.rcParams["text.usetex"]
                # else "log observed s",
                label=r"Observed $\log_{e}$ spliced"
                if matplotlib.rcParams["text.usetex"]
                else "Observed log spliced",
                fontsize=default_font_size + 1,
            )

    if n == number_of_genes - 1:
        axes_dict[f"dynamics_{n}"].set_xlabel(
            xlabel=r"shared time, $\hat{\mu}(t)$",
            loc="left",
            labelpad=small_labelpad,
            fontsize=default_font_size,
        )
        axes_dict[f"dynamics_{n}"].set_ylabel(
            ylabel=r"spliced, $\hat{\mu}(s)$",
            loc="bottom",
            labelpad=small_labelpad,
            fontsize=default_font_size,
        )
        axes_dict[f"dynamics_{n}"].yaxis.set_label_position("right")
        axes_dict[f"phase_{n}"].set_xlabel(
            xlabel=r"spliced, $\hat{\mu}(s)$",
            loc="left",
            labelpad=small_labelpad,
            fontsize=default_font_size,
        )
        axes_dict[f"phase_{n}"].set_ylabel(
            ylabel=r"unspliced, $\hat{\mu}(u)$",
            loc="bottom",
            labelpad=small_labelpad,
            fontsize=default_font_size,
        )
        axes_dict[f"phase_{n}"].yaxis.set_label_position("right")
    else:
        axes_dict[f"dynamics_{n}"].set_xlabel("")
        axes_dict[f"phase_{n}"].set_xlabel("")
        axes_dict[f"dynamics_{n}"].set_ylabel("")
        axes_dict[f"phase_{n}"].set_ylabel("")

    axes_dict[f"gene_{n}"].text(
        x=0.0,
        y=0.5,
        s=gene[:7],
        fontsize=8,
        weight="normal",
        rotation=0,
        va="center",
        ha="center",
        # transform=axes_dict[f"phase_{n}"].transAxes,
    )
    # axes_dict[f"phase_{n}"].set_title(gene, fontsize=7)
    # axes_dict[f"phase_{n}"].text(
    #     x=-0.4,
    #     y=0.5,
    #     s=gene,
    #     fontsize=8,
    #     weight="normal",
    #     rotation=90,
    #     va="center",
    #     ha="right",
    #     transform=axes_dict[f"phase_{n}"].transAxes,
    # )

    set_axis_limits_and_ticks(
        ax_dynamics=axes_dict[f"dynamics_{n}"],
        ax_phase=axes_dict[f"phase_{n}"],
        ress=ress,
        default_font_size=default_font_size,
    )


def set_axis_limits_and_ticks(ax_dynamics, ax_phase, ress, default_font_size=7):
    major_tick_labels_scale_factor = 0.6
    x_ticks = [0, np.power(10, get_closest_pow_of_10(ress["spliced"].max()))]
    y_ticks = [0, np.power(10, get_closest_pow_of_10(ress["unspliced"].max()))]
    ax_phase.set_xticks(x_ticks)
    ax_phase.set_yticks(y_ticks)
    ax_phase.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: "" if pos == 0 else construct_log_string(x)
        )
    )
    ax_phase.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax_phase.tick_params(
        axis="both",
        which="major",
        labelsize=default_font_size * major_tick_labels_scale_factor,
    )

    x_ticks = [0, 1]
    y_ticks = [0, np.power(10, get_closest_pow_of_10(ress["spliced"].max()))]
    ax_dynamics.set_xticks(x_ticks)
    ax_dynamics.set_yticks(y_ticks)
    ax_dynamics.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: "" if pos == 0 else x)
    )
    ax_dynamics.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: construct_log_string(x))
    )
    ax_dynamics.tick_params(
        axis="both",
        which="major",
        labelsize=default_font_size * major_tick_labels_scale_factor,
    )


def _set_axes_aspect(all_axes: Dict[str, Axes]) -> None:
    for k, ax in all_axes.items():
        if "predictive" in k or "data" in k or "cv" in k:
            ax.set_aspect("equal", adjustable="box")
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            x_range = x_max - x_min
            y_range = y_max - y_min

            square_aspect_fraction = 0.65

            ax.set_aspect(
                aspect=(x_range / y_range) * square_aspect_fraction,
                adjustable="box",
            )


# TODO: merge with rainbowplot_module
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
    state_info_colors: bool = False,
    save_plot: bool = False,
    show_data: bool = True,
    rainbow_plot_path: str | Path = "rainbow.pdf",
    default_font_size: int = 7,
) -> FigureBase:
    set_font_size(7)

    if genes is None:
        genes = get_genes(volcano_data, num_genes, negative_correlation)
        genes = genes.tolist()
    number_of_genes = len(genes)

    subplot_height = 1

    if show_data:
        horizontal_panels = 4
    else:
        horizontal_panels = 3

    subplot_width = 2.0 * subplot_height * horizontal_panels

    if fig is None:
        fig, ax = plt.subplots(
            number_of_genes,
            horizontal_panels,
            figsize=(subplot_width, subplot_height * number_of_genes),
        )
    else:
        ax = fig.subplots(number_of_genes, horizontal_panels)

    if state_info_colors:
        colors = setup_state_info_colors(adata, cell_state)
    else:
        colors = setup_colors(adata, cell_state)

    st, ut = get_posterior_samples_mean(data, posterior_samples)

    for n, gene in enumerate(genes):
        ress = get_data(gene, st, ut, adata, cell_state, posterior_samples)
        ax1 = ax[n, 1]
        ax2 = ax[n, 0]
        ax3 = ax[n, 2]
        if show_data:
            ax4 = ax[n, 3]

        if n == 0:
            ax1.set_title("Predictive dynamics", fontsize=default_font_size)
            ax2.set_title("Phase portrait", fontsize=default_font_size)
            ax3.set_title("Predictive spliced", fontsize=default_font_size)
            if show_data:
                ax4.set_title("Log spliced data", fontsize=default_font_size)

        plot_gene(ax1, ress, colors, add_line)
        scatterplot(ax2, ress, colors)
        (index,) = np.where(adata.var_names == gene)
        im = ax3.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            c=st[:, index].flatten(),
            cmap="cividis",
            edgecolors="none",
        )
        set_colorbar(im, ax3, labelsize=5, fig=fig, rainbow=True)
        ax3.axis("off")
        if show_data:
            im = ax4.scatter(
                adata.obsm[f"X_{basis}"][:, 0],
                adata.obsm[f"X_{basis}"][:, 1],
                s=3,
                c=ensure_numpy_array(adata.X[:, index]).flatten(),
                cmap="cividis",
                edgecolors="none",
            )
            set_colorbar(im, ax4, labelsize=5, fig=fig, rainbow=True)
            ax4.axis("off")
        set_labels(ax1, ax2, ax3, gene, number_of_genes, ress, n)

    sns.despine()
    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.5,  # left=0.15, right=0.4, top=0.92, bottom=0.12
    )

    if save_plot:
        for ext in ["", ".png"]:
            fig.savefig(
                f"{rainbow_plot_path}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)
    return fig


def set_subfigure_titles(ax, n, default_font_size):
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
        cmap="cividis",
        edgecolors="none",
    )


def adjust_subfigure(subfig, default_font_size):
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


@beartype
def setup_state_info_colors(
    adata: AnnData,
    cell_state: str,
    basis: str,
):
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


# TODO: remove following refactor to utils
# def setup_colors(adata, cell_state):
#     clusters = adata.obs.loc[:, cell_state]
#     return dict(
#         zip(
#             clusters.cat.categories,
#             sns.color_palette("deep", clusters.cat.categories.shape[0]),
#         )
#     )


def get_posterior_samples_mean(data, posterior_samples) -> NDArray[Any]:
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


def set_labels(ax1, ax2, ax3, gene, ngenes, ress, n, default_font_size=7):
    if n == 0:
        ax3.set_title("Predictive spliced", fontsize=7)
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


def scatterplot(ax, ress, colors):
    sns.scatterplot(
        x="spliced",
        y="unspliced",
        data=ress,
        alpha=0.4,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=colors,
        ax=ax,
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
