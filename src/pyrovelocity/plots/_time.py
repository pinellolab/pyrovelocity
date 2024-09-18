from os import PathLike
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureBase
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

from pyrovelocity.plots._common import set_colorbar, set_font_size
from pyrovelocity.utils import quartile_coefficient_of_dispersion

__all__ = [
    "plot_posterior_time",
    "plot_shared_time_uncertainty",
]


def plot_posterior_time(
    posterior_samples,
    adata,
    ax=None,
    fig=None,
    basis="umap",
    addition=True,
    position="left",
    cmap="cividis",
    s=3,
    show_colorbar=True,
    show_titles=True,
    alpha=1,
    plot_individual_obs=False,
):
    if addition:
        sns.set_style("white")
        sns.set_context("paper", font_scale=1)
        set_font_size(7)
        plt.figure()
        plt.hist(posterior_samples["cell_time"].mean(0), bins=100, label="test")
        plt.xlabel("mean of cell time")
        plt.ylabel("frequency")
        if show_titles:
            plt.title("Histogram of cell time posterior samples")
        plt.legend()
    pos_mean_time = posterior_samples["cell_time"].mean(0)
    adata.obs["cell_time"] = pos_mean_time / pos_mean_time.max()

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(2.36, 2)
    if plot_individual_obs:
        im = ax.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=s,
            alpha=alpha,
            c=adata.obs["cell_time"],
            cmap=cmap,
            linewidth=0,
        )
    else:
        im = ax.hexbin(
            x=adata.obsm[f"X_{basis}"][:, 0],
            y=adata.obsm[f"X_{basis}"][:, 1],
            C=adata.obs["cell_time"],
            gridsize=100,
            cmap=cmap,
            alpha=alpha,
            linewidths=0,
            edgecolors="none",
            reduce_C_function=np.mean,  # This will average cell_time values in each hexagon
        )
    if show_colorbar:
        set_colorbar(im, ax, labelsize=5, fig=fig, position=position)
    ax.axis("off")
    if show_titles:
        if "cytotrace" in adata.obs.columns:
            ax.set_title(
                "Pyro-Velocity shared time\ncorrelation with Cytotrace: %.2f"
                % (
                    spearmanr(
                        adata.obs["cell_time"].values,
                        1 - adata.obs.cytotrace.values,
                    )[0]
                ),
                fontsize=7,
            )
        else:
            ax.set_title("Pyro-Velocity shared time\n", fontsize=7)


@beartype
def plot_shared_time_uncertainty(
    adata: AnnData,
    posterior_samples: Dict[str, np.ndarray],
    vector_field_basis: str,
    shared_time_plot: PathLike | str,
    dotsize: Optional[int] = None,
    default_font_size: int = 12,
    duplicate_title: bool = False,
    plot_individual_obs: bool = False,
) -> FigureBase:
    cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    cell_time_mean_max = cell_time_mean.max()
    cell_times = posterior_samples["cell_time"] / cell_time_mean_max
    cell_time_mean = cell_times.mean(0).flatten()
    cell_time_std = cell_times.std(0).flatten()
    cell_time_cv = cell_time_std / cell_time_mean
    cell_time_qcd = quartile_coefficient_of_dispersion(cell_times).flatten()
    adata.obs["shared_time_std"] = cell_time_std
    adata.obs["shared_time_mean"] = cell_time_mean
    adata.obs["shared_time_cv"] = cell_time_cv
    adata.obs["shared_time_qcd"] = cell_time_qcd

    mean_string = (
        r"shared time $\mu$"
        if matplotlib.rcParams["text.usetex"]
        else "shared time μ"
    )
    cv_string = (
        r"shared time $\left.\sigma \right/ \mu$"
        if matplotlib.rcParams["text.usetex"]
        else "shared time σ/μ"
    )
    qcd_string = f"shared time QCD"

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[0.5, 0.5],
        height_ratios=[0.45, 0.45, 0.05],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(cell_time_mean, bins=100)
    ax1.set_title(mean_string)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(cell_time_qcd, bins=100)
    ax2.set_title(qcd_string)

    ax3 = fig.add_subplot(gs[1, 0])
    if plot_individual_obs:
        scv.pl.scatter(
            adata=adata,
            basis=vector_field_basis,
            c="shared_time_mean",
            ax=ax3,
            show=False,
            cmap="winter",
            fontsize=default_font_size,
            colorbar=False,
            s=dotsize,
            title="",
        )
    else:
        im3 = ax3.hexbin(
            x=adata.obsm[f"X_{vector_field_basis}"][:, 0],
            y=adata.obsm[f"X_{vector_field_basis}"][:, 1],
            C=adata.obs["shared_time_mean"],
            gridsize=100,
            cmap="winter",
            linewidths=0,
            edgecolors="none",
            vmin=min(cell_time_mean),
            vmax=max(cell_time_mean),
            reduce_C_function=np.mean,
        )
    ax3.axis("off")
    if duplicate_title:
        ax3.set_title(mean_string)

    ax4 = fig.add_subplot(gs[1, 1])
    if plot_individual_obs:
        scv.pl.scatter(
            adata=adata,
            basis=vector_field_basis,
            c="shared_time_cv",
            ax=ax4,
            show=False,
            cmap="winter",
            fontsize=default_font_size,
            colorbar=False,
            s=dotsize,
            title="",
        )
    else:
        im4 = ax4.hexbin(
            x=adata.obsm[f"X_{vector_field_basis}"][:, 0],
            y=adata.obsm[f"X_{vector_field_basis}"][:, 1],
            # C=adata.obs["shared_time_cv"],
            C=adata.obs["shared_time_qcd"],
            gridsize=100,
            cmap="winter",
            linewidths=0,
            edgecolors="none",
            vmin=min(cell_time_qcd),
            vmax=max(cell_time_qcd),
            reduce_C_function=np.mean,
        )
    ax4.axis("off")
    if duplicate_title:
        ax4.set_title(qcd_string)
    # select = adata.obs["shared_time_cv"] > np.quantile(
    #     adata.obs["shared_time_cv"], 0.9
    # )
    # sns.kdeplot(
    #     x=adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
    #     y=adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
    #     ax=ax[2],
    #     levels=3,
    #     fill=False,
    # )
    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    cbar_ax3 = fig.add_subplot(gs[2, 0])
    _add_colorbar(fig=fig, ax=ax3, cbar_ax=cbar_ax3)
    cbar_ax4 = fig.add_subplot(gs[2, 1])
    _add_colorbar(fig=fig, ax=ax4, cbar_ax=cbar_ax4)

    for ext in ["", ".png"]:
        fig.savefig(
            f"{shared_time_plot}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    plt.close(fig)

    return fig


@beartype
def _add_colorbar(
    fig: Figure,
    ax: Axes,
    cbar_ax: Axes,
    cbar_width_fraction: float = 0.6,
    cbar_height: float = 0.02,
) -> Axes:
    im = ax.collections[0]
    cbar = fig.colorbar(mappable=im, cax=cbar_ax, orientation="horizontal")
    cbar.locator = MaxNLocator(nbins=2)
    cbar.update_ticks()
    cbar_ax.xaxis.set_ticks_position("bottom")
    cbar_ax.xaxis.set_label_position("bottom")
    ax3_pos = ax.get_position()
    cbar_width = ax3_pos.width * cbar_width_fraction
    cbar_ax.set_position(
        [
            ax3_pos.x0 + (ax3_pos.width - cbar_width),
            ax3_pos.y0 - cbar_height,
            cbar_width,
            cbar_height,
        ]
    )
    return cbar_ax
