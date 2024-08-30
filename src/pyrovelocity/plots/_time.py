from os import PathLike
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from matplotlib.figure import FigureBase
from scipy.stats import spearmanr

from pyrovelocity.plots._common import set_colorbar, set_font_size

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
    im = ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=s,
        alpha=alpha,
        c=adata.obs["cell_time"],
        cmap=cmap,
        linewidth=0,
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
) -> FigureBase:
    cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    cell_time_mean_max = cell_time_mean.max()
    cell_times = posterior_samples["cell_time"] / cell_time_mean_max
    cell_time_mean = cell_times.mean(0).flatten()
    cell_time_std = cell_times.std(0).flatten()
    cell_time_cv = cell_time_std / cell_time_mean
    adata.obs["shared_time_std"] = cell_time_std
    adata.obs["shared_time_mean"] = cell_time_mean
    adata.obs["shared_time_cv"] = cell_time_cv

    cv_string = (
        r"shared time $\left.\sigma \right/ \mu$"
        if matplotlib.rcParams["text.usetex"]
        else "shared time σ/μ"
    )
    mean_string = (
        r"shared time $\mu$"
        if matplotlib.rcParams["text.usetex"]
        else "shared time μ"
    )

    set_font_size(7)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(6, 6)
    ax = ax.flatten()

    ax[0].hist(cell_time_mean, bins=100)
    ax[0].set_title(mean_string)
    ax[2].hist(cell_time_cv, bins=100)
    ax[2].set_title(cv_string)

    ax_st = scv.pl.scatter(
        adata=adata,
        basis=vector_field_basis,
        c="shared_time_mean",
        ax=ax[1],
        show=False,
        cmap="winter",
        fontsize=12,
        colorbar=True,
    )
    ax[1].axis("off")
    ax_st.set_title(mean_string)

    ax_cv = scv.pl.scatter(
        adata=adata,
        basis=vector_field_basis,
        c="shared_time_cv",
        ax=ax[3],
        show=False,
        cmap="winter",
        fontsize=12,
        colorbar=True,
    )
    ax[3].axis("off")
    ax_cv.set_title(cv_string)
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
    fig.tight_layout()
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
