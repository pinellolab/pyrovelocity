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

from pyrovelocity.plots._common import set_colorbar
from pyrovelocity.plots._common import set_font_size


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
    s=3,
):
    if addition:
        sns.set_style("white")
        sns.set_context("paper", font_scale=1)
        set_font_size(7)
        plt.figure()
        plt.hist(posterior_samples["cell_time"].mean(0), bins=100, label="test")
        plt.xlabel("mean of cell time")
        plt.ylabel("frequency")
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
        alpha=0.4,
        c=adata.obs["cell_time"],
        cmap="inferno",
        linewidth=0,
    )
    set_colorbar(im, ax, labelsize=5, fig=fig, position=position)
    ax.axis("off")
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
    posterior_samples: Dict[str, np.ndarray],
    adata: AnnData,
    vector_field_basis: str,
    shared_time_plot: PathLike | str,
) -> FigureBase:
    cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    cell_time_std = posterior_samples["cell_time"].std(0).flatten()
    adata.obs["shared_time_std"] = cell_time_std
    adata.obs["shared_time_mean"] = cell_time_mean

    set_font_size(7)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)
    ax = ax.flatten()

    ax[0].hist(cell_time_std / cell_time_mean, bins=100)
    ax[0].set_title("histogram of shared time CoV")
    ax_st = scv.pl.scatter(
        adata,
        c="shared_time_mean",
        ax=ax[1],
        show=False,
        cmap="inferno",
        fontsize=12,
        colorbar=True,
    )
    ax_cv = scv.pl.scatter(
        adata,
        c="shared_time_std",
        ax=ax[2],
        show=False,
        cmap="winter",
        fontsize=12,
        colorbar=True,
        title="shared time standard deviation",
    )
    ax_cv.set_xlabel("density estimate over 90th %")
    select = adata.obs["shared_time_std"] > np.quantile(
        adata.obs["shared_time_std"], 0.9
    )
    sns.kdeplot(
        x=adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
        y=adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
        ax=ax[2],
        levels=3,
        fill=False,
    )
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
