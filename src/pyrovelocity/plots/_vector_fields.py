from os import PathLike
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import FigureBase
from matplotlib.ticker import MaxNLocator
from scvelo.plotting.velocity_embedding_grid import default_arrow

from pyrovelocity.analysis.analyze import compute_mean_vector_field
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._uncertainty import (
    get_posterior_sample_angle_uncertainty,
)


logger = configure_logging(__name__)


__all__ = [
    "plot_vector_field_summary",
    "plot_vector_field_uncertain",
    "plot_mean_vector_field",
    "plot_arrow_examples",
]


@beartype
def plot_vector_field_summary(
    adata: AnnData,
    posterior_samples: Dict[str, np.ndarray],
    vector_field_basis: str,
    plot_name: PathLike | str,
    cell_state: str = "cell_type",
) -> FigureBase:
    # posterior_vector_field = posterior_samples["vector_field_posterior_samples"]
    posterior_time = posterior_samples["cell_time"]
    cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    pca_embeds_angle = posterior_samples["pca_embeds_angle"]
    # embed_radians = posterior_samples["embeds_angle"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    dot_size = 3.5
    font_size = 6.5
    scale = 0.35
    scale_high = 7.8
    scale_low = 7.8

    arrow = 3.6
    density = 0.4
    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cell_state].values,
            "X1": adata.obsm[f"X_{vector_field_basis}"][:, 0],
            "X2": adata.obsm[f"X_{vector_field_basis}"][:, 1],
        }
    )
    fig = plt.figure(figsize=(9.6, 2), constrained_layout=False)
    fig.subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    ax = fig.subplots(1, 6)
    pos = ax[0].get_position()

    sns.scatterplot(
        x="X1",
        y="X2",
        data=ress,
        alpha=0.9,
        s=dot_size,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        ax=ax[0],
        legend="brief",
    )
    ax[0].axis("off")
    ax[0].set_title("Cell types\n", fontsize=font_size)
    ax[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.48),
        ncol=5,
        fancybox=True,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    kwargs = dict(
        color="gray",
        s=dot_size,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        basis=vector_field_basis,
        fontsize=font_size,
        ax=ax[1],
        title="",
        **kwargs,
    )
    ax[1].set_title("Scvelo\n", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=font_size,
        basis=vector_field_basis,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    pca_cell_angles = pca_embeds_angle / np.pi * 180  # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_time_std,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
    )

    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="base magnitude",
        cmap="summer",
        cmax=None,
    )

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
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


def plot_vector_field_uncertain(
    adata,
    embed_mean,
    embeds_radian_or_magnitude,
    fig=None,
    cbar=True,
    basis="umap",
    scale=0.002,
    cbar_pos=[0.22, 0.28, 0.5, 0.05],
    p_mass_min=3.5,
    only_grid=False,
    ax=None,
    autoscale=False,
    density=0.3,
    arrow_size=5,
    uncertain_measure="angle",
    cmap="winter",
    cmax=0.305,
):
    if cmap == "inferno":
        colormap = cm.inferno
    elif cmap == "summer":
        colormap = cm.summer
    else:
        colormap = cm.winter

    # print(adata.shape)
    # print(embeds_radian_or_magnitude.shape)

    if uncertain_measure == "angle":
        adata.obs["uncertain"] = get_posterior_sample_angle_uncertainty(
            embeds_radian_or_magnitude / np.pi * 180
        )
    else:
        adata.obs["uncertain"] = embeds_radian_or_magnitude.std(axis=0)

    if uncertain_measure in ["base magnitude", "shared time", "PCA angle"]:
        adata.obs["uncertain"] = embeds_radian_or_magnitude

    dot_size = 1
    # plt.rcParams["image.cmap"] = "winter"
    if ax is None:
        ax = fig.subplots(1, 2)
    if isinstance(ax, list) and len(ax) == 2:
        if not only_grid:
            # norm = Normalize()
            # norm.autoscale(adata.obs["uncertain"])
            order = np.argsort(adata.obs["uncertain"].values)
            im = ax[0].scatter(
                adata.obsm[f"X_{basis}"][:, 0][order],
                adata.obsm[f"X_{basis}"][:, 1][order],
                # c=colormap(norm(adata.obs["uncertain"].values[order])),
                c=adata.obs["uncertain"].values[order],
                cmap=cmap,
                norm=None,
                vmin=np.percentile(uncertain, 5),
                vmax=np.percentile(uncertain, 95),
                s=dot_size,
                linewidth=1,
                edgecolors="face",
            )
            ax[0].axis("off")
            ax[0].set_title(
                f"Single-cell\n {uncertain_measure} uncertainty ", fontsize=7
            )
            ax = ax[1]

    X_grid, V_grid, uncertain = project_grid_points(
        adata.obsm[f"X_{basis}"],
        embed_mean,
        adata.obs["uncertain"].values,
        p_mass_min=p_mass_min,
        autoscale=autoscale,
        density=density,
    )

    # scale = None
    hl, hw, hal = default_arrow(arrow_size)
    quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
    # quiver_kwargs = {"angles": "xy", "scale_units": "width"}
    quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
    quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})
    quiver_kwargs.update({"linewidth": 1, "zorder": 3})
    norm = Normalize()
    norm.autoscale(uncertain)
    ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=1,
        linewidth=0,
        color="gray",
        alpha=0.22,
    )
    im = ax.quiver(
        X_grid[:, 0],
        X_grid[:, 1],
        V_grid[:, 0],
        V_grid[:, 1],
        uncertain,
        norm=None,
        cmap=cmap,
        edgecolors="face",
        scale=scale,
        clim=(
            np.percentile(uncertain, 5),
            np.percentile(uncertain, 95) if cmax is None else cmax,
        ),
        **quiver_kwargs,
    )
    ax.set_title(f"Averaged\n {uncertain_measure} uncertainty ", fontsize=7)
    ax.axis("off")
    if cbar:
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('bottom', size='5%', pad=0.1)
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal", shrink=0.6)
        ### cbar.ax.set_xticks([0, 180, 360], [0, 180, 360])
        ##fig.colorbar(im, ax=ax, shrink=0.6, location='bottom')
        pos = ax.get_position()
        cbar_ax = fig.add_axes(
            [pos.x0 + 0.05, pos.y0 - 0.02, pos.width * 0.6, pos.height / 17]
        )
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation="horizontal"
        )  # fraction=0.046, pad=0.04
        cbar.ax.tick_params(axis="x", labelsize=5.5)
        cbar.ax.locator = MaxNLocator(nbins=2, integer=True)
    # cbar.ax.set_xlabel(f"{uncertain_measure} uncertainty", fontsize=7)


def plot_mean_vector_field(
    posterior_samples,
    adata,
    ax,
    basis="umap",
    n_jobs=1,
    scale=0.2,
    density=0.4,
    spliced="spliced_pyro",
    raw=False,
):
    compute_mean_vector_field(
        posterior_samples=posterior_samples,
        adata=adata,
        basis=basis,
        n_jobs=n_jobs,
        spliced=spliced,
        raw=raw,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        basis=basis,
        vkey="velocity_pyro",
        linewidth=1,
        ax=ax,
        show=False,
        legend_loc="on data",
        density=density,
        scale=scale,
        arrow_size=3,
    )
    return adata.obsm[f"velocity_pyro_{basis}"]


# def project_grid_points(emb, velocity_emb, uncertain=None, p_mass_min=3.5, density=0.3):
def project_grid_points(
    emb,
    velocity_emb,
    uncertain=None,
    p_mass_min=1.0,
    density=0.3,
    autoscale=False,
):
    from scipy.stats import norm as normal
    from scvelo.tools.velocity_embedding import quiver_autoscale
    from sklearn.neighbors import NearestNeighbors

    X_grid = []
    grs = []
    grid_num = 50 * density
    smooth = 0.5

    for dim_i in range(2):
        m, M = np.min(emb[:, dim_i]), np.max(emb[:, dim_i])
        # m = m - .025 * np.abs(M - m)
        # M = M + .025 * np.abs(M - m)
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num))
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    n_neighbors = int(emb.shape[0] / 50)
    # print(n_neighbors)
    # nn = NearestNeighbors(n_neighbors=30, n_jobs=-1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(emb)
    dists, neighs = nn.kneighbors(X_grid)
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    if len(velocity_emb.shape) == 2:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None]
    else:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None, None]
    # print(V_grid.shape)

    p_mass_min *= np.percentile(p_mass, 99) / 100
    if autoscale:
        V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    if uncertain is None:
        return X_grid[p_mass > p_mass_min], V_grid[p_mass > p_mass_min]
    else:
        uncertain = (uncertain[neighs] * weight).sum(1) / np.maximum(1, p_mass)
        return (
            X_grid[p_mass > p_mass_min],
            V_grid[p_mass > p_mass_min],
            uncertain[p_mass > p_mass_min],
        )


def plot_arrow_examples(
    adata,
    v_maps,
    embeds_radian,
    embed_mean,
    ax=None,
    fig=None,
    cbar=True,
    basis="umap",
    n_sample=30,
    scale=0.0021,
    alpha=0.02,
    index=19,
    index2=0,
    scale2=0.04,
    num_certain=3,
    num_total=4,
    p_mass_min=1.0,
    density=0.3,
    arrow_size=4,
    customize_uncertain=None,
):
    X_grid, V_grid, uncertain = project_grid_points(
        adata.obsm[f"X_{basis}"],
        v_maps,
        get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180)
        if customize_uncertain is None
        else customize_uncertain,
        p_mass_min=p_mass_min,
        density=density,
    )
    # print(X_grid.shape, V_grid.shape, uncertain.shape)
    norm = Normalize()
    norm.autoscale(uncertain)
    colormap = cm.inferno

    indexes = np.argsort(uncertain)[::-1][
        index : (index + num_total - num_certain)
    ]
    hl, hw, hal = default_arrow(arrow_size)
    # print(hl, hw, hal)
    quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
    # quiver_kwargs = {"angles": "xy", "scale_units": "width"}
    quiver_kwargs = {"width": 0.002, "zorder": 0}
    quiver_kwargs.update({"headlength": hl / 2})
    quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})

    ax.scatter(
        adata.obsm[f"X_{basis}"][:, 0],
        adata.obsm[f"X_{basis}"][:, 1],
        s=1,
        linewidth=0,
        color="gray",
        alpha=alpha,
    )

    # normalize arrow size the constant
    V_grid[:, 0] = V_grid[:, 0] / np.sqrt(V_grid[:, 0] ** 2 + V_grid[:, 1] ** 2)
    V_grid[:, 1] = V_grid[:, 1] / np.sqrt(V_grid[:, 1] ** 2 + V_grid[:, 1] ** 2)

    for i in range(n_sample):
        for j in indexes:
            # ax.quiver(
            #    X_grid[j, 0],
            #    X_grid[j, 1],
            #    embed_mean[j, 0],
            #    embed_mean[j, 1],
            #    ec='black',
            #    scale=scale,
            #    color=colormap(norm(uncertain))[j],
            #    **quiver_kwargs,
            # )
            ax.quiver(
                X_grid[j, 0],
                X_grid[j, 1],
                V_grid[j][0][i],
                V_grid[j][1][i],
                ec="face",
                norm=Normalize(vmin=0, vmax=360),
                scale=scale,
                color=colormap(norm(uncertain))[j],
                linewidth=0,
                alpha=0.3,
                **quiver_kwargs,
            )
        ax.quiver(
            X_grid[j, 0],
            X_grid[j, 1],
            V_grid[j][0].mean(),
            V_grid[j][1].mean(),
            ec="black",
            alpha=1,
            norm=Normalize(vmin=0, vmax=360),
            scale=scale,
            linewidth=0,
            color=colormap(norm(uncertain))[j],
            **quiver_kwargs,
        )
    indexes = np.argsort(uncertain)[index2 : (index2 + num_certain)]
    for i in range(n_sample):
        for j in indexes:
            # ax.quiver(
            #    X_grid[j, 0],
            #    X_grid[j, 1],
            #    embed_mean[j, 0],
            #    embed_mean[j, 1],
            #    ec='black',
            #    scale=scale,
            #    color=colormap(norm(uncertain))[j],
            #    **quiver_kwargs,
            # )
            ax.quiver(
                X_grid[j, 0],
                X_grid[j, 1],
                V_grid[j][0][i],
                V_grid[j][1][i],
                # ec=colormap(norm(uncertain))[j],
                ec="face",
                scale=scale2,
                alpha=0.3,
                linewidth=0,
                color=colormap(norm(uncertain))[j],
                norm=Normalize(vmin=0, vmax=360),
                **quiver_kwargs,
            )
        ax.quiver(
            X_grid[j, 0],
            X_grid[j, 1],
            V_grid[j][0].mean(),
            V_grid[j][1].mean(),
            ec="black",
            alpha=1,
            linewidth=0,
            norm=Normalize(vmin=0, vmax=360),
            scale=scale2,
            color=colormap(norm(uncertain))[j],
            **quiver_kwargs,
        )
    ax.axis("off")
