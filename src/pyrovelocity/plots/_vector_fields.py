from os import PathLike

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import FigureBase
from matplotlib.ticker import MaxNLocator
from scvelo.plotting.velocity_embedding_grid import default_arrow

from pyrovelocity.analysis.analyze import compute_mean_vector_field
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._time import plot_posterior_time
from pyrovelocity.plots._uncertainty import (
    get_posterior_sample_angle_uncertainty,
)
from pyrovelocity.styles import configure_matplotlib_style

__all__ = [
    "plot_vector_field_summary",
    "plot_vector_field_uncertainty",
    "plot_mean_vector_field",
]

logger = configure_logging(__name__)

configure_matplotlib_style()


@beartype
def create_vector_field_summary_layout(
    fig_width: int | float = 12,
    fig_height: int | float = 2.5,
) -> Tuple[FigureBase, List[Axes], List[Axes]]:
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[1] * 6,
        height_ratios=[6, 1],
    )

    main_axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(6)]
    for ax in bottom_axes:
        ax.axis("off")

    return fig, main_axes, bottom_axes


@beartype
def plot_vector_field_summary(
    adata: AnnData,
    posterior_samples: Dict[str, np.ndarray],
    vector_field_basis: str,
    plot_name: PathLike | str,
    cell_state: str = "cell_type",
    state_color_dict: Optional[Dict[str, str]] = None,
    default_fontsize: int = 12 if matplotlib.rcParams["text.usetex"] else 9,
    default_title_padding: int = 2,
    dotsize: int | float = 3,
    scale: float = 0.35,
    arrow_size: float = 3.6,
    density: float = 0.4,
) -> FigureBase:
    posterior_time = posterior_samples["cell_time"]
    pca_embeds_angle = posterior_samples["pca_embeds_angle"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    (
        fig,
        ax,
        bottom_axes,
    ) = create_vector_field_summary_layout()

    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cell_state].values,
            "X1": adata.obsm[f"X_{vector_field_basis}"][:, 0],
            "X2": adata.obsm[f"X_{vector_field_basis}"][:, 1],
        }
    )

    sns.scatterplot(
        x="X1",
        y="X2",
        hue="cell_type",
        s=dotsize,
        palette=state_color_dict,
        data=ress,
        alpha=0.9,
        linewidth=0,
        edgecolor="none",
        ax=ax[0],
        legend="brief",
    )
    ax[0].get_legend().remove()
    ax[0].axis("off")
    ax[0].set_title(
        "Cell types",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    scv.pl.velocity_embedding_grid(
        adata,
        basis=vector_field_basis,
        fontsize=default_fontsize,
        ax=ax[1],
        title="",
        color="gray",
        s=dotsize,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    ax[1].axis("off")
    ax[1].set_title(
        "scVelo",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=default_fontsize,
        basis=vector_field_basis,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        color="gray",
        s=dotsize,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    ax[2].axis("off")
    ax[2].set_title(
        rf"Pyro\thinspace-Velocity"
        if matplotlib.rcParams["text.usetex"]
        else f"Pyro\u2009-Velocity",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    plot_posterior_time(
        posterior_samples,
        adata,
        ax=ax[3],
        basis=vector_field_basis,
        fig=fig,
        addition=False,
        position="right",
        cmap="winter",
        s=dotsize,
        show_colorbar=False,
        show_titles=False,
        alpha=1,
    )
    ax[3].set_title(
        "shared time",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    pca_cell_angles = pca_embeds_angle / np.pi * 180
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    # cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()
    # cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertainty(
        adata,
        embed_mean,
        cell_time_std,
        ax=ax[4],
        cbar=False,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow_size,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
        show_titles=False,
    )
    ax[4].set_title(
        r"shared time $\sigma$"
        if matplotlib.rcParams["text.usetex"]
        else "shared time σ",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    plot_vector_field_uncertainty(
        adata,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=False,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow_size,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
        show_titles=False,
    )
    ax[5].set_title(
        r"PCA angle $\sigma$"
        if matplotlib.rcParams["text.usetex"]
        else "PCA angle σ",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    handles, labels = ax[0].get_legend_handles_labels()
    bottom_axes[0].legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=(-0.1, -0.2),
        ncol=4,
        frameon=False,
        fancybox=True,
        markerscale=4,
        columnspacing=0.7,
        handletextpad=0.1,
    )

    for axi in ax:
        axi.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.98, bottom=0.08, wspace=0.1, hspace=0.2
    )

    for axi, cax in zip(ax[3:], bottom_axes[3:]):
        cax.axis("on")
        cbar = fig.colorbar(
            mappable=axi.collections[0],
            cax=cax,
            orientation="horizontal",
        )
        cbar.locator = MaxNLocator(nbins=2)
        cbar.update_ticks()
        ax_pos = axi.get_position()
        cbar_width = ax_pos.width * 0.6
        cbar_height = 0.05
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("bottom")
        cax.set_position(
            [
                ax_pos.x0 + (ax_pos.width - cbar_width),
                0.25,
                cbar_width,
                cbar_height,
            ]
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


def plot_vector_field_uncertainty(
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
    color_vector_field_by_measure=False,
    dot_size=1,
    show_titles: bool = True,
    default_fontsize: int = 7,
):
    if uncertain_measure == "angle":
        adata.obs["uncertain"] = get_posterior_sample_angle_uncertainty(
            embeds_radian_or_magnitude / np.pi * 180
        )
    elif uncertain_measure in ["base magnitude", "shared time", "PCA angle"]:
        adata.obs["uncertain"] = embeds_radian_or_magnitude
    else:
        adata.obs["uncertain"] = embeds_radian_or_magnitude.std(axis=0)

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
            if show_titles:
                ax[0].set_title(
                    # f"Single-cell\n {uncertain_measure} uncertainty ",
                    f"{uncertain_measure} uncertainty",
                    fontsize=default_fontsize,
                )
            ax = ax[1]

    if color_vector_field_by_measure:
        X_grid, V_grid, uncertain = project_grid_points(
            adata.obsm[f"X_{basis}"],
            embed_mean,
            adata.obs["uncertain"].values,
            p_mass_min=p_mass_min,
            autoscale=autoscale,
            density=density,
        )

        hl, hw, hal = default_arrow(arrow_size)
        quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
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
        if show_titles:
            ax.set_title(
                f"Averaged\n {uncertain_measure} uncertainty ",
                fontsize=default_fontsize,
            )
        ax.axis("off")
    else:
        order = np.argsort(adata.obs["uncertain"].values)
        ordered_uncertainty_measure = adata.obs["uncertain"].values[order]
        im = ax.scatter(
            adata.obsm[f"X_{basis}"][:, 0][order],
            adata.obsm[f"X_{basis}"][:, 1][order],
            # c=colormap(norm(adata.obs["uncertain"].values[order])),
            c=ordered_uncertainty_measure,
            cmap=cmap,
            norm=None,
            vmin=0
            if "angle" in uncertain_measure
            else np.percentile(ordered_uncertainty_measure, 5),
            vmax=360
            if "angle" in uncertain_measure
            else np.percentile(ordered_uncertainty_measure, 95),
            s=dot_size,
            linewidth=1,
            edgecolors="face",
        )
        ax.axis("off")
        if show_titles:
            ax.set_title(
                # f"Single-cell\n {uncertain_measure} uncertainty ", fontsize=default_fontsize
                f"{uncertain_measure} uncertainty",
                fontsize=default_fontsize,
            )
    if cbar:
        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("bottom", size="5%", pad=0.1)
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal", shrink=0.6)
        ## cbar.ax.set_xticks([0, 180, 360], [0, 180, 360])
        ## fig.colorbar(im, ax=ax, shrink=0.6, location='bottom')

        pos = ax.get_position()
        cax = fig.add_axes(
            [pos.x0 + 0.05, pos.y0 - 0.05, pos.width * 0.6, pos.height / 17]
        )

        cbar = fig.colorbar(
            im, cax=cax, orientation="horizontal"
        )  # fraction=0.046, pad=0.04
        cbar.ax.tick_params(axis="x", labelsize=5.5)
        cbar.ax.locator = MaxNLocator(nbins=2, integer=True)
        # cbar.ax.set_xlabel(f"{uncertain_measure} uncertainty", fontsize=default_fontsize)


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


# TODO: remove unused code
# cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
# cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
# cell_magnitudes_std = cell_magnitudes.std(axis=-2)
# cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
# plot_vector_field_uncertainty(
#     adata,
#     embed_mean,
#     cell_magnitudes_cov,
#     ax=ax[4],
#     cbar=False,
#     fig=fig,
#     basis=vector_field_basis,
#     scale=scale,
#     arrow_size=arrow_size,
#     p_mass_min=1,
#     autoscale=True,
#     density=density,
#     only_grid=False,
#     uncertain_measure="base magnitude",
#     cmap="summer",
#     cmax=None,
#     show_titles=False,
# )
# ax[4].set_title(
#     r"base magnitude $\sigma$"
#     if matplotlib.rcParams["text.usetex"]
#     else "base magnitude σ",
#     fontsize=default_fontsize,
#     pad=default_title_padding,
# )


# def plot_arrow_examples(
#     adata,
#     v_maps,
#     embeds_radian,
#     embed_mean,
#     ax=None,
#     fig=None,
#     cbar=True,
#     basis="umap",
#     n_sample=30,
#     scale=0.0021,
#     alpha=0.02,
#     index=19,
#     index2=0,
#     scale2=0.04,
#     num_certain=3,
#     num_total=4,
#     p_mass_min=1.0,
#     density=0.3,
#     arrow_size=4,
#     customize_uncertain=None,
# ):
#     X_grid, V_grid, uncertain = project_grid_points(
#         adata.obsm[f"X_{basis}"],
#         v_maps,
#         get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180)
#         if customize_uncertain is None
#         else customize_uncertain,
#         p_mass_min=p_mass_min,
#         density=density,
#     )
#     # print(X_grid.shape, V_grid.shape, uncertain.shape)
#     norm = Normalize()
#     norm.autoscale(uncertain)
#     colormap = cm.inferno

#     indexes = np.argsort(uncertain)[::-1][
#         index : (index + num_total - num_certain)
#     ]
#     hl, hw, hal = default_arrow(arrow_size)
#     # print(hl, hw, hal)
#     quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
#     # quiver_kwargs = {"angles": "xy", "scale_units": "width"}
#     quiver_kwargs = {"width": 0.002, "zorder": 0}
#     quiver_kwargs.update({"headlength": hl / 2})
#     quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})

#     ax.scatter(
#         adata.obsm[f"X_{basis}"][:, 0],
#         adata.obsm[f"X_{basis}"][:, 1],
#         s=1,
#         linewidth=0,
#         color="gray",
#         alpha=alpha,
#     )

#     # normalize arrow size the constant
#     V_grid[:, 0] = V_grid[:, 0] / np.sqrt(V_grid[:, 0] ** 2 + V_grid[:, 1] ** 2)
#     V_grid[:, 1] = V_grid[:, 1] / np.sqrt(V_grid[:, 1] ** 2 + V_grid[:, 1] ** 2)

#     for i in range(n_sample):
#         for j in indexes:
#             # ax.quiver(
#             #    X_grid[j, 0],
#             #    X_grid[j, 1],
#             #    embed_mean[j, 0],
#             #    embed_mean[j, 1],
#             #    ec='black',
#             #    scale=scale,
#             #    color=colormap(norm(uncertain))[j],
#             #    **quiver_kwargs,
#             # )
#             ax.quiver(
#                 X_grid[j, 0],
#                 X_grid[j, 1],
#                 V_grid[j][0][i],
#                 V_grid[j][1][i],
#                 ec="face",
#                 norm=Normalize(vmin=0, vmax=360),
#                 scale=scale,
#                 color=colormap(norm(uncertain))[j],
#                 linewidth=0,
#                 alpha=0.3,
#                 **quiver_kwargs,
#             )
#         ax.quiver(
#             X_grid[j, 0],
#             X_grid[j, 1],
#             V_grid[j][0].mean(),
#             V_grid[j][1].mean(),
#             ec="black",
#             alpha=1,
#             norm=Normalize(vmin=0, vmax=360),
#             scale=scale,
#             linewidth=0,
#             color=colormap(norm(uncertain))[j],
#             **quiver_kwargs,
#         )
#     indexes = np.argsort(uncertain)[index2 : (index2 + num_certain)]
#     for i in range(n_sample):
#         for j in indexes:
#             # ax.quiver(
#             #    X_grid[j, 0],
#             #    X_grid[j, 1],
#             #    embed_mean[j, 0],
#             #    embed_mean[j, 1],
#             #    ec='black',
#             #    scale=scale,
#             #    color=colormap(norm(uncertain))[j],
#             #    **quiver_kwargs,
#             # )
#             ax.quiver(
#                 X_grid[j, 0],
#                 X_grid[j, 1],
#                 V_grid[j][0][i],
#                 V_grid[j][1][i],
#                 # ec=colormap(norm(uncertain))[j],
#                 ec="face",
#                 scale=scale2,
#                 alpha=0.3,
#                 linewidth=0,
#                 color=colormap(norm(uncertain))[j],
#                 norm=Normalize(vmin=0, vmax=360),
#                 **quiver_kwargs,
#             )
#         ax.quiver(
#             X_grid[j, 0],
#             X_grid[j, 1],
#             V_grid[j][0].mean(),
#             V_grid[j][1].mean(),
#             ec="black",
#             alpha=1,
#             linewidth=0,
#             norm=Normalize(vmin=0, vmax=360),
#             scale=scale2,
#             color=colormap(norm(uncertain))[j],
#             **quiver_kwargs,
#         )
#     ax.axis("off")
