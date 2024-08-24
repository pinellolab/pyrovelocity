from pathlib import Path

import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype.typing import Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import distance
from scipy.stats import spearmanr

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots import (
    align_trajectory_diff,
    get_clone_trajectory,
    get_posterior_sample_angle_uncertainty,
    plot_posterior_time,
    plot_vector_field_uncertain,
)

__all__ = ["plot_lineage_fate_correlation"]

logger = configure_logging(__name__)


def plot_lineage_fate_correlation(
    posterior_samples_path: str | Path,
    adata_pyrovelocity: str | Path,
    adata_scvelo: str | Path,
    adata_cospar: AnnData,
    ax: Axes,
    fig: Figure,
    state_color_dict: Dict,
    ylabel: str = "Unipotent Monocyte lineage",
    dotsize: int = 3,
    scale: float = 0.35,
    arrow: float = 3.5,
):
    posterior_samples = CompressedPickle.load(posterior_samples_path)
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    adata_scvelo = scv.read(adata_scvelo)
    adata_pyrovelocity = scv.read(adata_pyrovelocity)
    adata_input_clone = get_clone_trajectory(adata_scvelo)
    adata_input_clone.obsm["clone_vector_emb"][
        np.isnan(adata_input_clone.obsm["clone_vector_emb"])
    ] = 0
    density = 0.35
    diff = align_trajectory_diff(
        [adata_input_clone, adata_scvelo, adata_scvelo],
        [
            adata_input_clone.obsm["clone_vector_emb"],
            adata_scvelo.obsm["velocity_emb"],
            embed_mean,
        ],
        embed="emb",
        density=density,
    )
    scvelo_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
    )
    pyro_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
    )
    scvelo_cos_mean = scvelo_cos.mean()
    pyro_cos_mean = pyro_cos.mean()
    print(scvelo_cos_mean, pyro_cos_mean)

    res = pd.DataFrame(
        {
            "X": adata_pyrovelocity.obsm["X_emb"][:, 0],
            "Y": adata_pyrovelocity.obsm["X_emb"][:, 1],
            "celltype": adata_pyrovelocity.obs.state_info,
        }
    )
    sns.scatterplot(
        data=res,
        x="X",
        y="Y",
        hue="celltype",
        palette=state_color_dict,
        ax=ax[0],
        s=dotsize,
        alpha=0.90,
        linewidth=0,
        legend=False,
    )
    ax[0].set_title("Cell types", fontsize=7)
    ax[0].set_ylabel(ylabel, fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata_input_clone,
        scale=scale,
        autoscale=True,
        show=False,
        s=dotsize,
        density=density,
        arrow_size=arrow,
        linewidth=1,
        vkey="clone_vector",
        basis="emb",
        ax=ax[1],
        title="Clonal progression",
        color="gray",
        arrow_color="black",
        fontsize=7,
    )

    scv.pl.velocity_embedding_grid(
        adata_scvelo,
        show=False,
        s=dotsize,
        density=density,
        scale=scale,
        autoscale=True,
        arrow_size=arrow,
        linewidth=1,
        basis="emb",
        ax=ax[2],
        title="Scvelo",
        fontsize=7,
        color="gray",
        arrow_color="black",
    )
    ax[2].set_title(
        "scVelo cosine similarity: %.2f" % scvelo_cos_mean, fontsize=7
    )
    cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    cell_time_std = posterior_samples["cell_time"].std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean
    print(cell_time_cov)

    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        cell_time_cov,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="shared time",
        cmap="winter",
    )
    ax[3].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )

    cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    print(cell_magnitudes_cov)

    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap="summer",
    )
    ax[4].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )

    pca_angles = posterior_samples["pca_embeds_angle"]
    pca_cell_angles = pca_angles / np.pi * 180
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)
    print(pca_angles_std)

    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap="inferno",
        cmax=360,
    )
    ax[5].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )

    scv.pl.scatter(
        adata_cospar[adata_pyrovelocity.obs_names.str.replace(r"-\d", ""), :],
        basis="emb",
        fontsize=7,
        color="fate_potency",
        cmap="inferno_r",
        show=False,
        ax=ax[6],
        s=dotsize,
    )
    ax[6].set_title("Clonal fate potency", fontsize=7)
    gold = adata_cospar[
        adata_pyrovelocity.obs_names.str.replace(r"-\d", ""), :
    ].obs.fate_potency
    select = ~np.isnan(gold)
    scv.pl.scatter(
        adata_scvelo,
        c="latent_time",
        basis="emb",
        s=dotsize,
        cmap="inferno",
        ax=ax[7],
        show=False,
        fontsize=7,
    )
    ax[7].set_title(
        "Scvelo latent time\ncorrelation: %.2f"
        % spearmanr(-gold[select], adata_scvelo.obs.latent_time.values[select])[
            0
        ],
        fontsize=7,
    )
    plot_posterior_time(
        posterior_samples,
        adata_pyrovelocity,
        ax=ax[8],
        basis="emb",
        fig=fig,
        addition=False,
        position="right",
    )
    ax[8].set_title(
        "Pyro-Velocity shared time\ncorrelation: %.2f"
        % spearmanr(
            -gold[select],
            posterior_samples["cell_time"].mean(0).flatten()[select],
        )[0],
        fontsize=7,
    )
