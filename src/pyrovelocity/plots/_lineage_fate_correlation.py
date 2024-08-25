from pathlib import Path

import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import distance
from scipy.stats import spearmanr

from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._time import plot_posterior_time
from pyrovelocity.plots._trajectory import (
    align_trajectory_diff,
    get_clone_trajectory,
)
from pyrovelocity.plots._uncertainty import (
    get_posterior_sample_angle_uncertainty,
)
from pyrovelocity.plots._vector_fields import plot_vector_field_uncertainty
from pyrovelocity.utils import load_anndata_from_path

__all__ = ["plot_lineage_fate_correlation"]

logger = configure_logging(__name__)


@beartype
def plot_lineage_fate_correlation(
    posterior_samples_path: str | Path | AnnData,
    adata_pyrovelocity: str | Path | AnnData,
    adata_scvelo: str | Path | AnnData,
    adata_cospar: str | Path | AnnData,
    ax: Axes | np.ndarray,
    fig: Figure,
    # state_color_dict: Dict,
    ylabel: str = "Unipotent Monocyte lineage",
    dotsize: int = 3,
    scale: float = 0.35,
    arrow: float = 3.5,
):
    """
    Plot lineage fate correlation with shared latent time estimates.

    Args:
        posterior_samples_path (str | Path): Path to the posterior samples.
        adata_pyrovelocity (str | Path): Path to the Pyro-Velocity AnnData object.
        adata_scvelo (str | Path): Path to the scVelo AnnData object.
        adata_cospar (AnnData): AnnData object with COSPAR results.
        ax (Axes): Matplotlib axes.
        fig (Figure): Matplotlib figure.
        state_color_dict (Dict): Dictionary with cell state colors.
        ylabel (str, optional): Label for y axis. Defaults to "Unipotent Monocyte lineage".
        dotsize (int, optional): Size of plotted points. Defaults to 3.
        scale (float, optional): Plot scale. Defaults to 0.35.
        arrow (float, optional): Arrow size. Defaults to 3.5.

    Examples:
        >>> # xdoctest: +SKIP
        >>> import matplotlib.pyplot as plt
        >>> import scanpy as sc
        >>> from pyrovelocity.io.datasets import larry_cospar, larry_mono
        >>> from pyrovelocity.utils import load_anndata_from_path
        >>> from pyrovelocity.plots import plot_lineage_fate_correlation
        ...
        >>> fig, ax = plt.subplots(1, 9)
        >>> fig.set_size_inches(17, 2.75)
        >>> fig.subplots_adjust(
        ...     hspace=0.4, wspace=0.2, left=0.01, right=0.99, top=0.95, bottom=0.3
        >>> )
        ...
        >>> data_set_name = "larry_mono"
        >>> model_name = "model2"
        >>> data_set_model_pairing = f"{data_set_name}_{model_name}"
        >>> model_path = f"models/{data_set_model_pairing}"
        ...
        >>> adata_pyrovelocity = load_anndata_from_path(f"{model_path}/postprocessed.h5ad")
        >>> # color_dict = dict(
        ... #     zip(
        ... #         adata_pyrovelocity.obs.state_info.cat.categories,
        ... #         adata_pyrovelocity.uns["state_info_colors"],
        ... #     )
        ... # )
        >>> adata_dynamical = load_anndata_from_path(f"data/processed/larry_mono_processed.h5ad")
        >>> adata_cospar = load_anndata_from_path(f"data/external/larry_cospar.h5ad")
        >>> plot_lineage_fate_correlation(
        ...     posterior_samples_path=f"{model_path}/pyrovelocity.pkl.zst",
        ...     adata_pyrovelocity=adata_pyrovelocity,
        ...     adata_scvelo=adata_dynamical,
        ...     adata_cospar=adata_cospar,
        ...     ax=ax,
        ...     fig=fig,
        ... )
    """
    posterior_samples = CompressedPickle.load(posterior_samples_path)
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    if isinstance(adata_pyrovelocity, str | Path):
        adata_pyrovelocity = load_anndata_from_path(adata_pyrovelocity)
    if isinstance(adata_scvelo, str | Path):
        adata_scvelo = load_anndata_from_path(adata_scvelo)
    if isinstance(adata_cospar, str | Path):
        adata_cospar = load_anndata_from_path(adata_cospar)

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
        # palette=state_color_dict,
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

    plot_vector_field_uncertainty(
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

    plot_vector_field_uncertainty(
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

    plot_vector_field_uncertainty(
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

    # The obs names in adata_pyrovelocity have a "-N" suffix that
    # is not present in the adata_cospar obs Index.
    patched_adata_pyrovelocity_obs_names = (
        adata_pyrovelocity.obs_names.str.replace(
            r"-\d",
            "",
            regex=True,
        )
    )
    adata_cospar_obs_subset = adata_cospar[
        patched_adata_pyrovelocity_obs_names, :
    ]
    scv.pl.scatter(
        adata=adata_cospar_obs_subset,
        basis="emb",
        fontsize=7,
        color="fate_potency_transition_map",
        cmap="inferno_r",
        show=False,
        ax=ax[6],
        s=dotsize,
    )
    ax[6].set_title("Clonal fate potency", fontsize=7)
    gold_standard = adata_cospar_obs_subset.obs.fate_potency_transition_map
    select = ~np.isnan(gold_standard)
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
        % spearmanr(
            -gold_standard[select], adata_scvelo.obs.latent_time.values[select]
        )[0],
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
            -gold_standard[select],
            posterior_samples["cell_time"].mean(0).flatten()[select],
        )[0],
        fontsize=7,
    )

    for ext in ["", ".png"]:
        fig.savefig(
            f"lineage_fate_correlation.pdf{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )