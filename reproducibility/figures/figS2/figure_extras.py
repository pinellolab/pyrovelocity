import errno
import os
import pickle
from logging import Logger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from astropy import units as u
from astropy.stats import circstd

# from scipy.stats import circvar, circstd, circmean
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_venn import venn2
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from statannotations.Annotator import Annotator

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import get_posterior_sample_angle_uncertainty
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.plot import set_colorbar
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import get_pylogger


def plots(
    conf: DictConfig,
    logger: Logger,
    fig_name: str = None,
    subset_pkl: str = None,
) -> None:
    """Compute cell-wise uncertainty metrics across all datasets

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    ncpus_use = 16

    for data_model in conf.train_models:
        ##################
        # load data
        ##################
        print(data_model)

        data_model_conf = conf.model_training[data_model]
        embedding = vector_field_basis = data_model_conf.vector_field_parameters.basis
        cell_state = data_model_conf.training_parameters.cell_state
        adata_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        pyrovelocity_posterior_samples_path = data_model_conf.posterior_samples_path

        pyrovelocity_data_path_3genes = subset_pkl

        adata = scv.read(adata_data_path)
        posterior_samples = CompressedPickle.load(pyrovelocity_posterior_samples_path)

        genes = ["NKG7", "IGHM", "GNLY"]
        (genes_index,) = np.where(adata.var_names.isin(genes))
        adata = adata[:, genes_index].copy()

        print(adata.shape)
        print(posterior_samples["ut"].shape)
        print(posterior_samples["st"].shape)

        if not os.path.exists(pyrovelocity_data_path_3genes):
            for key in ["ut", "st", "u_scale", "beta", "gamma", "alpha"]:
                posterior_samples[key] = posterior_samples[key][:, :, genes_index]

            if ("u_scale" in posterior_samples) and (
                "s_scale" in posterior_samples
            ):  # Gaussian models
                scale = posterior_samples["u_scale"] / posterior_samples["s_scale"]
            elif ("u_scale" in posterior_samples) and not (
                "s_scale" in posterior_samples
            ):  # Poisson Model 2
                scale = posterior_samples["u_scale"]
            else:  # Poisson Model 1
                scale = 1

            # calculate 3 genes version of uncertainties
            original_spaces_velocity_samples = (
                posterior_samples["beta"] * posterior_samples["ut"] / scale
                - posterior_samples["gamma"] * posterior_samples["st"]
            )
            original_spaces_embeds_magnitude = np.sqrt(
                (original_spaces_velocity_samples**2).sum(axis=-1)
            )

            (
                vector_field_posterior_samples,
                embeds_radian,
                fdri,
            ) = vector_field_uncertainty(
                adata,
                posterior_samples,
                basis=vector_field_basis,
                n_jobs=ncpus_use,
            )
            embeds_magnitude = np.sqrt(
                (vector_field_posterior_samples**2).sum(axis=-1)
            )

            compute_mean_vector_field(
                posterior_samples=posterior_samples,
                adata=adata,
                basis=vector_field_basis,
                n_jobs=ncpus_use,
            )
            vector_field_posterior_mean = adata.obsm[
                f"velocity_pyro_{vector_field_basis}"
            ]
            (
                pca_vector_field_posterior_samples,
                pca_embeds_radian,
                pca_fdri,
            ) = vector_field_uncertainty(
                adata,
                posterior_samples,
                basis="pca",
                n_jobs=ncpus_use,
            )

            posterior_samples[
                "original_spaces_embeds_magnitude"
            ] = original_spaces_embeds_magnitude
            posterior_samples["genes"] = genes
            posterior_samples[
                "vector_field_posterior_samples"
            ] = vector_field_posterior_samples
            posterior_samples[
                "vector_field_posterior_mean"
            ] = vector_field_posterior_mean
            posterior_samples["fdri"] = fdri
            posterior_samples["embeds_magnitude"] = embeds_magnitude
            posterior_samples["embeds_angle"] = embeds_radian
            posterior_samples["ut_mean"] = posterior_samples["ut"].mean(0).squeeze()
            posterior_samples["st_mean"] = posterior_samples["st"].mean(0).squeeze()
            posterior_samples[
                "pca_vector_field_posterior_samples"
            ] = pca_vector_field_posterior_samples
            posterior_samples["pca_embeds_angle"] = pca_embeds_radian
            posterior_samples["pca_fdri"] = pca_fdri

            del posterior_samples["u"]
            del posterior_samples["s"]
            del posterior_samples["ut"]
            del posterior_samples["st"]
            CompressedPickle.save(pyrovelocity_data_path_3genes, posterior_samples)
            # with open(pyrovelocity_data_path_3genes, "wb") as f:
            #     pickle.dump(posterior_samples, f)
        else:
            posterior_samples = CompressedPickle.load(pyrovelocity_data_path_3genes)
            # with open(pyrovelocity_data_path_3genes, "rb") as f:
            #     posterior_samples = pickle.load(f)
            print((posterior_samples["fdri"] < 0.001).sum())

    adata.obs.loc[:, "vector_field_rayleigh_test"] = posterior_samples["fdri"]

    fig, ax = plt.subplots(1, 6)
    fig.set_size_inches(15, 2.8)
    kwargs = dict(
        color="gray",
        s=1,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=0.3,
        frameon=False,
        density=0.4,
        arrow_size=3,
        linewidth=1,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=7,
        basis=vector_field_basis,
        title="",
        ax=ax[0],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[0].set_title("Pyro-Velocity\n", fontsize=7)

    pca_embeds_angle = posterior_samples["pca_embeds_angle"]
    posterior_time = posterior_samples["cell_time"]
    embeds_radian = posterior_samples["embeds_angle"]
    cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    posterior_vector_field = posterior_samples["vector_field_posterior_samples"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    pca_cell_angles = pca_embeds_angle / np.pi * 180  # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()

    dot_size = 3.5
    font_size = 6.5
    scale = 0.35
    scale_high = 7.8
    scale_low = 7.8
    arrow = 3.6
    density = 0.4

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_time_std,
        ax=ax[1],
        cbar=True,
        fig=fig,
        basis=embedding,
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
        ax=ax[2],
        cbar=True,
        fig=fig,
        basis=embedding,
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
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis=embedding,
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

    fdri = posterior_samples["fdri"]
    pca_fdri = posterior_samples["pca_fdri"]
    for index, fdr in enumerate([fdri, pca_fdri]):
        adata.obs.loc[:, "vector_field_rayleigh_test"] = fdr
        basis = "tsne"
        im = ax[4 + index].scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            alpha=0.9,
            c=adata.obs["vector_field_rayleigh_test"],
            cmap="inferno_r",
            linewidth=0,
        )
        set_colorbar(im, ax[4 + index], labelsize=5, fig=fig, position="right")
        ax[4 + index].axis("off")

    ax[4].set_title(
        f"UMAP angle Rayleigh test {(fdri<0.05).sum()/fdri.shape[0]:.2f}", fontsize=7
    )
    ax[5].set_title(
        f"PCA angle Rayleigh test {(pca_fdri<0.05).sum()/pca_fdri.shape[0]:.2f}",
        fontsize=7,
    )

    for ext in ["", ".png"]:
        fig.savefig(
            f"{fig_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


@hydra.main(version_base="1.2", config_path="..", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plot results
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figureS2_extras.path}\n"
    )
    Path(conf.reports.figureS2_extras.path).mkdir(parents=True, exist_ok=True)
    confS2 = conf.reports.figureS2_extras

    if os.path.isfile(confS2.subset_genes_plot):
        logger.info(
            f"\n\nFigure S2 extras outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS2_extras.path}\n"
        )
    else:
        for fig_name in [confS2.subset_genes_plot]:
            plots(conf, logger, fig_name=fig_name, subset_pkl=confS2.subset_pkl)


if __name__ == "__main__":
    main()
