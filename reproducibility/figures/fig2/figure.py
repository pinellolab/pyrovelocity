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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_venn import venn2
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger


"""Loads trained figure 2 data and produces figure 2.

Inputs:
  data:
    "models/pancreas_model1/trained.h5ad"
    "models/pbmc68k_model1/trained.h5ad"
  models:
    "models/pancreas_model1/pyrovelocity.pkl"
    "models/pbmc68k_model1/pyrovelocity.pkl"

Outputs:
  figures:
    "reports/fig2/fig2_raw_gene_selection_model1.{tif,svg}"
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    ##################
    # load data
    ##################

    pyrovelocity_pancreas_data_path = (
        conf.model_training.pancreas_model1.pyrovelocity_data_path
    )
    pyrovelocity_pancreas_model2_data_path = (
        conf.model_training.pancreas_model2.pyrovelocity_data_path
    )
    trained_pancreas_data_path = conf.model_training.pancreas_model1.trained_data_path
    trained_pancreas_model2_data_path = (
        conf.model_training.pancreas_model2.trained_data_path
    )

    pyrovelocity_pbmc68k_data_path = (
        conf.model_training.pbmc68k_model1.pyrovelocity_data_path
    )
    trained_pbmc68k_data_path = conf.model_training.pbmc68k_model1.trained_data_path

    figure2_tif_path = conf.reports.figure2.tif_path
    figure2_svg_path = conf.reports.figure2.svg_path

    revision_marker_path = conf.reports.figure2.biomarker_selection_path
    revision_phaseport_path = conf.reports.figure2.biomarker_phaseportrait_path

    revision_uncertainty_plot_path = conf.reports.figure2.uncertainty_param_plot_path
    revision_uncertainty_uncertain_plot_path = (
        conf.reports.figure2.uncertainty_magnitude_plot_path
    )

    if os.path.isfile(pyrovelocity_pancreas_data_path) and os.access(
        pyrovelocity_pancreas_data_path, os.R_OK
    ):
        logger.info(f"Loading: {pyrovelocity_pancreas_data_path}")
        with open(pyrovelocity_pancreas_data_path, "rb") as f:
            posterior_samples = pickle.load(f)
        with open(pyrovelocity_pancreas_model2_data_path, "rb") as f:
            model2_posterior_samples = pickle.load(f)
    else:
        logger.error(
            f"{pyrovelocity_pancreas_data_path} does not exist or is not accessible"
        )
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), pyrovelocity_pancreas_data_path
        )

    # posterior_samples = posterior_samples["posterior_samples"]
    v_map_all = posterior_samples["vector_field_posterior_samples"]
    embeds_radian = posterior_samples["embeds_angle"]
    fdri = posterior_samples["fdri"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    # model2_posterior_samples = model2_posterior_samples["posterior_samples"]
    model2_v_map_all = model2_posterior_samples["vector_field_posterior_samples"]
    model2_embeds_radian = model2_posterior_samples["embeds_angle"]
    model2_fdri = model2_posterior_samples["fdri"]
    model2_embed_mean = model2_posterior_samples["vector_field_posterior_mean"]

    if os.path.isfile(pyrovelocity_pbmc68k_data_path) and os.access(
        pyrovelocity_pbmc68k_data_path, os.R_OK
    ):
        logger.info(f"Loading: {pyrovelocity_pbmc68k_data_path}")
        with open(pyrovelocity_pbmc68k_data_path, "rb") as f:
            posterior_samples_pbmc = pickle.load(f)
    else:
        logger.error(
            f"{pyrovelocity_pbmc68k_data_path} does not exist or is not accessible"
        )
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), pyrovelocity_pbmc68k_data_path
        )

    # posterior_samples_pbmc = posterior_samples["posterior_samples"]
    v_map_all_pbmc = posterior_samples_pbmc["vector_field_posterior_samples"]
    embeds_radian_pbmc = posterior_samples_pbmc["embeds_angle"]
    fdri_pbmc = posterior_samples_pbmc["fdri"]
    embed_mean_pbmc = posterior_samples_pbmc["vector_field_posterior_mean"]

    if os.path.isfile(trained_pancreas_data_path) and os.access(
        trained_pancreas_data_path, os.R_OK
    ):
        logger.info(f"Loading: {trained_pancreas_data_path}")
        adata = scv.read(trained_pancreas_data_path)
        adata_model2 = scv.read(trained_pancreas_model2_data_path)
    else:
        logger.error(
            f"{trained_pancreas_data_path} does not exist or is not accessible"
        )
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), trained_pancreas_data_path
        )
    # Model 1 and Model 2 should have the same default expression
    assert (adata.X - adata_model2.X).sum() == 0

    if os.path.isfile(trained_pbmc68k_data_path) and os.access(
        trained_pbmc68k_data_path, os.R_OK
    ):
        logger.info(f"Loading: {trained_pbmc68k_data_path}")
        adata_pbmc = scv.read(trained_pbmc68k_data_path)
    else:
        logger.error(f"{trained_pbmc68k_data_path} does not exist or is not accessible")
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), trained_pbmc68k_data_path
        )

    #################
    # generate figure
    #################

    fig = plt.figure(figsize=(7.07, 6.5))
    dot_size = 3
    font_size = 7
    subfig = fig.subfigures(3, 1, wspace=0.0, hspace=0, height_ratios=[1.2, 1.2, 2.6])

    ress = pd.DataFrame(
        {
            "cell_type": adata_pbmc.obs["celltype"].values,
            "X1": adata_pbmc.obsm["X_tsne"][:, 0],
            "X2": adata_pbmc.obsm["X_tsne"][:, 1],
        }
    )

    logger.info(f"\n\nplotting pbmc cell types\n\n")
    pbmcfig_A0 = subfig[0].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[4, 2])
    ax = pbmcfig_A0[0].subplots(1, 4)
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
        bbox_to_anchor=[4.0, -0.01],
        ncol=3,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    ax[0].text(
        -0.1,
        1.15,
        "a",
        transform=ax[0].transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="right",
    )

    logger.info(f"\n\nplotting pbmc scvelo velocity embedding stream\n\n")
    kwargs = dict(
        color="gray",
        density=0.8,
        add_margin=0.1,
        s=dot_size,
        show=False,
        alpha=0.2,
        min_mass=3.5,
        frameon=False,
    )
    scv.pl.velocity_embedding_stream(
        adata_pbmc, basis="tsne", fontsize=font_size, ax=ax[1], title="", **kwargs
    )
    ax[1].set_title("Scvelo\n", fontsize=7)

    logger.info(f"\n\nplotting pbmc pyrovelocity velocity embedding stream\n\n")
    scv.pl.velocity_embedding_stream(
        adata_pbmc,
        fontsize=font_size,
        basis="tsne",
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    logger.info(f"\n\nplotting pbmc pyrovelocity vector examples\n\n")
    plot_arrow_examples(
        adata_pbmc,
        np.transpose(v_map_all_pbmc, (1, 2, 0)),
        embeds_radian_pbmc,
        ax=ax[3],
        n_sample=30,
        fig=pbmcfig_A0,
        basis="tsne",
        scale=None,
        alpha=0.3,
        index=100,
        scale2=None,
        num_certain=0,
        num_total=4,
    )
    ax[3].set_title("Single cell\nvector field", fontsize=7)

    logger.info(f"\n\nplotting pbmc pyrovelocity vector field uncertainty\n\n")
    plot_vector_field_uncertain(
        adata_pbmc,
        embed_mean_pbmc,
        embeds_radian_pbmc,
        fig=pbmcfig_A0[1],
        cbar=True,
        basis="tsne",
        scale=0.018,
        arrow_size=5,
    )
    pbmcfig_A0[0].subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    pbmcfig_A0[1].subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    pbmcfig_A0[0].text(-0.06, 0.58, "PBMC", size=7, rotation="vertical", va="center")

    ress = pd.DataFrame(
        {
            "cell_type": adata.obs["clusters"].values,
            "X1": adata.obsm["X_umap"][:, 0],
            "X2": adata.obsm["X_umap"][:, 1],
        }
    )
    subfig_A0 = subfig[1].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[4, 2])

    logger.info(f"\n\nplotting pancreas cell types\n\n")
    ax = subfig_A0[0].subplots(1, 4)
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
        bbox_to_anchor=[2.9, -0.01],
        ncol=4,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    ax[0].text(
        -0.1,
        1.18,
        "b",
        transform=ax[0].transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="right",
    )

    logger.info(f"\n\nplotting pancreas scvelo velocity embedding stream\n\n")
    kwargs = dict(
        color="gray",
        density=0.8,
        add_margin=0.1,
        s=dot_size,
        show=False,
        alpha=0.2,
        min_mass=3.5,
        frameon=False,
    )
    scv.pl.velocity_embedding_stream(
        adata, fontsize=font_size, ax=ax[1], title="", **kwargs
    )
    ax[1].set_title("Scvelo\n", fontsize=7)

    logger.info(f"\n\nplotting pancreas pyrovelocity velocity embedding stream\n\n")
    scv.pl.velocity_embedding_stream(
        adata,
        fontsize=font_size,
        basis="umap",
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    logger.info(f"\n\nplotting pancreas pyrovelocity vector examples\n\n")
    plot_arrow_examples(
        adata,
        np.transpose(v_map_all, (1, 2, 0)),
        embeds_radian,
        ax=ax[3],
        n_sample=30,
        fig=fig,
        basis="umap",
        scale=0.005,
        alpha=0.2,
        index=5,
        scale2=0.03,
        num_certain=6,
        num_total=6,
    )
    ax[3].set_title("Single cell\nvector field", fontsize=7)

    logger.info(f"\n\nplotting pancreas pyrovelocity vector field uncertainty\n\n")
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        embeds_radian,
        fig=subfig_A0[1],
        cbar=True,
        basis="umap",
        scale=0.03,
        # scale=1.0,
        arrow_size=5,
    )
    subfig_A0[0].subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.80, bottom=0.33
    )
    subfig_A0[1].subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.80, bottom=0.33
    )
    subfig_A0[0].text(-0.06, 0.58, "Pancreas", size=7, rotation="vertical", va="center")

    logger.info(f"\n\nplotting pancreas pyrovelocity shared time\n\n")
    subfig_B = subfig[2].subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.6, 4])
    ax = subfig_B[0].subplots(2, 1)
    plot_posterior_time(
        posterior_samples, adata, ax=ax[0], fig=subfig_B[0], addition=False
    )
    subfig_B[0].subplots_adjust(
        hspace=0.3, wspace=0.1, left=0.01, right=0.8, top=0.92, bottom=0.17
    )
    logger.info(f"\n\nplotting pancreas pyrovelocity gene ranking\n\n")
    volcano_data2, _ = plot_gene_ranking(
        [posterior_samples],
        [adata],
        ax=ax[1],
        time_correlation_with="st",
        assemble=True,
    )
    ax[0].text(
        -0.22,
        1.15,
        "c",
        transform=ax[0].transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[1].text(
        -0.1,
        1.15,
        "d",
        transform=ax[1].transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
        ha="right",
    )

    logger.info(f"\n\nplotting pancreas pyrovelocity rainbow plot\n\n")
    _ = rainbowplot(
        volcano_data2,
        adata,
        posterior_samples,
        subfig_B[1],
        data=["st", "ut"],
        num_genes=4,
    )

    logger.info(f"\n\nsaving figure 2 to TIF\n\n")
    fig.savefig(
        figure2_tif_path,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )

    logger.info(f"\n\nsaving figure 2 to SVG\n\n")
    fig.savefig(
        figure2_svg_path,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )
    logger.info(f"\n\nCOMPLETE: plot figure 2\n\n")

    fig_revision, ax = plt.subplots(1, 3)
    fig_revision.set_size_inches(7.9, 2.2)
    model1_genes = (
        volcano_data2.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=False)
        .head(4)
        .index
    )
    print(model1_genes)
    model2_volcano, _ = plot_gene_ranking(
        [model2_posterior_samples],
        [adata_model2],
        ax=ax[0],
        time_correlation_with="st",
        assemble=False,
        selected_genes=model1_genes,
    )
    ax[0].set_title("Model 1 genes on Model 2 plot", fontsize=7)
    model2_genes = (
        model2_volcano.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=False)
        .head(4)
        .index
    )
    _, _ = plot_gene_ranking(
        [posterior_samples],
        [adata],
        ax=ax[1],
        time_correlation_with="st",
        assemble=False,
        selected_genes=model2_genes,
    )
    ax[1].set_title("Model 2 genes on Model 1 plot", fontsize=7)
    model1_geneset = set(
        volcano_data2.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=False)
        .head(50)
        .index
    )
    model2_geneset = set(
        model2_volcano.sort_values("mean_mae", ascending=False)
        .head(300)
        .sort_values("time_correlation", ascending=False)
        .head(50)
        .index
    )
    venn2(
        [model1_geneset, model2_geneset],
        ["Model1\ntop 50", "Model 2\ntop 50"],
        ax=ax[2],
    )
    ax[2].set_title("Marker comparison", fontsize=7)
    print(model1_genes.intersection(model2_genes))

    fig_revision.savefig(
        revision_marker_path,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )

    print(v_map_all.shape)
    logger.info(f"Loading: {v_map_all.shape}")

    # (30, 3696, 2)
    assert v_map_all.shape == (30, 3696, 2)
    embeds_magnitude = np.sqrt((v_map_all**2).sum(axis=-1))
    fig, ax = plt.subplots()
    fig.set_size_inches(13.5, 9.5)
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        embeds_magnitude,
        fig=fig,
        ax=None,
        cbar=True,
        basis="umap",
        scale=0.03,
        # scale=1.0,
        arrow_size=5,
        uncertain_measure="magnitude",
    )
    fig.subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.12, right=0.95, top=0.95, bottom=0.42
    )
    fig.savefig(revision_uncertainty_uncertain_plot_path)

    with PdfPages(revision_phaseport_path) as pdf:
        fig_revision2 = rainbowplot(
            volcano_data2,
            adata,
            posterior_samples,
            None,
            data=["st", "ut"],
            num_genes=8,
            genes=np.hstack([model1_genes, model2_genes]),
        )
        pdf.savefig()

        fig_revision2 = rainbowplot(
            model2_volcano,
            adata_model2,
            model2_posterior_samples,
            None,
            data=["st", "ut"],
            num_genes=8,
            genes=np.hstack([model1_genes, model2_genes]),
        )
        pdf.savefig()
        plt.close()

    with PdfPages(revision_uncertainty_plot_path) as pdf:
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(18, 12)
        for index, kinetics in enumerate(["alpha", "beta", "gamma"]):
            print(posterior_samples[kinetics].squeeze().shape)
            print(np.isin(adata.var_names, list(model1_geneset)).sum())
            df = pd.DataFrame(
                posterior_samples[kinetics].squeeze()[
                    :, np.isin(adata.var_names, list(model1_geneset))
                ],
                columns=adata.var_names[np.isin(adata.var_names, list(model1_geneset))],
            )
            df_long = df.melt()
            print(df_long.head())
            ax1 = sns.boxplot(x="index", y="value", data=df_long, ax=ax[index])
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
            ax1.set_ylabel(kinetics)
        fig.subplots_adjust(
            hspace=0.4, wspace=0.45, left=0.08, right=0.95, top=0.9, bottom=0.15
        )
        pdf.savefig()

        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(18, 12)
        for index, kinetics in enumerate(["alpha", "beta", "gamma"]):
            df = pd.DataFrame(
                model2_posterior_samples[kinetics].squeeze()[
                    :, np.isin(adata_model2.var_names, list(model2_geneset))
                ],
                columns=adata_model2.var_names[
                    np.isin(adata_model2.var_names, list(model2_geneset))
                ],
            )
            df_long = df.melt()
            print(df_long.head())
            ax1 = sns.boxplot(x="index", y="value", data=df_long, ax=ax[index])
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
            ax1.set_ylabel(kinetics)
        fig.subplots_adjust(
            hspace=0.4, wspace=0.45, left=0.08, right=0.95, top=0.9, bottom=0.15
        )
        pdf.savefig()

        plt.close()
    logger.info(f"\n\nCOMPLETE: plot revision figure\n\n")


@hydra.main(version_base="1.2", config_path="..", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plot results
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figure2.path}\n"
    )
    Path(conf.reports.figure2.path).mkdir(parents=True, exist_ok=True)

    if (
        os.path.isfile(conf.reports.figure2.tif_path)
        and os.path.isfile(conf.reports.figure2.svg_path)
        and os.path.isfile(conf.reports.figure2.biomarker_selection_path)
    ):
        logger.info(
            f"\n\nFigure 2 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figure2.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure 2\n\n")
        plots(conf, logger)


if __name__ == "__main__":
    main()
