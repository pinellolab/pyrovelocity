import argparse
import os
import pickle
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import scvelo as scv
from omegaconf import DictConfig
from scipy.stats import spearmanr

from pyrovelocity.config import config_setup
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.utils import get_pylogger


"""Loads model1-trained pancreas data and generates figures.

Inputs:
  data:
    "fig2_pancreas_processed.h5ad"
    "fig2_pancreas_data.pkl"

Outputs:
  figures:
    "fig2_test_sub.pdf"
    "fig2_test_volcano_sub.pdf"
    "fig2_test_rainbow_sub.pdf"
    "fig2_test_vecfield_sub.pdf"
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    ###########
    # load data
    ###########

    trained_data_path = conf.model_training.pancreas_model1.trained_data_path
    pyrovelocity_data_path = conf.model_training.pancreas_model1.pyrovelocity_data_path

    pancreas_model1 = conf.reports.figure2.pancreas_model1
    shared_time_plot = pancreas_model1.shared_time_plot
    volcano_plot = pancreas_model1.volcano_plot
    rainbow_plot = pancreas_model1.rainbow_plot
    vector_field_plot = pancreas_model1.vector_field_plot

    logger.info(f"Loading trained data: {trained_data_path}")
    adata = scv.read(trained_data_path)

    logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
    with open(pyrovelocity_data_path, "rb") as f:
        result_dict = pickle.load(f)
    adata_model_pos = result_dict["adata_model_pos"]

    ##################
    # generate figures
    ##################

    # shared time plot

    if os.path.isfile(shared_time_plot):
        logger.info(f"{shared_time_plot} exists")
    else:
        logger.info(f"Generating figure: {shared_time_plot}")
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(15, 3)
        scv.pl.scatter(
            adata,
            color="latent_time",
            show=False,
            ax=ax[0],
            title="scvelo %.2f"
            % spearmanr(1 - adata.obs.cytotrace, adata.obs.latent_time)[0],
            cmap="RdBu_r",
            basis="umap",
        )
        scv.pl.scatter(
            adata,
            color="cell_time",
            show=False,
            basis="umap",
            ax=ax[1],
            title="pyro %.2f"
            % spearmanr(1 - adata.obs.cytotrace, adata.obs.cell_time)[0],
        )
        scv.pl.scatter(adata, color="1-Cytotrace", show=False, ax=ax[2])
        print(spearmanr(adata.obs.cytotrace, adata.obs.cell_time))
        print(spearmanr(adata.obs.cell_time, adata.obs.latent_time))
        fig.savefig(
            shared_time_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    # volcano plot

    if os.path.isfile(volcano_plot):
        logger.info(f"{volcano_plot} exists")
    else:
        logger.info(f"Generating figure: {volcano_plot}")
        fig, ax = plt.subplots()
        volcano_data, _ = plot_gene_ranking(
            [adata_model_pos], [adata], ax=ax, time_correlation_with="st"
        )
        fig.savefig(
            volcano_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )
        print(
            volcano_data.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=False)
            .head(8)
        )

    # rainbow plot

    if os.path.isfile(rainbow_plot):
        logger.info(f"{rainbow_plot} exists")
    else:
        logger.info(f"Generating figure: {rainbow_plot}")
        fig = us_rainbowplot(
            volcano_data.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=False)
            .head(5)
            .index,
            adata,
            adata_model_pos,
            data=["st", "ut"],
        )
        fig.savefig(
            rainbow_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

    # mean vector field plot

    if os.path.isfile(vector_field_plot):
        logger.info(f"{vector_field_plot} exists")
    else:
        logger.info(f"Generating figure: {vector_field_plot}")
        fig, ax = plt.subplots()
        # embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax)
        scv.pl.velocity_embedding_grid(
            adata,
            basis="umap",
            vkey="velocity_pyro",
            linewidth=1,
            ax=ax,
            show=False,
            legend_loc="on data",
            density=0.4,
            scale=0.2,
            arrow_size=3,
        )
        fig.savefig(
            vector_field_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def main(config_path: str) -> None:
    """Plot results
    Args:
        config_path {Text}: path to config
    """
    conf = config_setup(config_path)

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  model data: {conf.model_training.pancreas_model1.path}\n"
        f"  reports: {conf.reports.figure2.path}\n"
    )
    Path(conf.model_training.pancreas_model1.path).mkdir(parents=True, exist_ok=True)
    Path(conf.reports.figure2.path).mkdir(parents=True, exist_ok=True)

    plots(conf, logger)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)
