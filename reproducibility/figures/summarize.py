import os
import pickle
from logging import Logger
from pathlib import Path
from statistics import harmonic_mean
from typing import Text

import hydra
import matplotlib.pyplot as plt
import scvelo as scv
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.data import load_data
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes


"""Loads model-trained data and generates figures.

for data_model in
    [
        pancreas_model1,
        pancreas_model2,
        pbmc68k_model1,
        pbmc68k_model2,
        pons_model1,
        pons_model2,
    ]

Inputs:
  data:
    "models/{data_model}/trained.h5ad"
    "models/{data_model}/pyrovelocity.pkl"

Outputs:
  figures:
    shared_time_plot: reports/{data_model}/shared_time.pdf
    volcano_plot: reports/{data_model}/volcano.pdf
    rainbow_plot: reports/{data_model}/rainbow.pdf
    uncertainty_param_plot: reports/{data_model}/param_uncertainties.pdf
    vector_field_plot: reports/{data_model}/vector_field.pdf
    biomarker_selection_plot: reports/{data_model}/markers_selection_scatterplot.tif
    biomarker_phaseportrait_plot: reports/{data_model}/markers_phaseportrait.pdf
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """
    for data_model in conf.reports.model_summary.summarize:
        ##################
        # load data
        ##################
        print(data_model)

        data_model_conf = conf.model_training[data_model]
        cell_state = data_model_conf.training_parameters.cell_state
        trained_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        reports_data_model_conf = conf.reports.model_summary[data_model]

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  reports: {reports_data_model_conf.path}\n"
        )
        Path(reports_data_model_conf.path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        print_anndata(adata)

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        with open(pyrovelocity_data_path, "rb") as f:
            result_dict = pickle.load(f)
        adata_model_pos = result_dict["adata_model_pos"]

        ##################
        # generate figures
        ##################

        # volcano plot

        volcano_plot = reports_data_model_conf.volcano_plot

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

        rainbow_plot = reports_data_model_conf.rainbow_plot

        if os.path.isfile(rainbow_plot):
            logger.info(f"{rainbow_plot} exists")
        else:
            volcano_data, _ = compute_volcano_data(
                [adata_model_pos], [adata], time_correlation_with="st"
            )
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
                cell_state=cell_state,
            )
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{rainbow_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )

        # mean vector field plot

        vector_field_plot = reports_data_model_conf.vector_field_plot
        if os.path.isfile(vector_field_plot):
            logger.info(f"{vector_field_plot} exists")
        else:
            logger.info(f"Generating figure: {vector_field_plot}")
            fig, ax = plt.subplots()

            vector_field_basis = data_model_conf.vector_field_parameters.basis

            # embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax)
            scv.pl.velocity_embedding_grid(
                adata,
                basis=vector_field_basis,
                color=cell_state,
                title="",
                vkey="velocity_pyro",
                linewidth=1,
                ax=ax,
                show=False,
                legend_loc="right margin",
                density=0.4,
                scale=0.2,
                arrow_size=2,
                arrow_length=2,
                arrow_color="black",
            )
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{vector_field_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plots figures for model summary.
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nPlotting summary figure(s) in: {conf.reports.model_summary.summarize}\n\n"
    )

    plots(conf, logger)


if __name__ == "__main__":
    main()
