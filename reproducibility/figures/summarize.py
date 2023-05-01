import os
import pickle
from logging import Logger
from pathlib import Path
from statistics import harmonic_mean
from typing import Text

import hydra
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import seaborn as sns
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.data import load_data
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import anndata_counts_to_df
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

        dataframe_path = reports_data_model_conf.dataframe_path
        volcano_plot = reports_data_model_conf.volcano_plot
        rainbow_plot = reports_data_model_conf.rainbow_plot
        vector_field_plot = reports_data_model_conf.vector_field_plot
        shared_time_plot = reports_data_model_conf.shared_time_plot
        output_filenames = [
            dataframe_path,
            volcano_plot,
            rainbow_plot,
            vector_field_plot,
            shared_time_plot,
        ]
        if all(os.path.isfile(f) for f in output_filenames):
            logger.info(
                "\n\t" + "\n\t".join(output_filenames) + "\nAll output files exist"
            )
            return logger.warn("Remove output files if you want to regenerate them.")

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        print_anndata(adata)

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        with open(pyrovelocity_data_path, "rb") as f:
            posterior_samples = pickle.load(f)

        ##################
        # save dataframe
        ##################

        logger.info(f"Saving AnnData object to dataframe: {dataframe_path}")
        (
            df,
            total_obs,
            total_var,
            max_spliced,
            max_unspliced,
        ) = anndata_counts_to_df(adata)

        CompressedPickle.save(
            dataframe_path,
            (
                df,
                total_obs,
                total_var,
                max_spliced,
                max_unspliced,
            ),
        )

        ##################
        # generate figures
        ##################
        vector_field_basis = data_model_conf.vector_field_parameters.basis

        # shared time plot
        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        adata.obs["shared_time_uncertain"] = cell_time_std
        adata.obs["shared_time_mean"] = cell_time_mean
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(9.6, 7)
        ax = ax.flatten()
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_mean",
            ax=ax[0],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_uncertain",
            ax=ax[1],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        select = adata.obs["shared_time_uncertain"] > np.quantile(
            adata.obs["shared_time_uncertain"], 0.9
        )
        sns.kdeplot(
            adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
            adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
            ax=ax[1],
            levels=3,
            fill=False,
        )
        ax[2].hist(cell_time_std, bins=100)
        ax[3].hist(cell_time_std / cell_time_mean, bins=100)
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
            print(posterior_samples.keys())
            for key in posterior_samples.keys():
                print(posterior_samples[key].shape)
            fig, ax = plt.subplots()

            volcano_data, _ = plot_gene_ranking(
                [posterior_samples], [adata], ax=ax, time_correlation_with="st"
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
                posterior_samples,
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

        if os.path.isfile(vector_field_plot):
            logger.info(f"{vector_field_plot} exists")
        else:
            logger.info(f"Generating figure: {vector_field_plot}")
            fig, ax = plt.subplots()

            # embed_mean = plot_mean_vector_field(posterior_samples, adata, ax=ax)
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
