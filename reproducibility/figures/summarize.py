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
    for data_model in conf.reports.model_summary.summarize:
        ##################
        # load data
        ##################
        print(data_model)

        data_model_conf = conf.model_training[data_model]
        trained_data_path = data_model_conf.trained_data_path
        model_path = data_model_conf.model_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        vector_field_basis = data_model_conf.vector_field_parameters.basis

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
