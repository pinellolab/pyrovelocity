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
from statannotations.Annotator import Annotator

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


"""Loads trained figure 2 data and produces extra plots

Inputs:
  models:
    "models/pancreas_model1/pyrovelocity.pkl"
    "models/pbmc68k_model1/pyrovelocity.pkl"

Outputs:
  figures:
    "reports/fig2/fig2_pancreas_pbmc_uncertainties_comparison.pdf"
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    time_dev_list = []
    mag_dev_list = []
    names = []
    for data_model in conf.reports.model_summary.summarize:
        ##################
        # load data
        ##################
        print(data_model)
        data_model_conf = conf.model_training[data_model]
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

        with open(pyrovelocity_data_path, "rb") as f:
            posterior_samples = pickle.load(f)
        cell_magnitudes = np.sqrt((posterior_samples["vector_field_posterior_samples"]**2).sum(axis=-1))
        cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
        #cell_magnitudes_mean = np.sqrt((posterior_samples["vector_field_posterior_mean"] ** 2).sum(axis=-1))
        cell_magnitudes_std = cell_magnitudes.std(axis=-2)
        cell_magnitudes_deviance = cell_magnitudes_std / cell_magnitudes_mean

        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        cell_time_deviance = cell_time_std / cell_time_mean
        time_dev_list.append(cell_time_deviance)
        mag_dev_list.append(cell_magnitudes_deviance)
        names += [data_model] * len(cell_time_deviance)

    time_dev_list = np.hstack(time_dev_list)
    mag_dev_list = np.hstack(mag_dev_list)

    metrics_df = pd.DataFrame({"time_deviance": time_dev_list, "magnitude_deviance": mag_dev_list, "dataset": names})
    logger.info(metrics_df.head())
    shared_time_plot = conf.reports.figure2_extras.shared_time_plot
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(9.6, 3.5)
    order = ('pancreas_model2', 'pbmc68k_model2')
    sns.boxplot(x='dataset', y='time_deviance', data=metrics_df, ax=ax[0], order=order)
    sns.boxplot(x='dataset', y='magnitude_deviance', data=metrics_df, ax=ax[1], order=order)
    pairs = [('pancreas_model2', 'pbmc68k_model2')]
    time_annotator = Annotator(ax[0], pairs, data=metrics_df, x='dataset', y='time_deviance', order=order)
    time_annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    time_annotator.apply_and_annotate()
    mag_annotator = Annotator(ax[1], pairs, data=metrics_df, x='dataset', y='magnitude_deviance', order=order)
    mag_annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    mag_annotator.apply_and_annotate()

    #ax[1].set_ylim(0, 3)
    print(shared_time_plot)
    fig.savefig(
        shared_time_plot,
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
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figure2.path}\n"
    )
    Path(conf.reports.figure2.path).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(conf.reports.figure2_extras.shared_time_plot):
        logger.info(
            f"\n\nFigure 2 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figure2.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure 2\n\n")
        plots(conf, logger)


if __name__ == "__main__":
    main()
