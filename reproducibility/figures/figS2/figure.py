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
from statannotations.Annotator import Annotator

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import get_posterior_sample_angle_uncertainty
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger


def plots(
    conf: DictConfig,
    logger: Logger,
    fig_name: str = None,
) -> None:
    """Design cell-wise uncertainties metrics across all datasets

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    rayleigh_umap_angles_test = []
    rayleigh_pca_angles_test = []

    for data_model in conf.train_models:
        ##################
        # load data
        ##################
        print(data_model)
        data_model_conf = conf.model_training[data_model]
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
        umap_fdri = np.sum(posterior_samples["fdri"] < 0.05)
        pca_fdri = np.sum(posterior_samples["pca_fdri"] < 0.05)
        num_cells = posterior_samples["fdri"].shape[0]
        rayleigh_umap_angles_test.append(umap_fdri/num_cells)
        rayleigh_pca_angles_test.append(pca_fdri/num_cells)

    cell_uncertainties_metrics = pd.DataFrame({"dataset": conf.train_models, "rayleigh_umap_angles_test": rayleigh_umap_angles_test, "rayleigh_pca_angles_test":rayleigh_pca_angles_test})
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(9.6, 3.5)
    sns.barplot(data=cell_uncertainties_metrics, x='dataset', y='rayleigh_umap_angles_test', ax=ax[0])
    sns.barplot(data=cell_uncertainties_metrics, x='dataset', y='rayleigh_pca_angles_test', ax=ax[1])
    fig.autofmt_xdate(rotation=45)
    fig.savefig(
        fig_name,
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
        f"  reports: {conf.reports.figureS3.path}\n"
    )
    Path(conf.reports.figureS2.path).mkdir(parents=True, exist_ok=True)
    confS2 = conf.reports.figureS2
    print('------test---------')

    if os.path.isfile(confS2.violin_plots_other_lin) and os.path.isfile(confS2.violin_plots_larry_lin):
        logger.info(
            f"\n\nFigure S2 extras outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS2_extras.path}\n"
        )
    else:
        print('------test---------')
        for fig_name in [confS2.violin_plots_other_lin, confS2.violin_plots_larry_lin]:
            plots(
                conf,
                logger,
                fig_name=fig_name
            )


if __name__ == "__main__":
    main()
