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
from pyrovelocity.plot import get_posterior_sample_angle_uncertainty
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger

from scipy.stats import pearsonr


def plots(
    conf: DictConfig,
    logger: Logger,
    fig_name: str = None,
) -> None:
    """Compute cell-wise uncertainty metrics across all datasets

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    fig, ax = plt.subplots(2, len(conf.train_models))
    fig.set_size_inches(28, 5.8)
    for index, data_model in enumerate(conf.train_models):
        ##################
        # load data
        ##################
        print(data_model)

        data_model_conf = conf.model_training[data_model]
        cell_state = data_model_conf.training_parameters.cell_state
        adata_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

        adata = scv.read(adata_data_path)
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
        mean_arrow_length = posterior_samples["vector_field_posterior_mean"]
        magnitude = np.sqrt((mean_arrow_length**2).sum(axis=-1))

        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        umap_cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
        umap_angle_uncertain = get_posterior_sample_angle_uncertainty(umap_cell_angles)
        print(umap_angle_uncertain.shape)

        res = pd.DataFrame({'magnitude': magnitude, 'umap_angle': umap_angle_uncertain, 'shared_time': cell_time_std})
        sns.regplot(x='magnitude', y='umap_angle', data=res, ax=ax[0][index], scatter_kws=dict(s=1, linewidth=0))
        r1 = pearsonr(magnitude, umap_angle_uncertain)
        ax[0][index].text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r1[0], r1[1]),
                   transform=ax[0][index].transAxes)
        r2 = pearsonr(magnitude, cell_time_std)
        sns.regplot(x='magnitude', y='shared_time', data=res, ax=ax[1][index], scatter_kws=dict(s=1, linewidth=0))
        ax[1][index].text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r2[0], r2[1]),
                   transform=ax[1][index].transAxes)
    fig.tight_layout()

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
        f"  reports: {conf.reports.figureS2.path}\n"
    )
    Path(conf.reports.figureS2.path).mkdir(parents=True, exist_ok=True)
    confS2 = conf.reports.figureS2_extra_2

    if os.path.isfile(confS2.mean_length_vs_uncertain):
        logger.info(
            f"\n\nFigure S2 extras outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS2_extra_2}\n"
        )
    else:
        for fig_name in [confS2.mean_length_vs_uncertain]:
            plots(conf, logger, fig_name=fig_name)


if __name__ == "__main__":
    main()
