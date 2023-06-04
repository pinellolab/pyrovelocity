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
from scipy.stats import pearsonr
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

    fig, ax = plt.subplots(3, len(conf.train_models))
    fig.set_size_inches(28, 8)
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
        umap_magnitude = np.sqrt((mean_arrow_length**2).sum(axis=-1))

        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        umap_cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
        umap_angle_uncertain = get_posterior_sample_angle_uncertainty(umap_cell_angles)

        cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
        cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
        cell_magnitudes_std = cell_magnitudes.std(axis=-2)
        cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
        print(umap_angle_uncertain.shape)

        res = pd.DataFrame(
            {
                "umap_magnitude": umap_magnitude,
                "umap_angle": umap_angle_uncertain,
                "shared_time": cell_time_std,
                "raw_magnitudes_cov": cell_magnitudes_cov,
            }
        )
        sns.regplot(
            x="umap_magnitude",
            y="umap_angle",
            data=res,
            ax=ax[0][index],
            scatter_kws=dict(s=1, linewidth=0),
        )
        r1 = pearsonr(umap_magnitude, umap_angle_uncertain)
        ax[0][index].text(
            0.05,
            0.8,
            f"r={r1[0]:.2f}, p={r1[1]:.2g}",
            transform=ax[0][index].transAxes,
        )
        ax[0][index].set_title(data_model)
        ax[0][index].set_ylim(0, 360)
        r2 = pearsonr(umap_magnitude, cell_time_std)
        sns.regplot(
            x="umap_magnitude",
            y="shared_time",
            data=res,
            ax=ax[1][index],
            scatter_kws=dict(s=1, linewidth=0),
        )
        ax[1][index].text(
            0.05,
            0.8,
            f"r={r2[0]:.2f}, p={r2[1]:.2g}",
            transform=ax[1][index].transAxes,
        )
        r3 = pearsonr(umap_angle_uncertain, cell_magnitudes_cov)
        sns.regplot(
            x="umap_angle",
            y="raw_magnitudes_cov",
            data=res,
            ax=ax[2][index],
            scatter_kws=dict(s=1, linewidth=0),
        )
        ax[2][index].text(
            0.05,
            0.8,
            f"r={r3[0]:.2f}, p={r3[1]:.2g}",
            transform=ax[2][index].transAxes,
        )
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
