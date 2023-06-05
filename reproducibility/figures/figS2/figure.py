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

    rayleigh_umap_angles_test = []
    rayleigh_pca_angles_test = []
    multiclass_macro_aucs_test_all_models = []
    macro_labels = []
    sample_size = 30

    for data_model in conf.train_models:
        ##################
        # load data
        ##################
        print(data_model)

        if data_model == "pbmc10k_model2_coarse":
            macro_labels += [data_model] * sample_size
            data_model = "pbmc10k_model2"
            data_model_conf = conf.model_training[data_model]
            cell_state = "celltype_low_resolution"
        else:
            macro_labels += [data_model] * sample_size
            data_model_conf = conf.model_training[data_model]
            cell_state = data_model_conf.training_parameters.cell_state
        adata_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

        adata = scv.read(adata_data_path)
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
        umap_fdri = np.sum(posterior_samples["fdri"] < 0.001)
        pca_fdri = np.sum(posterior_samples["pca_fdri"] < 0.001)
        num_cells = posterior_samples["fdri"].shape[0]
        rayleigh_umap_angles_test.append(umap_fdri / num_cells)
        rayleigh_pca_angles_test.append(pca_fdri / num_cells)

        print(posterior_samples["cell_time"].shape)
        multiclass_macro_aucs_test = []
        for sample in range(sample_size):
            le = LabelEncoder()
            y_sample = le.fit_transform(adata.obs[cell_state].values)

            lr = LogisticRegression(random_state=42, C=1.0, penalty=None, max_iter=100)
            x_all = posterior_samples["cell_time"][sample]
            macro_roc_auc_ovr_crossval = cross_val_score(
                lr, x_all, y_sample, cv=5, scoring="f1_macro"
            )
            multiclass_macro_aucs_test.append(np.mean(macro_roc_auc_ovr_crossval))
        multiclass_macro_aucs_test_all_models += multiclass_macro_aucs_test

    print(len(macro_labels))
    print(len(multiclass_macro_aucs_test_all_models))
    print(multiclass_macro_aucs_test_all_models)
    global_uncertainties_aucs = pd.DataFrame(
        {"dataset": macro_labels, "f1_macro": multiclass_macro_aucs_test_all_models}
    )
    cell_uncertainties_metrics = pd.DataFrame(
        {
            "dataset": conf.train_models,
            "rayleigh_umap_angles_test": rayleigh_umap_angles_test,
            "rayleigh_pca_angles_test": rayleigh_pca_angles_test,
        }
    )
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 3.5)
    sns.barplot(
        data=cell_uncertainties_metrics,
        x="dataset",
        y="rayleigh_umap_angles_test",
        ax=ax[0],
    )
    sns.barplot(
        data=cell_uncertainties_metrics,
        x="dataset",
        y="rayleigh_pca_angles_test",
        ax=ax[1],
    )

    print(global_uncertainties_aucs.groupby("dataset").median())
    grouped_index = (
        global_uncertainties_aucs.groupby("dataset")
        .median()
        .sort_values(by="f1_macro")
        .index
    )
    print(grouped_index)
    sns.boxplot(
        data=global_uncertainties_aucs,
        x="dataset",
        y="f1_macro",
        ax=ax[2],
        order=grouped_index,
    )
    fig.autofmt_xdate(rotation=45)
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
    confS2 = conf.reports.figureS2

    if os.path.isfile(confS2.boxplot_other_lin):
        logger.info(
            f"\n\nFigure S2 extras outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS2.path}\n"
        )
    else:
        for fig_name in [confS2.boxplot_other_lin]:
            plots(conf, logger, fig_name=fig_name)


if __name__ == "__main__":
    main()
