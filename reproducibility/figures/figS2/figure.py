import os
from logging import Logger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import pretty_print_dict
from pyrovelocity.utils import print_anndata


def load_data(data_model_conf):
    adata_data_path = data_model_conf.trained_data_path
    pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

    adata = scv.read(adata_data_path)
    posterior_samples = CompressedPickle.load(pyrovelocity_data_path)

    return adata, posterior_samples


def train_classifier(posterior_samples, sample_size, adata, cell_state):
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

    return multiclass_macro_aucs_test


def compute_rayleigh_statistics(
    data_model, data_model_conf, sample_size, posterior_samples
):
    macro_labels = [data_model] * sample_size
    if data_model == "pbmc10k_model2_coarse":
        cell_state = "celltype_low_resolution"
    else:
        cell_state = data_model_conf.training_parameters.cell_state

    umap_fdri = np.sum(posterior_samples["fdri"] < 0.001)
    pca_fdri = np.sum(posterior_samples["pca_fdri"] < 0.001)
    num_cells = posterior_samples["fdri"].shape[0]

    rayleigh_umap_angle = umap_fdri / num_cells
    rayleigh_pca_angle = pca_fdri / num_cells

    return macro_labels, rayleigh_umap_angle, rayleigh_pca_angle, cell_state


def make_rayleigh_classifier_plot(
    cell_uncertainties_metrics, global_uncertainties_aucs, fig_name
):
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

    grouped_index = (
        global_uncertainties_aucs.groupby("dataset")
        .median()
        .sort_values(by="f1_macro")
        .index
    )

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


import time

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


def compute_distances_and_correlation(adata, posterior_samples):
    expression_vectors = adata.layers["raw_spliced"]
    n_cells, n_genes = expression_vectors.shape

    max_cells_for_subsample = 7000
    subsample_size = 5000
    distance_metric = "euclidean"

    if n_cells <= max_cells_for_subsample:
        expression_distances_matrix = pairwise_distances(
            expression_vectors, metric=distance_metric, n_jobs=-1
        )

    correlations = []
    p_values = []
    for temporal_coordinates in posterior_samples["cell_time"][:2]:
        temporal_coordinates = temporal_coordinates.reshape(-1)
        sort_indices = np.argsort(temporal_coordinates)

        if n_cells > max_cells_for_subsample:
            subsample_indices = np.random.choice(
                sort_indices, subsample_size, replace=False
            )
            sorted_subsample_indices = np.argsort(
                temporal_coordinates[subsample_indices]
            )
            final_subsample_indices = subsample_indices[sorted_subsample_indices]

            selected_expression_vectors = expression_vectors[final_subsample_indices, :]
            selected_temporal_coordinates = temporal_coordinates[
                final_subsample_indices
            ]

            selected_expression_distances_matrix = pairwise_distances(
                selected_expression_vectors, metric="euclidean", n_jobs=-1
            )
        else:
            selected_expression_vectors = expression_vectors[sort_indices, :]
            selected_temporal_coordinates = temporal_coordinates[sort_indices]
            selected_expression_distances_matrix = expression_distances_matrix[
                sort_indices, :
            ][:, sort_indices]

        n_selected_cells = selected_expression_vectors.shape[0]

        selected_expression_distances = selected_expression_distances_matrix[
            np.triu_indices(n_selected_cells, k=1)
        ]

        temporal_differences_matrix = (
            selected_temporal_coordinates - selected_temporal_coordinates.reshape(-1, 1)
        )
        temporal_differences = temporal_differences_matrix[
            np.triu_indices(n_selected_cells, k=1)
        ]

        correlation, p_value = spearmanr(
            selected_expression_distances, temporal_differences
        )
        correlations.append(correlation)
        p_values.append(p_value)

    return correlations, p_values
    # return correlations, p_values, time_dist, time_temporal, time_corr, total_time


def plots(conf: DictConfig, logger: Logger) -> None:
    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  figure S2: {conf.reports.figureS2.path}\n"
    )
    Path(conf.reports.figureS2.path).mkdir(parents=True, exist_ok=True)

    confS2 = conf.reports.figureS2
    print_config_tree(confS2, logger, ())

    if os.path.isfile(confS2.rayleigh_classifier_plot):
        logger.info(
            f"\n\nFigure already exists:\n\n" f"  {confS2.rayleigh_classifier_plot}\n"
        )
        rayleigh_classifier_plot_exists = True
    else:
        rayleigh_classifier_plot_exists = False

    if os.path.isfile(confS2.distance_time_correlation_plot):
        logger.info(
            f"\n\nFigure already exists:\n\n"
            f"  {confS2.distance_time_correlation_plot}\n"
        )
        distance_time_correlation_plot_exists = True
    else:
        distance_time_correlation_plot_exists = False

    if rayleigh_classifier_plot_exists & distance_time_correlation_plot_exists:
        logger.info(
            f"\n\nFigure S2 extras outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS2.path}\n"
        )
        return

    if not rayleigh_classifier_plot_exists:
        rayleigh_umap_angles_test = []
        rayleigh_pca_angles_test = []
        multiclass_macro_aucs_test_all_models = []
        macro_labels = []
        sample_size = 30

    for data_model in conf.train_models:
        logger.info(f"\n\nLoading data for {data_model}\n\n")
        if data_model == "pbmc10k_model2_coarse":
            data_model_conf = conf.model_training["pbmc10k_model2"]
        else:
            data_model_conf = conf.model_training[data_model]

        adata, posterior_samples = load_data(data_model_conf)
        print_anndata(adata)
        pretty_print_dict(posterior_samples)
        correlations, p_values = compute_distances_and_correlation(
            adata, posterior_samples
        )
        breakpoint()

        if not rayleigh_classifier_plot_exists:
            logger.info(f"\n\nComputing Rayleigh statistics for {data_model}\n\n")
            (
                m_labels,
                rayleigh_umap,
                rayleigh_pca,
                cell_state,
            ) = compute_rayleigh_statistics(
                data_model, data_model_conf, sample_size, posterior_samples
            )
            macro_labels += m_labels
            rayleigh_umap_angles_test.append(rayleigh_umap)
            rayleigh_pca_angles_test.append(rayleigh_pca)

            logger.info(f"\n\nTraining classifier for {data_model}\n\n")
            multiclass_macro_aucs_test = train_classifier(
                posterior_samples, sample_size, adata, cell_state
            )
            multiclass_macro_aucs_test_all_models += multiclass_macro_aucs_test

    if not rayleigh_classifier_plot_exists:
        logger.info(f"\n\nCreating {confS2.rayleigh_classifier_plot}\n\n")
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

        make_rayleigh_classifier_plot(
            cell_uncertainties_metrics,
            global_uncertainties_aucs,
            confS2.rayleigh_classifier_plot,
        )


@hydra.main(version_base="1.2", config_path="..", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plot results
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    plots(conf, logger)


if __name__ == "__main__":
    main()
