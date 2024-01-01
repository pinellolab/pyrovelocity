import os
from logging import Logger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from annoy import AnnoyIndex
from omegaconf import DictConfig
from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.utils import get_pylogger
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


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

        lr = LogisticRegression(
            random_state=42, C=1.0, penalty=None, max_iter=100
        )
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


def compute_distances_and_correlation(
    adata,
    posterior_samples,
    n_neighbors=None,
    max_cells_for_subsample=7000,
    subsample_size=5000,
    distance_metric="euclidean",
    use_approx_nn=True,
    neighborhood_fraction=0.1,
    minimum_neighborhood_size=300,
):
    """
    Compute the correlation between expression distances and temporal differences.

    Parameters:
        adata (anndata.AnnData): The annotated data object.
        posterior_samples (dict): Dictionary containing temporal coordinates samples.
        n_neighbors (int, optional): Number of neighbors to consider for computing distances.
        max_cells_for_subsample (int, optional): Maximum number of cells before subsampling is used.
        subsample_size (int, optional): Number of cells to subsample if n_cells > max_cells_for_subsample.
        distance_metric (str, optional): Metric to use for distance computation. Defaults to 'euclidean'.
        use_approx_nn (bool, optional): Whether to use Annoy for nearest neighbors computation. Defaults to True.
        neighborhood_fraction (float, optional): Fraction of subsample size to be used as neighborhood size if n_neighbors is None. Defaults to 0.1.
        minimum_neighborhood_size (int, optional): Minimum neighborhood size if n_neighbors is None. Defaults to 300.

    Returns:
        list: correlations between expression distances and temporal differences.
        list: p-values associated with the correlations.
    """

    def compute_nearest_neighbors(
        expression_vectors, n_neighbors, distance_metric
    ):
        if use_approx_nn:
            f = expression_vectors.shape[1]
            t = AnnoyIndex(f, metric=distance_metric)

            if hasattr(expression_vectors, "todense"):
                expression_vectors = expression_vectors.todense()

            for i in range(expression_vectors.shape[0]):
                vector = np.array(expression_vectors[i, :]).flatten()
                t.add_item(i, vector)

            t.build(10)
            indices = [
                t.get_nns_by_item(i, n_neighbors)
                for i in range(expression_vectors.shape[0])
            ]
            return indices
        else:
            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors, metric=distance_metric
            ).fit(expression_vectors)
            indices = nbrs.kneighbors(return_distance=False)
            return indices

    expression_vectors = adata.layers["raw_spliced"]
    n_cells, _ = expression_vectors.shape

    if n_neighbors is None:
        n_neighbors = round(
            subsample_size * neighborhood_fraction
            if n_cells > max_cells_for_subsample
            else max(minimum_neighborhood_size, n_cells * neighborhood_fraction)
        )
        print(f"neighborhood size set to {n_neighbors}")

    expression_distances_matrix = None
    if n_cells <= max_cells_for_subsample:
        expression_distances_matrix = pairwise_distances(
            expression_vectors, metric=distance_metric, n_jobs=-1
        )

    correlations = []
    p_values = []
    for temporal_coordinates in tqdm(
        posterior_samples["cell_time"][:3],
        desc="computing distance-time correlations",
    ):
        temporal_coordinates = temporal_coordinates.reshape(-1)

        selected_expression_vectors = expression_vectors
        selected_temporal_coordinates = temporal_coordinates
        selected_expression_distances_matrix = expression_distances_matrix

        if n_cells > max_cells_for_subsample:
            subsample_indices = np.random.choice(
                np.arange(n_cells), subsample_size, replace=False
            )
            selected_expression_vectors = expression_vectors[
                subsample_indices, :
            ]
            selected_temporal_coordinates = temporal_coordinates[
                subsample_indices
            ]
            selected_expression_distances_matrix = pairwise_distances(
                selected_expression_vectors, metric=distance_metric, n_jobs=-1
            )

        n_selected_cells = selected_expression_vectors.shape[0]
        indices = compute_nearest_neighbors(
            selected_expression_vectors, n_neighbors, distance_metric
        )

        selected_expression_distances = selected_expression_distances_matrix[
            np.arange(n_selected_cells)[:, None], indices
        ].flatten()
        selected_temporal_differences = np.abs(
            selected_temporal_coordinates[np.arange(n_selected_cells)[:, None]]
            - selected_temporal_coordinates[indices]
        ).flatten()

        correlation, p_value = spearmanr(
            selected_expression_distances, selected_temporal_differences
        )
        correlations.append(correlation)
        p_values.append(p_value)

    return correlations, p_values


def plot_distance_time_correlation(
    correlations, p_values, file_path, dataset_labels
):
    """
    Create a violin plot of the correlation between expression distances and temporal differences.

    Parameters:
        correlations (list of list of float): A list containing correlation values for multiple datasets.
        p_values (list of list of float): A list containing p-values associated with correlations.
        file_path (str): The path to save the plot.
        dataset_labels (list of str): A list containing labels for each dataset.
    """

    data = []
    for i in range(len(correlations)):
        data.extend(
            {
                "dataset": dataset_labels[i],
                "correlation": correlations[i][j],
                "p_value": p_values[i][j],
            }
            for j in range(len(correlations[i]))
        )
    data = pd.DataFrame(data)

    order = (
        data.groupby("dataset")["correlation"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        x="dataset", y="correlation", data=data, inner="box", ax=ax, order=order
    )

    ax.set_xlabel("dataset")
    ax.set_ylabel("distance-shared time correlation")
    ax.set_title("")

    for ext in ["", ".png"]:
        fig.savefig(
            f"{file_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def profile_function(func, *args, **kwargs):
    import cProfile

    profiler = cProfile.Profile()
    profiler.runctx(
        "func(*args, **kwargs)",
        globals(),
        {"func": func, "args": args, "kwargs": kwargs},
    )
    profiler.dump_stats("profiling_results.out")
    profiler.print_stats(sort="cumtime")


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
            f"\n\nFigure already exists:\n\n"
            f"  {confS2.rayleigh_classifier_plot}\n"
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

    if not distance_time_correlation_plot_exists:
        correlations_all_models = []
        p_values_all_models = []

    for data_model in tqdm(conf.train_models, desc="processing data sets"):
        logger.info(f"\n\nLoading data for {data_model}\n\n")
        if data_model == "pbmc10k_model2_coarse":
            data_model_conf = conf.model_training["pbmc10k_model2"]
        else:
            data_model_conf = conf.model_training[data_model]

        adata, posterior_samples = load_data(data_model_conf)
        # print_anndata(adata)
        # pretty_print_dict(posterior_samples)

        if (
            not distance_time_correlation_plot_exists
            and "_coarse" not in data_model
        ):
            # profile_function(compute_distances_and_correlation, adata, posterior_samples)
            correlations, p_values = compute_distances_and_correlation(
                adata, posterior_samples
            )
            correlations_all_models.append(correlations)
            p_values_all_models.append(p_values)

        if not rayleigh_classifier_plot_exists:
            logger.info(
                f"\n\nComputing Rayleigh statistics for {data_model}\n\n"
            )
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

    if not distance_time_correlation_plot_exists:
        logger.info(f"\n\nCreating {confS2.distance_time_correlation_plot}\n\n")
        dataset_labels = [
            label.split("_model")[0]
            for label in conf.train_models
            if not label.endswith("_coarse")
        ]

        plot_distance_time_correlation(
            correlations_all_models,
            p_values_all_models,
            confS2.distance_time_correlation_plot,
            dataset_labels,
        )

    if not rayleigh_classifier_plot_exists:
        logger.info(f"\n\nCreating {confS2.rayleigh_classifier_plot}\n\n")
        global_uncertainties_aucs = pd.DataFrame(
            {
                "dataset": macro_labels,
                "f1_macro": multiclass_macro_aucs_test_all_models,
            }
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
