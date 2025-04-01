from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict, List, Tuple

from pyrovelocity.logging import configure_logging
from pyrovelocity.metrics.trajectory import cross_boundary_correctness

__all__ = ["calculate_cross_boundary_correctness"]

logger = configure_logging(__name__)

GROUND_TRUTH_TRANSITIONS = {
    "pancreas": [
        ("Ngn3 high EP", "Pre-endocrine"),
        ("Pre-endocrine", "Alpha"),
        ("Pre-endocrine", "Beta"),
        ("Pre-endocrine", "Delta"),
        ("Pre-endocrine", "Epsilon"),
    ],
    "bonemarrow": [("HSC_1", "Ery_1"), ("HSC_1", "HSC_2"), ("Ery_1", "Ery_2")],
    "pons": [("COPs", "NFOLs"), ("NFOLs", "MFOLs")],
    "larry_mono": [("Undifferentiated", "Monocyte")],
    "larry_neu": [("Undifferentiated", "Neutrophil")],
    "larry_multilineage": [
        ("Undifferentiated", "Monocyte"),
        ("Undifferentiated", "Neutrophil"),
    ],
    "larry": [
        ("Undifferentiated", "Monocyte"),
        ("Undifferentiated", "Neutrophil"),
        ("Undifferentiated", "pDC"),
        ("Undifferentiated", "Meg"),
        ("Undifferentiated", "Mast"),
        ("Undifferentiated", "Lymphoid"),
        ("Undifferentiated", "Erythroid"),
        ("Undifferentiated", "Eos"),
        ("Undifferentiated", "Ccr7_DC"),
        ("Undifferentiated", "Baso"),
    ],
}

DATASET_CONFIGS = {
    "pancreas": {"cluster_key": "clusters", "embedding_key": "X_umap"},
    "bonemarrow": {"cluster_key": "clusters", "embedding_key": "X_umap"},
    "pons": {"cluster_key": "celltype", "embedding_key": "X_umap"},
    "larry_mono": {"cluster_key": "state_info", "embedding_key": "X_emb"},
    "larry_neu": {"cluster_key": "state_info", "embedding_key": "X_emb"},
    "larry_multilineage": {
        "cluster_key": "state_info",
        "embedding_key": "X_emb",
    },
    "larry": {"cluster_key": "state_info", "embedding_key": "X_emb"},
}


@beartype
def calculate_cross_boundary_correctness(
    model_results: List[Dict[str, str | Path | AnnData]],
    output_dir: str | Path,
) -> Tuple[Path, Path, Path]:
    """
    Calculate cross-boundary correctness metrics for multiple datasets and models.

    Args:
        model_results: List of dictionaries containing model results.
            Each dictionary should have:
            - data_model: str - Dataset and model name (e.g., 'pancreas_model1')
            - postprocessed_data: Union[str, Path, AnnData] - Path to postprocessed AnnData file
              or the actual AnnData object
        output_dir: Directory to save results to

    Returns:
        Tuple of paths to:
        - Summary CSV file
        - Individual dataset results directory
        - Plot file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = output_dir / "cb_correctness_results"
    results_dir.mkdir(exist_ok=True)

    summary_df = pd.DataFrame()

    dataset_models = {}
    for result in model_results:
        data_model = result["data_model"]
        dataset_name = data_model.split("_model")[0]
        model_name = data_model.split("_")[-1]

        if dataset_name not in dataset_models:
            dataset_models[dataset_name] = []

        dataset_models[dataset_name].append(
            {
                "model_name": model_name,
                "data_model": data_model,
                "postprocessed_data": result["postprocessed_data"],
            }
        )

    for dataset_name, models in dataset_models.items():
        if (
            dataset_name not in GROUND_TRUTH_TRANSITIONS
            or dataset_name not in DATASET_CONFIGS
        ):
            logger.warning(
                f"No ground truth transitions or configuration defined for {dataset_name}, skipping."
            )
            continue

        config = DATASET_CONFIGS[dataset_name]
        cluster_key = config["cluster_key"]
        embedding_key = config["embedding_key"]
        transitions = GROUND_TRUTH_TRANSITIONS[dataset_name]

        dataset_results_file = results_dir / f"{dataset_name}_CBDir_scores.csv"

        dataset_df = pd.DataFrame(
            columns=["model"] + [f"{u}->{v}" for u, v in transitions] + ["Mean"]
        )
        for model_data in models:
            model_name = model_data["model_name"]
            adata_path_or_obj = model_data["postprocessed_data"]

            logger.info(f"Processing {dataset_name} with model {model_name}")

            if isinstance(adata_path_or_obj, AnnData):
                adata = adata_path_or_obj
            else:
                adata = sc.read_h5ad(adata_path_or_obj)

            velocity_key = (
                "velocity_pyro"
                if model_name in ["model1", "model2"]
                else "velocity"
            )

            cb_scores, cb_mean = cross_boundary_correctness(
                adata=adata,
                k_cluster=cluster_key,
                cluster_edges=transitions,
                k_velocity=velocity_key,
                x_emb=embedding_key,
            )

            transition_scores = [
                np.mean(cb_scores.get((u, v), [0])) for u, v in transitions
            ]
            row_data = [model_name] + transition_scores + [cb_mean]

            dataset_df.loc[len(dataset_df)] = row_data

        dataset_df.to_csv(dataset_results_file, index=False)
        logger.info(
            f"Saved results for {dataset_name} to {dataset_results_file}"
        )

        mean_column = dataset_df.set_index("model")["Mean"]
        summary_df[dataset_name] = mean_column

    if not summary_df.empty:
        summary_df["Mean Across All Data"] = summary_df.mean(axis=1)

    summary_file = output_dir / "cross_boundary_correctness_summary.csv"
    summary_df.to_csv(summary_file)

    if not summary_df.empty:
        plot_file = output_dir / "cross_boundary_correctness_plot.pdf"
        plot_path = create_summary_plot(summary_df, plot_file)
    else:
        plot_path = output_dir / "empty_plot.pdf"
        plt.figure()
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.savefig(plot_path)
        plt.close()

    return summary_file, results_dir, plot_path


def create_summary_plot(
    summary_df: pd.DataFrame, output_path: str | Path
) -> Path:
    """Create a summary plot of cross-boundary correctness metrics."""
    plot_df = summary_df.reset_index().melt(
        id_vars="index",
        var_name="Dataset",
        value_name="Cross Boundary Direction Correctness",
    )
    plot_df.rename(columns={"index": "Model"}, inplace=True)

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        data=plot_df,
        x="Dataset",
        y="Cross Boundary Direction Correctness",
        hue="Model",
        order=["Mean Across All Data"]
        + [col for col in summary_df.columns if col != "Mean Across All Data"],
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend(title="")

    output_path = Path(output_path)
    for ext in ["", ".png"]:
        plt.savefig(f"{output_path}{ext}", bbox_inches="tight")

    plt.close()
    return output_path
