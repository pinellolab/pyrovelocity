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

DEFAULT_COLOR_PALETTE = [
    "#0B559F",
    "#1D9E74",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
]


@beartype
def plot_cross_boundary_correctness(
    plot_df: pd.DataFrame,
    output_path: str | Path,
    order: List[str],
    model_order: List[str],
    color_palette: List[str] = DEFAULT_COLOR_PALETTE,
    default_fontsize: int = 16,
) -> Path:
    """Create a summary plot of cross-boundary correctness metrics.

    Args:
        plot_df: DataFrame in long format with columns Model, Dataset, and
            Cross Boundary Direction Correctness
        output_path: Path to save the plot
        order: Optional order of datasets in the plot
        model_order: Optional order of models in the plot
        color_palette: Optional list of colors to use for the models. If provided,
            must be at least as long as the number of models. If not provided,
            default colors will be used.
        default_fontsize: Font size for the plot

    Returns:
        Path to the saved plot
    """
    num_datasets = len(plot_df["Dataset"].unique())
    fig_width = max(15, num_datasets * 2)

    plt.figure(figsize=(fig_width, 8))

    order = [
        dataset for dataset in order if dataset in plot_df["Dataset"].unique()
    ]

    for dataset in plot_df["Dataset"].unique():
        if dataset not in order:
            order.append(dataset)

    hue_order = [
        model for model in model_order if model in plot_df["Model"].unique()
    ]

    for model in plot_df["Model"].unique():
        if model not in hue_order:
            hue_order.append(model)

    palette = {}

    for i, model in enumerate(hue_order):
        if i < len(color_palette):
            palette[model] = color_palette[i]
        else:
            logger.error(
                f"Not enough colors provided in color_palette, "
                f"expected at least {len(hue_order)}"
            )

    ax = sns.barplot(
        data=plot_df,
        x="Dataset",
        y="Cross Boundary Direction Correctness",
        hue="Model",
        order=order,
        hue_order=hue_order,
        palette=palette,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    plt.ylabel(
        "Cross Boundary Direction Correctness",
        fontsize=default_fontsize,
        fontweight="bold",
    )
    plt.xlabel("", fontsize=default_fontsize, fontweight="bold")

    labels = ax.get_xticklabels()
    new_labels = []

    for label in labels:
        text = label.get_text()
        formatted_text = text.replace("mono", "monocytes")
        formatted_text = formatted_text.replace("neu", "neutrophils")
        formatted_text = formatted_text.replace("_", " ").capitalize()
        formatted_text = formatted_text.replace("Larry", "LARRY")

        new_labels.append(formatted_text)

    ax.set_xticklabels(
        new_labels, rotation=0, ha="center", fontsize=default_fontsize
    )

    plt.yticks(fontsize=default_fontsize)

    plt.legend(
        title="",
        fontsize=default_fontsize,
        loc="upper right",
    )

    plt.tight_layout(pad=2.0)

    output_path = Path(output_path)
    for ext in ["", ".png"]:
        plt.savefig(f"{output_path}{ext}", bbox_inches="tight", dpi=300)

    plt.close()
    return output_path


@beartype
def calculate_cross_boundary_correctness(
    model_results: List[Dict[str, str | Path | AnnData]],
    output_dir: str | Path,
    dataset_configs: Dict[str, Dict[str, str]],
    ground_truth_transitions: Dict[str, List[Tuple[str, str]]],
    model_velocity_keys: Dict[str, str],
    color_palette: List[str] = DEFAULT_COLOR_PALETTE,
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
        dataset_configs: Mapping of dataset names to their cluster and embedding keys
        ground_truth_transitions: Mapping of dataset names to their ground truth cell transitions
        model_velocity_keys: Mapping of model types to their velocity keys
        color_palette: Optional list of colors to use for models in the plot

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

    results_by_dataset = {}

    dataset_samples = {}

    all_model_types = set()

    for result in model_results:
        data_model = result["data_model"]
        postprocessed_data_path = result["postprocessed_data"]

        dataset_name, model_type = parse_data_model(data_model)

        all_model_types.add(model_type)

        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = {}

        dataset_samples[dataset_name] = postprocessed_data_path

        if not is_valid_dataset(
            dataset_name, ground_truth_transitions, dataset_configs
        ):
            logger.warning(
                f"No ground truth transitions or configuration defined for {dataset_name}, skipping."
            )
            continue

        cb_mean = process_model_for_dataset(
            dataset_name=dataset_name,
            model_type=model_type,
            adata_path=postprocessed_data_path,
            dataset_configs=dataset_configs,
            ground_truth_transitions=ground_truth_transitions,
            model_velocity_keys=model_velocity_keys,
            results_dir=results_dir,
        )

        results_by_dataset[dataset_name][model_type] = cb_mean

    for dataset_name, sample_path in dataset_samples.items():
        if (
            "scvelo" not in results_by_dataset.get(dataset_name, {})
            and dataset_name in dataset_configs
        ):
            add_benchmark_model_results(
                dataset_name=dataset_name,
                sample_path=sample_path,
                model_type="scvelo",
                results_by_dataset=results_by_dataset,
                dataset_configs=dataset_configs,
                ground_truth_transitions=ground_truth_transitions,
                model_velocity_keys=model_velocity_keys,
                results_dir=results_dir,
            )
            all_model_types.add("scvelo")

    summary_file, plot_file = generate_summary_outputs(
        results_by_dataset=results_by_dataset,
        output_dir=output_dir,
        all_model_types=all_model_types,
        ground_truth_transitions=ground_truth_transitions,
        model_velocity_keys=model_velocity_keys,
        color_palette=color_palette,
    )

    return summary_file, results_dir, plot_file


@beartype
def parse_data_model(data_model: str) -> Tuple[str, str]:
    """
    Parse a data_model string to extract dataset name and model type.

    Args:
        data_model: String in format like 'pancreas_model1', 'larry_neu_model2'

    Returns:
        Tuple of (dataset_name, model_type)
    """
    if "_model" in data_model:
        parts = data_model.split("_model")
        return parts[0], f"model{parts[1]}"

    for model_prefix in ["_scvelo", "_velovi", "_regvelo"]:
        if model_prefix in data_model:
            parts = data_model.split(model_prefix)
            return parts[0], model_prefix.lstrip("_")

    parts = data_model.split("_")
    if len(parts) > 1:
        return parts[0], "_".join(parts[1:])

    return data_model, "unknown"


@beartype
def is_valid_dataset(
    dataset_name: str,
    ground_truth_transitions: Dict[str, List[Tuple[str, str]]],
    dataset_configs: Dict[str, Dict[str, str]],
) -> bool:
    """Check if a dataset has required configuration."""
    return (
        dataset_name in ground_truth_transitions
        and dataset_name in dataset_configs
    )


@beartype
def process_model_for_dataset(
    dataset_name: str,
    model_type: str,
    adata_path: str | Path | AnnData,
    dataset_configs: Dict[str, Dict[str, str]],
    ground_truth_transitions: Dict[str, List[Tuple[str, str]]],
    model_velocity_keys: Dict[str, str],
    results_dir: Path,
) -> float:
    """
    Process a model for a specific dataset and calculate cross-boundary correctness.

    Args:
        dataset_name: Name of the dataset
        model_type: Type of the model (model1, model2, scvelo, etc.)
        adata_path: Path to AnnData file or AnnData object
        dataset_configs: Configuration for datasets
        ground_truth_transitions: Ground truth transitions for datasets
        model_velocity_keys: Mapping of model types to velocity keys
        results_dir: Directory to save results

    Returns:
        Mean cross-boundary correctness score
    """
    logger.info(f"Processing {dataset_name} with model {model_type}")

    config = dataset_configs[dataset_name]
    cluster_key = config["cluster_key"]
    embedding_key = config["embedding_key"]
    transitions = ground_truth_transitions[dataset_name]

    if isinstance(adata_path, AnnData):
        adata = adata_path
    else:
        adata = sc.read_h5ad(adata_path)

    if model_type in model_velocity_keys:
        velocity_key = model_velocity_keys[model_type]
        logger.info(f"Using velocity key {velocity_key} for {model_type}")
    else:
        logger.warning(
            f"Model type {model_type} not found in model_velocity_keys, using default velocity key"
        )
        velocity_key = "velocity"

    if velocity_key not in adata.layers:
        logger.warning(
            f"Velocity key '{velocity_key}' not found in {dataset_name} AnnData layers, skipping"
        )

        cb_mean = 0.0
        empty_scores = {(u, v): [0.0] for u, v in transitions}
        save_results_to_csv(
            dataset_name=dataset_name,
            model_type=model_type,
            cb_scores=empty_scores,
            cb_mean=cb_mean,
            transitions=transitions,
            results_dir=results_dir,
        )
        return cb_mean

    cb_scores, cb_mean = cross_boundary_correctness(
        adata=adata,
        k_cluster=cluster_key,
        cluster_edges=transitions,
        k_velocity=velocity_key,
        x_emb=embedding_key,
    )

    save_results_to_csv(
        dataset_name=dataset_name,
        model_type=model_type,
        cb_scores=cb_scores,
        cb_mean=cb_mean,
        transitions=transitions,
        results_dir=results_dir,
    )

    return cb_mean


@beartype
def save_results_to_csv(
    dataset_name: str,
    model_type: str,
    cb_scores: Dict[Tuple[str, str], float],
    cb_mean: float,
    transitions: List[Tuple[str, str]],
    results_dir: Path,
) -> None:
    """Save results for a dataset/model to CSV file."""
    dataset_results_file = results_dir / f"{dataset_name}_CBDir_scores.csv"

    transition_scores = [
        np.mean(cb_scores.get((u, v), [0])) for u, v in transitions
    ]
    row_data = [model_type] + transition_scores + [cb_mean]

    if dataset_results_file.exists():
        dataset_df = pd.read_csv(dataset_results_file)

        if model_type not in dataset_df["model"].values:
            dataset_df.loc[len(dataset_df)] = row_data
            dataset_df.to_csv(dataset_results_file, index=False)
            logger.info(f"Updated results for {dataset_name} with {model_type}")
    else:
        dataset_df = pd.DataFrame(
            columns=["model"] + [f"{u}->{v}" for u, v in transitions] + ["Mean"]
        )
        dataset_df.loc[len(dataset_df)] = row_data
        dataset_df.to_csv(dataset_results_file, index=False)
        logger.info(f"Created results file for {dataset_name}")


@beartype
def generate_summary_outputs(
    results_by_dataset: Dict[str, Dict[str, float]],
    output_dir: Path,
    all_model_types: set,
    ground_truth_transitions: Dict[str, List[Tuple[str, str]]],
    model_velocity_keys: Dict[str, str],
    color_palette: List[str] = DEFAULT_COLOR_PALETTE,
) -> Tuple[Path, Path]:
    """
    Generate summary CSV and plot from all results.

    Args:
        results_by_dataset: Results organized by dataset and model
        output_dir: Directory to save outputs
        all_model_types: Set of all model types
        ground_truth_transitions: Mapping of dataset names to ground truth transitions
        model_velocity_keys: Mapping of model types to velocity keys, used for ordering models
        color_palette: Optional list of colors to use for models in the plot

    Returns:
        Tuple of (summary_file_path, plot_file_path)
    """
    summary_file = output_dir / "cross_boundary_correctness_summary.csv"

    summary_data = []

    if model_velocity_keys:
        model_order = list(model_velocity_keys.keys())
        for model_type in all_model_types:
            if model_type not in model_order:
                model_order.append(model_type)
    else:
        model_order = sorted(all_model_types)

    for model_type in model_order:
        if model_type not in all_model_types:
            continue

        row = {"Model": model_type}
        valid_scores = []

        for dataset_name, model_scores in results_by_dataset.items():
            if model_type in model_scores:
                score = model_scores[model_type]
                row[dataset_name] = score
                valid_scores.append(score)

        if valid_scores:
            row["mean"] = np.mean(valid_scores)

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        summary_df.set_index("Model", inplace=True)
        summary_df.to_csv(summary_file)
        logger.info(f"Saved summary to {summary_file}")

        plot_df = create_plot_dataframe(summary_df)

        dataset_order = ["mean"]

        for dataset in ground_truth_transitions.keys():
            if dataset in summary_df.columns:
                dataset_order.append(dataset)

        for dataset in summary_df.columns:
            if dataset != "mean" and dataset not in dataset_order:
                dataset_order.append(dataset)

        plot_file = output_dir / "cross_boundary_correctness_plot.pdf"
        plot_path = plot_cross_boundary_correctness(
            plot_df=plot_df,
            output_path=plot_file,
            order=dataset_order,
            model_order=model_order,
            color_palette=color_palette,
        )
        logger.info(f"Created plot at {plot_file}")
    else:
        plot_file = output_dir / "no_plot.pdf"
        logger.error(f"No data to plot")

    return summary_file, plot_file


@beartype
def create_plot_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Convert summary DataFrame to long format for plotting."""
    plot_data = []

    for model in summary_df.index:
        for dataset in summary_df.columns:
            if pd.notna(summary_df.loc[model, dataset]):
                plot_data.append(
                    {
                        "Model": model,
                        "Dataset": dataset,
                        "Cross Boundary Direction Correctness": summary_df.loc[
                            model, dataset
                        ],
                    }
                )

    return pd.DataFrame(plot_data)


@beartype
def add_benchmark_model_results(
    dataset_name: str,
    sample_path: str | Path | AnnData,
    model_type: str,
    results_by_dataset: Dict[str, Dict[str, float]],
    dataset_configs: Dict[str, Dict[str, str]],
    ground_truth_transitions: Dict[str, List[Tuple[str, str]]],
    model_velocity_keys: Dict[str, str],
    results_dir: Path,
) -> None:
    """
    Add results for benchmark models by extracting from existing AnnData.

    Args:
        dataset_name: Name of the dataset
        sample_path: Path to an AnnData file or AnnData object with the benchmark model results
        model_type: Benchmark model type (e.g., "scvelo")
        results_by_dataset: Dictionary to store results
        dataset_configs: Configuration for datasets
        ground_truth_transitions: Ground truth transitions for datasets
        model_velocity_keys: Mapping of model types to velocity keys
        results_dir: Directory to save individual results
    """
    logger.info(f"Adding {model_type} results for dataset {dataset_name}")

    config = dataset_configs[dataset_name]
    cluster_key = config["cluster_key"]
    embedding_key = config["embedding_key"]
    transitions = ground_truth_transitions[dataset_name]

    if isinstance(sample_path, AnnData):
        adata = sample_path
    else:
        adata = sc.read_h5ad(sample_path)

    if model_type in model_velocity_keys:
        velocity_key = model_velocity_keys[model_type]
    else:
        logger.warning(
            f"Model type {model_type} not found in model_velocity_keys, using default velocity key"
        )
        velocity_key = "velocity"

    if velocity_key not in adata.layers:
        logger.error(
            f"Velocity key '{velocity_key}' not found in {dataset_name} AnnData layers"
        )

    cb_scores, cb_mean = cross_boundary_correctness(
        adata=adata,
        k_cluster=cluster_key,
        cluster_edges=transitions,
        k_velocity=velocity_key,
        x_emb=embedding_key,
    )

    if dataset_name not in results_by_dataset:
        results_by_dataset[dataset_name] = {}
    results_by_dataset[dataset_name][model_type] = cb_mean

    save_results_to_csv(
        dataset_name=dataset_name,
        model_type=model_type,
        cb_scores=cb_scores,
        cb_mean=cb_mean,
        transitions=transitions,
        results_dir=results_dir,
    )
