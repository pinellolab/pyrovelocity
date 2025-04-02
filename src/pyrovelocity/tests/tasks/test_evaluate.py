"""Tests for evaluate.py module."""

import os

from pyrovelocity.tasks.evaluate import (
    calculate_cross_boundary_correctness,
)
from pyrovelocity.workflows.constants import (
    GROUND_TRUTH_TRANSITIONS,
    MODEL_VELOCITY_KEYS,
)

TEST_DATASET_CONFIGS = {
    "pancreas": {"cluster_key": "clusters", "embedding_key": "X_umap"},
}


def test_calculate_cross_boundary_correctness_pancreas(
    adata_postprocessed_pancreas_50_7,
    tmp_path,
):
    """Test cross-boundary correctness calculation for pancreas dataset."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = [
        {
            "data_model": "pancreas_model2",
            "postprocessed_data": adata_postprocessed_pancreas_50_7,
        }
    ]

    summary_file, results_dir, plot_file = calculate_cross_boundary_correctness(
        model_results=model_results,
        output_dir=output_dir,
        dataset_configs=TEST_DATASET_CONFIGS,
        ground_truth_transitions=GROUND_TRUTH_TRANSITIONS,
        model_velocity_keys=MODEL_VELOCITY_KEYS,
    )

    assert os.path.exists(summary_file)
    assert os.path.exists(results_dir)
    assert os.path.exists(plot_file)
    assert os.path.exists(f"{plot_file}.png")

    dataset_results_file = results_dir / "pancreas_CBDir_scores.csv"
    assert os.path.exists(dataset_results_file)


def test_calculate_cross_boundary_correctness_multiple_models(
    adata_postprocessed_pancreas_50_7,
    tmp_path,
):
    """Test cross-boundary correctness calculation with multiple model types."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = [
        {
            "data_model": "pancreas_model1",
            "postprocessed_data": adata_postprocessed_pancreas_50_7,
        },
        {
            "data_model": "pancreas_model2",
            "postprocessed_data": adata_postprocessed_pancreas_50_7,
        },
        {
            "data_model": "pancreas_scvelo",
            "postprocessed_data": adata_postprocessed_pancreas_50_7,
        },
    ]

    summary_file, results_dir, plot_file = calculate_cross_boundary_correctness(
        model_results=model_results,
        output_dir=output_dir,
        dataset_configs=TEST_DATASET_CONFIGS,
        ground_truth_transitions=GROUND_TRUTH_TRANSITIONS,
        model_velocity_keys=MODEL_VELOCITY_KEYS,
    )

    assert os.path.exists(summary_file)
    assert os.path.exists(results_dir)
    assert os.path.exists(plot_file)
    assert os.path.exists(f"{plot_file}.png")

    dataset_results_file = results_dir / "pancreas_CBDir_scores.csv"
    assert os.path.exists(dataset_results_file)


def test_calculate_cross_boundary_correctness_empty_results(
    tmp_path,
):
    """Test cross-boundary correctness calculation with empty results."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results = []

    summary_file, results_dir, plot_file = calculate_cross_boundary_correctness(
        model_results=model_results,
        output_dir=output_dir,
        dataset_configs=TEST_DATASET_CONFIGS,
        ground_truth_transitions=GROUND_TRUTH_TRANSITIONS,
        model_velocity_keys=MODEL_VELOCITY_KEYS,
    )

    assert os.path.exists(results_dir)
