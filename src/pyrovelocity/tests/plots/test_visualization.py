"""
Tests for the visualization module for model comparison.

This module contains tests for the visualization functions in the
pyrovelocity.plots.visualization module, which provides tools for
visualizing model comparison results.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

from pyrovelocity.models.comparison import BayesianModelComparison, ComparisonResult
from pyrovelocity.plots.visualization import (
    plot_model_comparison,
    plot_model_comparison_grid,
    plot_pointwise_comparison,
    plot_posterior_predictive_check,
    plot_diagnostic_metrics,
    plot_waic_loo_comparison,
)


@pytest.fixture
def comparison_result():
    """Create a sample ComparisonResult for testing."""
    values = {"model1": 100.0, "model2": 110.0, "model3": 90.0}
    standard_errors = {"model1": 5.0, "model2": 6.0, "model3": 4.5}
    differences = {
        "model1": {"model2": -10.0, "model3": 10.0},
        "model2": {"model1": 10.0, "model3": 20.0},
        "model3": {"model1": -10.0, "model2": -20.0},
    }
    
    return ComparisonResult(
        metric_name="WAIC",
        values=values,
        differences=differences,
        standard_errors=standard_errors,
    )


@pytest.fixture
def comparison_results_list():
    """Create a list of sample ComparisonResults for testing."""
    # WAIC result
    waic_values = {"model1": 100.0, "model2": 110.0, "model3": 90.0}
    waic_standard_errors = {"model1": 5.0, "model2": 6.0, "model3": 4.5}
    waic_differences = {
        "model1": {"model2": -10.0, "model3": 10.0},
        "model2": {"model1": 10.0, "model3": 20.0},
        "model3": {"model1": -10.0, "model2": -20.0},
    }
    
    waic_result = ComparisonResult(
        metric_name="WAIC",
        values=waic_values,
        differences=waic_differences,
        standard_errors=waic_standard_errors,
    )
    
    # LOO result
    loo_values = {"model1": 95.0, "model2": 105.0, "model3": 85.0}
    loo_standard_errors = {"model1": 4.0, "model2": 5.0, "model3": 3.5}
    loo_differences = {
        "model1": {"model2": -10.0, "model3": 10.0},
        "model2": {"model1": 10.0, "model3": 20.0},
        "model3": {"model1": -10.0, "model2": -20.0},
    }
    
    loo_result = ComparisonResult(
        metric_name="LOO",
        values=loo_values,
        differences=loo_differences,
        standard_errors=loo_standard_errors,
    )
    
    # Bayes factor result
    bf_values = {"model1": 1.0, "model2": 0.5, "model3": 2.0}
    bf_differences = {
        "model1": {"model2": 0.5, "model3": -1.0},
        "model2": {"model1": -0.5, "model3": -1.5},
        "model3": {"model1": 1.0, "model2": 1.5},
    }
    
    bf_result = ComparisonResult(
        metric_name="Bayes Factor",
        values=bf_values,
        differences=bf_differences,
    )
    
    return [waic_result, loo_result, bf_result]


@pytest.fixture
def pointwise_data():
    """Create sample pointwise data for testing."""
    np.random.seed(42)
    model1_pointwise = np.random.normal(10, 2, 100)
    model2_pointwise = np.random.normal(12, 3, 100)
    
    return {
        "model1_name": "Model A",
        "model2_name": "Model B",
        "model1_pointwise": model1_pointwise,
        "model2_pointwise": model2_pointwise,
    }


@pytest.fixture
def posterior_predictive_data():
    """Create sample data for posterior predictive checks."""
    np.random.seed(42)
    observed_data = np.random.normal(0, 1, 50)
    
    # Generate 100 posterior samples
    predicted_data = np.zeros((100, 50))
    for i in range(100):
        predicted_data[i] = observed_data + np.random.normal(0, 0.5, 50)
    
    return {
        "observed_data": observed_data,
        "predicted_data": predicted_data,
        "model_name": "Test Model",
    }


@pytest.fixture
def diagnostic_metrics_data():
    """Create sample diagnostic metrics data for testing."""
    return {
        "model1": {
            "WAIC": 100.0,
            "LOO": 95.0,
            "MSE": 0.05,
            "MAE": 0.02,
            "Coverage": 0.95,
            "R2": 0.85,
        },
        "model2": {
            "WAIC": 110.0,
            "LOO": 105.0,
            "MSE": 0.07,
            "MAE": 0.03,
            "Coverage": 0.92,
            "R2": 0.80,
        },
        "model3": {
            "WAIC": 90.0,
            "LOO": 85.0,
            "MSE": 0.04,
            "MAE": 0.015,
            "Coverage": 0.97,
            "R2": 0.88,
        },
    }


def test_plot_model_comparison(comparison_result, tmp_path):
    """Test plot_model_comparison function."""
    output_path = tmp_path / "model_comparison.png"
    
    # Test with default parameters
    fig = plot_model_comparison(comparison_result)
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path
    fig = plot_model_comparison(
        comparison_result,
        title="Test Comparison",
        figsize=(8, 5),
        save_path=output_path,
        show_differences=True,
        highlight_best=True,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Test without highlighting best model
    fig = plot_model_comparison(
        comparison_result,
        highlight_best=False,
    )
    assert isinstance(fig, plt.Figure)
    
    # Test without showing differences
    fig = plot_model_comparison(
        comparison_result,
        show_differences=False,
    )
    assert isinstance(fig, plt.Figure)
    
    plt.close("all")


def test_plot_model_comparison_grid(comparison_results_list, tmp_path):
    """Test plot_model_comparison_grid function."""
    output_path = tmp_path / "model_comparison_grid.png"
    
    # Test with default parameters
    fig = plot_model_comparison_grid(comparison_results_list)
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path and titles
    titles = ["WAIC Comparison", "LOO Comparison", "Bayes Factor Comparison"]
    fig = plot_model_comparison_grid(
        comparison_results_list,
        titles=titles,
        figsize=(15, 8),
        save_path=output_path,
        highlight_best=True,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Test with a single result
    fig = plot_model_comparison_grid([comparison_results_list[0]])
    assert isinstance(fig, plt.Figure)
    
    plt.close("all")


def test_plot_pointwise_comparison(pointwise_data, tmp_path):
    """Test plot_pointwise_comparison function."""
    output_path = tmp_path / "pointwise_comparison.png"
    
    # Test with default parameters
    fig = plot_pointwise_comparison(
        model1_name=pointwise_data["model1_name"],
        model2_name=pointwise_data["model2_name"],
        model1_pointwise=pointwise_data["model1_pointwise"],
        model2_pointwise=pointwise_data["model2_pointwise"],
    )
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path
    fig = plot_pointwise_comparison(
        model1_name=pointwise_data["model1_name"],
        model2_name=pointwise_data["model2_name"],
        model1_pointwise=pointwise_data["model1_pointwise"],
        model2_pointwise=pointwise_data["model2_pointwise"],
        metric_name="LOO",
        figsize=(8, 8),
        save_path=output_path,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    plt.close("all")


def test_plot_posterior_predictive_check(posterior_predictive_data, tmp_path):
    """Test plot_posterior_predictive_check function."""
    output_path = tmp_path / "posterior_predictive_check.png"
    
    # Test with default parameters
    fig = plot_posterior_predictive_check(
        observed_data=posterior_predictive_data["observed_data"],
        predicted_data=posterior_predictive_data["predicted_data"],
        model_name=posterior_predictive_data["model_name"],
    )
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path
    fig = plot_posterior_predictive_check(
        observed_data=posterior_predictive_data["observed_data"],
        predicted_data=posterior_predictive_data["predicted_data"],
        model_name=posterior_predictive_data["model_name"],
        figsize=(10, 8),
        save_path=output_path,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    plt.close("all")


def test_plot_diagnostic_metrics(diagnostic_metrics_data, tmp_path):
    """Test plot_diagnostic_metrics function."""
    output_path = tmp_path / "diagnostic_metrics.png"
    
    # Test with default parameters
    fig = plot_diagnostic_metrics(diagnostic_metrics_data)
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path
    fig = plot_diagnostic_metrics(
        models=diagnostic_metrics_data,
        figsize=(10, 8),
        save_path=output_path,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    plt.close("all")


@patch.object(BayesianModelComparison, "compare_models")
def test_plot_waic_loo_comparison(mock_compare_models, comparison_results_list, tmp_path):
    """Test plot_waic_loo_comparison function."""
    output_path = tmp_path / "waic_loo_comparison.png"
    
    # Mock the compare_models method to return the comparison results
    mock_compare_models.side_effect = [
        comparison_results_list[0],  # WAIC result
        comparison_results_list[1],  # LOO result
    ]
    
    # Create mock objects
    comparison_instance = BayesianModelComparison()
    models = {"model1": MagicMock(), "model2": MagicMock(), "model3": MagicMock()}
    posterior_samples = {"model1": {}, "model2": {}, "model3": {}}
    data = {}
    
    # Test with default parameters
    fig = plot_waic_loo_comparison(
        comparison_instance=comparison_instance,
        models=models,
        posterior_samples=posterior_samples,
        data=data,
    )
    assert isinstance(fig, plt.Figure)
    
    # Test with save_path
    fig = plot_waic_loo_comparison(
        comparison_instance=comparison_instance,
        models=models,
        posterior_samples=posterior_samples,
        data=data,
        figsize=(12, 6),
        save_path=output_path,
    )
    
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Verify that compare_models was called twice (once for WAIC, once for LOO)
    assert mock_compare_models.call_count == 2
    
    plt.close("all")


def test_import_visualization_module():
    """Test that the visualization module can be imported."""
    import pyrovelocity.plots.visualization
    
    # Check that the module has the expected functions
    assert hasattr(pyrovelocity.plots.visualization, "plot_model_comparison")
    assert hasattr(pyrovelocity.plots.visualization, "plot_model_comparison_grid")
    assert hasattr(pyrovelocity.plots.visualization, "plot_pointwise_comparison")
    assert hasattr(pyrovelocity.plots.visualization, "plot_posterior_predictive_check")
    assert hasattr(pyrovelocity.plots.visualization, "plot_diagnostic_metrics")
    assert hasattr(pyrovelocity.plots.visualization, "plot_waic_loo_comparison")