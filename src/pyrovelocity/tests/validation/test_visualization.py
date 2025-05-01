"""
Tests for the PyroVelocity visualization tools.

This module contains tests for the visualization tools, which provide tools
for visualizing the comparison between different implementations of PyroVelocity.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Import the visualization tools
from pyrovelocity.validation.visualization import (
    plot_parameter_comparison,
    plot_parameter_distributions,
    plot_performance_comparison,
    plot_performance_radar,
    plot_uncertainty_comparison,
    plot_uncertainty_heatmap,
    plot_velocity_comparison,
    plot_velocity_vector_field,
)

# Set matplotlib to use a non-interactive backend for testing
matplotlib.use('Agg')


@pytest.fixture
def comparison_results():
    """Create mock comparison results for testing."""
    # Create mock parameter comparison results
    parameter_comparison = {
        "alpha": {
            "legacy_vs_modular": {
                "mse": 0.01,
                "correlation": 0.95,
                "kl_divergence": 0.05,
                "wasserstein_distance": 0.02,
            },
            "legacy_vs_jax": {
                "mse": 0.02,
                "correlation": 0.90,
                "kl_divergence": 0.10,
                "wasserstein_distance": 0.03,
            },
            "modular_vs_jax": {
                "mse": 0.015,
                "correlation": 0.92,
                "kl_divergence": 0.08,
                "wasserstein_distance": 0.025,
            },
        },
        "beta": {
            "legacy_vs_modular": {
                "mse": 0.015,
                "correlation": 0.93,
                "kl_divergence": 0.06,
                "wasserstein_distance": 0.025,
            },
            "legacy_vs_jax": {
                "mse": 0.025,
                "correlation": 0.88,
                "kl_divergence": 0.12,
                "wasserstein_distance": 0.035,
            },
            "modular_vs_jax": {
                "mse": 0.02,
                "correlation": 0.90,
                "kl_divergence": 0.10,
                "wasserstein_distance": 0.03,
            },
        },
        "gamma": {
            "legacy_vs_modular": {
                "mse": 0.02,
                "correlation": 0.90,
                "kl_divergence": 0.08,
                "wasserstein_distance": 0.03,
            },
            "legacy_vs_jax": {
                "mse": 0.03,
                "correlation": 0.85,
                "kl_divergence": 0.15,
                "wasserstein_distance": 0.04,
            },
            "modular_vs_jax": {
                "mse": 0.025,
                "correlation": 0.87,
                "kl_divergence": 0.12,
                "wasserstein_distance": 0.035,
            },
        },
    }

    # Create mock velocity comparison results
    velocity_comparison = {
        "legacy_vs_modular": {
            "mse": 0.02,
            "correlation": 0.90,
            "cosine_similarity": 0.95,
            "magnitude_similarity": 0.92,
        },
        "legacy_vs_jax": {
            "mse": 0.03,
            "correlation": 0.85,
            "cosine_similarity": 0.90,
            "magnitude_similarity": 0.87,
        },
        "modular_vs_jax": {
            "mse": 0.025,
            "correlation": 0.87,
            "cosine_similarity": 0.92,
            "magnitude_similarity": 0.89,
        },
    }

    # Create mock uncertainty comparison results
    uncertainty_comparison = {
        "legacy_vs_modular": {
            "mse": 0.03,
            "correlation": 0.85,
            "distribution_similarity": 0.90,
        },
        "legacy_vs_jax": {
            "mse": 0.04,
            "correlation": 0.80,
            "distribution_similarity": 0.85,
        },
        "modular_vs_jax": {
            "mse": 0.035,
            "correlation": 0.82,
            "distribution_similarity": 0.87,
        },
    }

    # Create mock performance comparison results
    performance_comparison = {
        "legacy_vs_modular": {
            "training_time_ratio": 1.2,
            "inference_time_ratio": 1.1,
            "memory_usage_ratio": 1.3,
        },
        "legacy_vs_jax": {
            "training_time_ratio": 0.8,
            "inference_time_ratio": 0.7,
            "memory_usage_ratio": 0.9,
        },
        "modular_vs_jax": {
            "training_time_ratio": 0.7,
            "inference_time_ratio": 0.6,
            "memory_usage_ratio": 0.7,
        },
    }

    return {
        "parameter_comparison": parameter_comparison,
        "velocity_comparison": velocity_comparison,
        "uncertainty_comparison": uncertainty_comparison,
        "performance_comparison": performance_comparison,
    }


@pytest.fixture
def implementation_results():
    """Create mock implementation results for testing."""
    # Create mock results for each implementation
    return {
        "legacy": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(10, 5),
            "uncertainty": np.random.rand(10, 5),
            "performance": {
                "training_time": 1.0,
                "inference_time": 0.5,
                "memory_usage": 100.0,
            },
        },
        "modular": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(10, 5),
            "uncertainty": np.random.rand(10, 5),
            "performance": {
                "training_time": 1.2,
                "inference_time": 0.6,
                "memory_usage": 120.0,
            },
        },
        "jax": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(10, 5),
            "uncertainty": np.random.rand(10, 5),
            "performance": {
                "training_time": 0.8,
                "inference_time": 0.4,
                "memory_usage": 80.0,
            },
        },
    }


@pytest.fixture
def implementation_results_different_shapes():
    """Create mock implementation results with different shapes for testing."""
    # Create mock results for each implementation with different shapes
    return {
        "modular": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(100, 5),  # 100 cells
            "uncertainty": np.random.rand(5),
            "performance": {
                "training_time": 1.2,
                "inference_time": 0.6,
                "memory_usage": 120.0,
            },
        },
        "jax": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(300, 5),  # 300 cells
            "uncertainty": np.random.rand(5),
            "performance": {
                "training_time": 0.8,
                "inference_time": 0.4,
                "memory_usage": 80.0,
            },
        },
    }


@pytest.fixture
def implementation_results_single():
    """Create mock implementation results with only one implementation for testing."""
    # Create mock results for a single implementation
    return {
        "modular": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 5),
                "beta": np.random.rand(10, 5),
                "gamma": np.random.rand(10, 5),
            },
            "velocity": np.random.rand(10, 5),
            "uncertainty": np.random.rand(10, 5),
            "performance": {
                "training_time": 1.2,
                "inference_time": 0.6,
                "memory_usage": 120.0,
            },
        }
    }


class TestParameterVisualization:
    """Tests for parameter visualization tools."""

    def test_plot_parameter_comparison(self, comparison_results):
        """Test plotting parameter comparison."""
        # Plot parameter comparison
        fig = plot_parameter_comparison(comparison_results["parameter_comparison"])

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_parameter_distributions(self, implementation_results):
        """Test plotting parameter distributions."""
        # Plot parameter distributions
        fig = plot_parameter_distributions(implementation_results, parameter="alpha")

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)


class TestVelocityVisualization:
    """Tests for velocity visualization tools."""

    def test_plot_velocity_comparison(self, comparison_results):
        """Test plotting velocity comparison."""
        # Plot velocity comparison
        fig = plot_velocity_comparison(comparison_results["velocity_comparison"])

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_velocity_vector_field(self, implementation_results):
        """Test plotting velocity vector field."""
        # Create mock coordinates
        coordinates = np.random.rand(10, 2)

        # Plot velocity vector field
        fig = plot_velocity_vector_field(implementation_results, coordinates)

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_velocity_vector_field_different_shapes(self, implementation_results_different_shapes):
        """Test plotting velocity vector field with different shapes."""
        # Create mock coordinates
        coordinates = np.random.rand(300, 2)  # Match the larger shape

        # Plot velocity vector field
        fig = plot_velocity_vector_field(implementation_results_different_shapes, coordinates)

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_velocity_vector_field_single_implementation(self, implementation_results_single):
        """Test plotting velocity vector field with a single implementation."""
        # Create mock coordinates
        coordinates = np.random.rand(10, 2)

        # Plot velocity vector field
        fig = plot_velocity_vector_field(implementation_results_single, coordinates)

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)


class TestUncertaintyVisualization:
    """Tests for uncertainty visualization tools."""

    def test_plot_uncertainty_comparison(self, comparison_results):
        """Test plotting uncertainty comparison."""
        # Plot uncertainty comparison
        fig = plot_uncertainty_comparison(comparison_results["uncertainty_comparison"])

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_uncertainty_heatmap(self, implementation_results):
        """Test plotting uncertainty heatmap."""
        # Plot uncertainty heatmap
        fig = plot_uncertainty_heatmap(implementation_results)

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)


class TestPerformanceVisualization:
    """Tests for performance visualization tools."""

    def test_plot_performance_comparison(self, comparison_results):
        """Test plotting performance comparison."""
        # Plot performance comparison
        fig = plot_performance_comparison(comparison_results["performance_comparison"])

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def test_plot_performance_radar(self, implementation_results):
        """Test plotting performance radar chart."""
        # Plot performance radar chart
        fig = plot_performance_radar(implementation_results)

        # Check that a figure was created
        assert isinstance(fig, plt.Figure)

        # Close the figure to avoid memory leaks
        plt.close(fig)
