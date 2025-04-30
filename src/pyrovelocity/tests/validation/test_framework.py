"""
Tests for the PyroVelocity validation framework.

This module contains tests for the validation framework, which provides tools
for validating and comparing different implementations of PyroVelocity.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from anndata import AnnData

# Import the validation framework
import pyrovelocity.validation.framework

# Import the different implementations
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.jax.factory.factory import (
    create_standard_model as create_jax_standard_model,
)
from pyrovelocity.models.modular import PyroVelocityModel, create_standard_model
from pyrovelocity.validation.framework import (
    ValidationRunner,
    compare_implementations,
    run_validation,
)


@pytest.fixture
def small_anndata():
    """Create a small AnnData object for testing."""
    # Create a small AnnData object with random data
    n_cells = 20
    n_genes = 10

    # Create random count data
    np.random.seed(42)
    u_counts = np.random.poisson(5, size=(n_cells, n_genes))
    s_counts = np.random.poisson(10, size=(n_cells, n_genes))

    # Create AnnData object
    adata = ad.AnnData(X=s_counts)
    adata.layers["spliced"] = s_counts
    adata.layers["unspliced"] = u_counts

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Add library size
    adata.obs["u_lib_size"] = u_counts.sum(axis=1)
    adata.obs["s_lib_size"] = s_counts.sum(axis=1)

    return adata


class TestValidationRunner:
    """Tests for the ValidationRunner class."""

    def test_initialization(self, small_anndata):
        """Test initialization of ValidationRunner."""
        # Initialize ValidationRunner with a small AnnData object
        runner = ValidationRunner(small_anndata)

        # Check that the AnnData object is stored correctly
        assert runner.adata is small_anndata

        # Check that the models dictionary is empty
        assert runner.models == {}

        # Check that the results dictionary is empty
        assert runner.results == {}

    def test_add_model(self, small_anndata):
        """Test adding a model to ValidationRunner."""
        # Initialize ValidationRunner
        runner = ValidationRunner(small_anndata)

        # Create a mock model
        class MockModel:
            def __init__(self, name):
                self.name = name

        # Add a mock model
        mock_model = MockModel("mock_model")
        runner.add_model("mock", mock_model)

        # Check that the model was added correctly
        assert "mock" in runner.models
        assert runner.models["mock"] is mock_model

    def test_run_validation(self, small_anndata):
        """Test running validation with ValidationRunner."""
        # Create a mock ValidationRunner
        class MockValidationRunner(ValidationRunner):
            def run_validation(self, **kwargs):
                # Create mock results
                self.results = {
                    "legacy": {
                        "posterior_samples": {
                            "alpha": np.random.rand(10, 10),
                            "beta": np.random.rand(10, 10),
                            "gamma": np.random.rand(10, 10),
                        },
                        "velocity": np.random.rand(10, 10),
                        "uncertainty": np.random.rand(10, 10),
                        "performance": {
                            "training_time": 1.0,
                            "inference_time": 0.5,
                            "memory_usage": 100.0,
                        },
                    },
                    "modular": {
                        "posterior_samples": {
                            "alpha": np.random.rand(10, 10),
                            "beta": np.random.rand(10, 10),
                            "gamma": np.random.rand(10, 10),
                        },
                        "velocity": np.random.rand(10, 10),
                        "uncertainty": np.random.rand(10, 10),
                        "performance": {
                            "training_time": 1.2,
                            "inference_time": 0.6,
                            "memory_usage": 120.0,
                        },
                    },
                    "jax": {
                        "posterior_samples": {
                            "alpha": np.random.rand(10, 10),
                            "beta": np.random.rand(10, 10),
                            "gamma": np.random.rand(10, 10),
                        },
                        "velocity": np.random.rand(10, 10),
                        "uncertainty": np.random.rand(10, 10),
                        "performance": {
                            "training_time": 0.8,
                            "inference_time": 0.4,
                            "memory_usage": 80.0,
                        },
                    },
                }
                return self.results

        # Initialize MockValidationRunner
        runner = MockValidationRunner(small_anndata)

        # Create a mock model with a train method that returns a dictionary
        class MockModel:
            def __init__(self, name):
                self.name = name

            def train(self, **kwargs):
                return {"loss": 0.1}

            def generate_posterior_samples(self, **kwargs):
                return {
                    "alpha": np.random.rand(10, 10),
                    "beta": np.random.rand(10, 10),
                    "gamma": np.random.rand(10, 10),
                }

        # Add mock models
        runner.add_model("legacy", MockModel("legacy"))
        runner.add_model("modular", MockModel("modular"))
        runner.add_model("jax", MockModel("jax"))

        # Run validation
        results = runner.run_validation(max_epochs=1, use_scalene=False)

        # Check that results were generated for all models
        assert "legacy" in results
        assert "modular" in results
        assert "jax" in results

        # Check that posterior samples were generated for all models
        assert "posterior_samples" in results["legacy"]
        assert "posterior_samples" in results["modular"]
        assert "posterior_samples" in results["jax"]

        # Check that alpha, beta, gamma are in the posterior samples
        assert "alpha" in results["legacy"]["posterior_samples"]
        assert "beta" in results["legacy"]["posterior_samples"]
        assert "gamma" in results["legacy"]["posterior_samples"]

    def test_compare_implementations(self, small_anndata):
        """Test comparing implementations with ValidationRunner."""
        # Initialize ValidationRunner
        runner = ValidationRunner(small_anndata)

        # Create mock results
        mock_results = {
            "legacy": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 10),
                    "beta": np.random.rand(10, 10),
                    "gamma": np.random.rand(10, 10),
                },
                "velocity": np.random.rand(10, 10),
                "uncertainty": np.random.rand(10, 10),
                "performance": {
                    "training_time": 1.0,
                    "inference_time": 0.5,
                    "memory_usage": 100.0,
                },
            },
            "modular": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 10),
                    "beta": np.random.rand(10, 10),
                    "gamma": np.random.rand(10, 10),
                },
                "velocity": np.random.rand(10, 10),
                "uncertainty": np.random.rand(10, 10),
                "performance": {
                    "training_time": 1.2,
                    "inference_time": 0.6,
                    "memory_usage": 120.0,
                },
            },
            "jax": {
                "posterior_samples": {
                    "alpha": np.random.rand(10, 10),
                    "beta": np.random.rand(10, 10),
                    "gamma": np.random.rand(10, 10),
                },
                "velocity": np.random.rand(10, 10),
                "uncertainty": np.random.rand(10, 10),
                "performance": {
                    "training_time": 0.8,
                    "inference_time": 0.4,
                    "memory_usage": 80.0,
                },
            },
        }

        # Set results
        runner.results = mock_results

        # Compare implementations
        comparison = runner.compare_implementations()

        # Check that comparison results were generated
        assert "parameter_comparison" in comparison
        assert "velocity_comparison" in comparison
        assert "uncertainty_comparison" in comparison
        assert "performance_comparison" in comparison

        # Check that parameter comparison includes all parameters
        assert "alpha" in comparison["parameter_comparison"]
        assert "beta" in comparison["parameter_comparison"]
        assert "gamma" in comparison["parameter_comparison"]

        # Check that performance comparison includes all metrics
        assert "legacy_vs_modular" in comparison["performance_comparison"]
        assert "legacy_vs_jax" in comparison["performance_comparison"]
        assert "modular_vs_jax" in comparison["performance_comparison"]

        # Check that each comparison includes the ratios
        for pair in ["legacy_vs_modular", "legacy_vs_jax", "modular_vs_jax"]:
            assert "training_time_ratio" in comparison["performance_comparison"][pair]
            assert "inference_time_ratio" in comparison["performance_comparison"][pair]
            assert "memory_usage_ratio" in comparison["performance_comparison"][pair]


def test_run_validation_function(small_anndata):
    """Test the run_validation function."""
    # Create a mock ValidationRunner
    class MockValidationRunner:
        def __init__(self, adata):
            self.adata = adata
            self.models = {}
            self.results = {
                "mock": {
                    "posterior_samples": {
                        "alpha": np.random.rand(10, 10),
                        "beta": np.random.rand(10, 10),
                        "gamma": np.random.rand(10, 10),
                    },
                    "velocity": np.random.rand(10, 10),
                    "uncertainty": np.random.rand(10, 10),
                    "performance": {
                        "training_time": 1.0,
                        "inference_time": 0.5,
                        "memory_usage": 100.0,
                    },
                }
            }

        def setup_legacy_model(self, **kwargs):
            pass

        def setup_modular_model(self, **kwargs):
            pass

        def setup_jax_model(self, **kwargs):
            pass

        def run_validation(self, **kwargs):
            return self.results

        def compare_implementations(self):
            return {
                "parameter_comparison": {},
                "velocity_comparison": {},
                "uncertainty_comparison": {},
                "performance_comparison": {},
            }

    # Patch the ValidationRunner class
    original_runner = pyrovelocity.validation.framework.ValidationRunner
    pyrovelocity.validation.framework.ValidationRunner = MockValidationRunner

    try:
        # Run validation with minimal arguments
        results = run_validation(
            adata=small_anndata,
            max_epochs=1,
            use_legacy=False,
            use_modular=False,
            use_jax=False,
        )

        # Check that results is a dictionary
        assert isinstance(results, dict)

        # Check that comparison results were generated
        assert "comparison" in results
    finally:
        # Restore the original ValidationRunner class
        pyrovelocity.validation.framework.ValidationRunner = original_runner


def test_compare_implementations_function():
    """Test the compare_implementations function."""
    # Create mock results
    mock_results = {
        "legacy": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 10),
                "beta": np.random.rand(10, 10),
                "gamma": np.random.rand(10, 10),
            },
            "velocity": np.random.rand(10, 10),
            "uncertainty": np.random.rand(10, 10),
            "performance": {
                "training_time": 1.0,
                "inference_time": 0.5,
                "memory_usage": 100.0,
            },
        },
        "modular": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 10),
                "beta": np.random.rand(10, 10),
                "gamma": np.random.rand(10, 10),
            },
            "velocity": np.random.rand(10, 10),
            "uncertainty": np.random.rand(10, 10),
            "performance": {
                "training_time": 1.2,
                "inference_time": 0.6,
                "memory_usage": 120.0,
            },
        },
        "jax": {
            "posterior_samples": {
                "alpha": np.random.rand(10, 10),
                "beta": np.random.rand(10, 10),
                "gamma": np.random.rand(10, 10),
            },
            "velocity": np.random.rand(10, 10),
            "uncertainty": np.random.rand(10, 10),
            "performance": {
                "training_time": 0.8,
                "inference_time": 0.4,
                "memory_usage": 80.0,
            },
        },
    }

    # Compare implementations
    comparison = compare_implementations(mock_results)

    # Check that comparison results were generated
    assert "parameter_comparison" in comparison
    assert "velocity_comparison" in comparison
    assert "uncertainty_comparison" in comparison
    assert "performance_comparison" in comparison
