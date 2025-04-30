"""
End-to-end integration tests for the PyroVelocity modular implementation.

This module tests the entire pipeline from data loading to model training,
inference, and velocity computation.
"""

from importlib.resources import files

import anndata as ad
import numpy as np
import pyro
import pytest
import scanpy as sc
import torch

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.models.modular.factory import (
    create_model,
    create_standard_model,
    standard_model_config,
)
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import register_standard_components

# Fixture hash for data validation
FIXTURE_HASH = (
    "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"
)


@pytest.fixture(scope="module", autouse=True)
def setup_registries():
    """Register standard components in registries before running tests."""
    register_standard_components()
    return


@pytest.fixture
def test_data():
    """Load test data from the fixtures."""
    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    return load_anndata_from_json(
        filename=str(fixture_file_path),
        expected_hash=FIXTURE_HASH,
    )


@pytest.fixture
def setup_data(test_data):
    """Set up the test data for velocity inference."""
    # Set up AnnData using the direct method
    return PyroVelocityModel.setup_anndata(test_data)


class TestEndToEndPipeline:
    """Tests for the end-to-end pipeline."""

    def test_standard_model_training(self, setup_data):
        """Test training a standard model."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create the model
        model = create_standard_model()

        # Train the model (with reduced epochs for testing)
        model.train(
            adata=setup_data,
            max_epochs=10,
            learning_rate=0.01,
            use_gpu=False,
        )

        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(
            adata=setup_data,
            num_samples=30
        )

        # Verify that samples contain alpha, beta, gamma
        assert "alpha" in posterior_samples
        assert "beta" in posterior_samples
        assert "gamma" in posterior_samples

        # Verify sample shapes
        gene_count = setup_data.shape[1]
        assert posterior_samples["alpha"].shape[1] == gene_count
        assert posterior_samples["beta"].shape[1] == gene_count
        assert posterior_samples["gamma"].shape[1] == gene_count

        # Store results in AnnData
        adata_out = model.store_results_in_anndata(
            adata=setup_data,
            posterior_samples=posterior_samples
        )

        # Verify that adata has been updated with velocity information
        assert "pyrovelocity_velocity" in adata_out.layers
        assert adata_out.layers["pyrovelocity_velocity"].shape == (
            setup_data.shape[0],
            setup_data.shape[1],
        )

        # Verify that latent time has been computed
        assert "pyrovelocity_latent_time" in adata_out.obs
        assert adata_out.obs["pyrovelocity_latent_time"].shape[0] == setup_data.shape[0]

    def test_custom_model_training(self, setup_data):
        """Test training a custom model."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create a custom model configuration
        config = standard_model_config()

        # Modify the configuration to use different guide
        config.inference_guide.name = "auto"
        config.inference_guide.params = {"guide_type": "AutoDelta"}

        # Create the model
        model = create_model(config)

        # Train the model (with reduced epochs for testing)
        model.train(
            adata=setup_data,
            max_epochs=10,
            learning_rate=0.01,
            use_gpu=False,
        )

        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(
            adata=setup_data,
            num_samples=30
        )

        # Verify that samples contain alpha, beta, gamma
        assert "alpha" in posterior_samples
        assert "beta" in posterior_samples
        assert "gamma" in posterior_samples

        # Store results in AnnData
        adata_out = model.store_results_in_anndata(
            adata=setup_data,
            posterior_samples=posterior_samples
        )

        # Verify that adata has been updated with velocity information
        assert "pyrovelocity_velocity" in adata_out.layers
        assert adata_out.layers["pyrovelocity_velocity"].shape == (
            setup_data.shape[0],
            setup_data.shape[1],
        )

    def test_model_comparison(self, setup_data):
        """Test model comparison between different model configurations."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create model configurations
        model_configs = {
            "standard_poisson": standard_model_config(),
            "autodelta_guide": standard_model_config(),
        }

        # Modify the configurations
        model_configs["autodelta_guide"].inference_guide.name = "auto"
        model_configs["autodelta_guide"].inference_guide.params = {
            "guide_type": "AutoDelta"
        }

        # Create models
        models = {
            name: create_model(config) for name, config in model_configs.items()
        }

        # Train models and store results
        adatas = {}
        for name, model in models.items():
            # Train the model
            model.train(
                adata=setup_data,
                max_epochs=10,
                learning_rate=0.01,
                use_gpu=False,
            )

            # Generate posterior samples
            posterior_samples = model.generate_posterior_samples(
                adata=setup_data,
                num_samples=30
            )

            # Store results in AnnData
            adatas[name] = model.store_results_in_anndata(
                adata=setup_data.copy(),
                posterior_samples=posterior_samples
            )

        # Compare models
        # For this test, just verify that all models trained successfully
        for name, adata_out in adatas.items():
            # Verify posterior samples were stored
            assert "pyrovelocity_alpha" in adata_out.uns
            assert "pyrovelocity_beta" in adata_out.uns
            assert "pyrovelocity_gamma" in adata_out.uns

            # Verify velocity
            assert "pyrovelocity_velocity" in adata_out.layers
            assert adata_out.layers["pyrovelocity_velocity"].shape == (
                setup_data.shape[0],
                setup_data.shape[1],
            )

    def test_adata_compatibility(self, test_data):
        """Test compatibility with the AnnData interface."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Set up AnnData for velocity
        adata = test_data.copy()
        adata = PyroVelocityModel.setup_anndata(adata)

        # Create a model
        model = create_standard_model()

        # Train with minimal epochs
        model.train(
            adata=adata,
            max_epochs=2,
            learning_rate=0.01,
            use_gpu=False,
        )

        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(
            adata=adata,
            num_samples=30
        )

        # Store results in AnnData
        adata_out = model.store_results_in_anndata(
            adata=adata,
            posterior_samples=posterior_samples
        )

        # Verify that standard AnnData attributes are preserved
        assert adata_out.n_obs == adata.n_obs
        assert adata_out.n_vars == adata.n_vars

        # Verify that standard layers are present
        assert "spliced" in adata_out.layers
        assert "unspliced" in adata_out.layers

        # Verify that velocity-specific layers are added
        assert "pyrovelocity_velocity" in adata_out.layers

        # Verify that velocity-specific observations are added
        assert "pyrovelocity_latent_time" in adata_out.obs
