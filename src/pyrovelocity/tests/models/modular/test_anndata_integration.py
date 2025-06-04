"""
Tests for direct AnnData integration in PyroVelocityModel.

This module contains tests for the direct AnnData integration methods in PyroVelocityModel,
including:

- setup_anndata: Set up AnnData for use with PyroVelocityModel
- train: Train the model using AnnData
- generate_posterior_samples: Generate posterior samples
- store_results_in_anndata: Store results in AnnData
"""

import os

import anndata
import numpy as np
import pytest
import torch
from anndata import AnnData

from pyrovelocity.models.modular.factory import create_legacy_model1
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    # Create a simple AnnData object
    n_cells = 10
    n_genes = 5
    X = np.random.randn(n_cells, n_genes)

    # Create spliced and unspliced layers
    spliced = np.abs(np.random.randn(n_cells, n_genes))
    unspliced = np.abs(np.random.randn(n_cells, n_genes))

    # Create AnnData object
    adata = AnnData(X=X)
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["raw_spliced"] = spliced
    adata.layers["raw_unspliced"] = unspliced

    # Add library size information
    adata.obs["u_lib_size_raw"] = unspliced.sum(axis=1)
    adata.obs["s_lib_size_raw"] = spliced.sum(axis=1)

    return adata


@pytest.fixture
def model():
    """Create a PyroVelocityModel for testing."""
    return create_legacy_model1()


def test_setup_anndata(sample_adata, model):
    """Test setup_anndata method."""
    # Call the method
    adata = PyroVelocityModel.setup_anndata(sample_adata)

    # Check that the required fields are added
    assert "u_lib_size" in adata.obs
    assert "s_lib_size" in adata.obs
    assert "u_lib_size_mean" in adata.obs
    assert "s_lib_size_mean" in adata.obs
    assert "u_lib_size_scale" in adata.obs
    assert "s_lib_size_scale" in adata.obs
    assert "ind_x" in adata.obs

    # Check that the library sizes are computed correctly
    np.testing.assert_allclose(
        np.exp(adata.obs["u_lib_size"]) - 1e-6, adata.obs["u_lib_size_raw"]
    )
    np.testing.assert_allclose(
        np.exp(adata.obs["s_lib_size"]) - 1e-6, adata.obs["s_lib_size_raw"]
    )


def test_generate_posterior_samples(sample_adata, model):
    """Test generate_posterior_samples method."""
    # Set up AnnData
    adata = PyroVelocityModel.setup_anndata(sample_adata)

    # Create mock inference state
    # Create mock posterior samples
    import torch

    from pyrovelocity.models.modular.inference.unified import InferenceState
    mock_posterior_samples = {
        "alpha": torch.randn(2, sample_adata.n_vars),
        "beta": torch.randn(2, sample_adata.n_vars),
        "gamma": torch.randn(2, sample_adata.n_vars),
    }

    # Create mock inference state
    mock_inference_state = InferenceState(
        method="svi",
        params={},
        posterior_samples=mock_posterior_samples,
    )

    # Create a new state with the mock inference state
    new_state = ModelState(
        dynamics_state=model.state.dynamics_state,
        prior_state=model.state.prior_state,
        likelihood_state=model.state.likelihood_state,
        guide_state=model.state.guide_state,
        metadata={"inference_state": mock_inference_state}
    )

    # Update the model with the new state
    model = model.with_state(new_state)

    # Mock the extract_posterior_samples function
    import pyrovelocity.models.modular.inference.unified
    original_extract_posterior_samples = pyrovelocity.models.modular.inference.unified.extract_posterior_samples

    def mock_extract_posterior_samples(state, num_samples=None, seed=None):
        return mock_posterior_samples

    pyrovelocity.models.modular.inference.unified.extract_posterior_samples = mock_extract_posterior_samples

    try:
        # Generate posterior samples
        posterior_samples = model.generate_posterior_samples(num_samples=2)

        # Check that the posterior samples are returned
        assert isinstance(posterior_samples, dict)
        assert "alpha" in posterior_samples
        assert "beta" in posterior_samples
        assert "gamma" in posterior_samples

        # Check that the posterior samples have the correct shape
        assert posterior_samples["alpha"].shape == (2, sample_adata.n_vars)
        assert posterior_samples["beta"].shape == (2, sample_adata.n_vars)
        assert posterior_samples["gamma"].shape == (2, sample_adata.n_vars)
    finally:
        # Restore the original function
        pyrovelocity.models.modular.inference.unified.extract_posterior_samples = original_extract_posterior_samples


def test_store_results_in_anndata(sample_adata, model, monkeypatch):
    """Test store_results_in_anndata method."""
    # Set up AnnData
    adata = PyroVelocityModel.setup_anndata(sample_adata)

    # Create mock posterior samples
    posterior_samples = {
        "alpha": np.random.randn(2, sample_adata.n_vars),
        "beta": np.random.randn(2, sample_adata.n_vars),
        "gamma": np.random.randn(2, sample_adata.n_vars),
    }

    # Mock compute_velocity to return a simple result
    def mock_compute_velocity(*args, **kwargs):
        return {
            "velocity": np.random.randn(sample_adata.n_obs, sample_adata.n_vars),
            "latent_time": np.random.rand(sample_adata.n_obs),
            "alpha": np.random.randn(sample_adata.n_vars),
            "beta": np.random.randn(sample_adata.n_vars),
            "gamma": np.random.randn(sample_adata.n_vars),
        }

    # Apply the mock
    import pyrovelocity.models.modular.inference.posterior
    monkeypatch.setattr(
        pyrovelocity.models.modular.inference.posterior,
        "compute_velocity",
        mock_compute_velocity,
    )

    # Store results in AnnData
    adata_out = model.store_results_in_anndata(adata, posterior_samples)

    # Check that the results are stored in the correct locations
    assert "pyrovelocity_alpha" in adata_out.var
    assert "pyrovelocity_beta" in adata_out.var
    assert "pyrovelocity_gamma" in adata_out.var
    assert "pyrovelocity_latent_time" in adata_out.obs
    assert "pyrovelocity_velocity" in adata_out.layers

    # We can't check exact values because the mock returns random values
    # Just check that the shapes are correct
    assert adata_out.var["pyrovelocity_alpha"].shape == (sample_adata.n_vars,)
    assert adata_out.var["pyrovelocity_beta"].shape == (sample_adata.n_vars,)
    assert adata_out.var["pyrovelocity_gamma"].shape == (sample_adata.n_vars,)
    assert adata_out.layers["pyrovelocity_velocity"].shape == (
        sample_adata.n_obs,
        sample_adata.n_vars,
    )
    assert adata_out.obs["pyrovelocity_latent_time"].shape == (sample_adata.n_obs,)
