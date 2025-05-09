"""Tests for likelihood models."""

import numpy as np
import pyro.distributions
import pytest
import torch
from anndata._core.anndata import AnnData

from pyrovelocity.models.modular.components.likelihoods import (
    LegacyLikelihoodModel,
    PoissonLikelihoodModel,
)
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_likelihood_models():
    """Register likelihood models for testing."""
    # Save original registry state
    original_registry = dict(LikelihoodModelRegistry._registry)

    # Clear registry and register test components
    LikelihoodModelRegistry.clear()
    LikelihoodModelRegistry._registry["poisson"] = PoissonLikelihoodModel
    LikelihoodModelRegistry._registry["legacy"] = LegacyLikelihoodModel

    yield

    # Restore original registry state
    LikelihoodModelRegistry._registry = original_registry


@pytest.fixture
def test_adata():
    """Create a test AnnData object."""
    # Create a simple AnnData object for testing
    n_cells = 10
    n_genes = 20
    X = np.random.poisson(lam=5, size=(n_cells, n_genes))
    adata = AnnData(X=X)
    # Add X to layers as well, which is required by the likelihood models
    adata.layers["X"] = X.copy()
    return adata


def test_poisson_likelihood_model(test_adata):
    """Test PoissonLikelihoodModel."""
    # Get model from registry
    model_class = LikelihoodModelRegistry.get("poisson")
    assert model_class == PoissonLikelihoodModel

    # Create model instance
    model = model_class()

    # Generate distributions
    batch_size = 8
    latent_dim = 5
    cell_state = torch.zeros((batch_size, latent_dim))
    dists = model(test_adata, cell_state=cell_state, batch_size=batch_size)

    # Check output
    assert isinstance(dists, dict)
    assert "obs_counts" in dists
    assert isinstance(dists["obs_counts"], pyro.distributions.Poisson)

    # Check distribution parameters
    assert dists["obs_counts"].batch_shape == (batch_size, test_adata.n_vars)

    # Check rate parameter
    rate = dists["obs_counts"].rate
    assert rate.shape == (batch_size, test_adata.n_vars)
    assert torch.all(rate > 0)  # Rate should be positive


def test_legacy_likelihood_model(test_adata):
    """Test LegacyLikelihoodModel."""
    # Get model from registry
    model_class = LikelihoodModelRegistry.get("legacy")
    assert model_class == LegacyLikelihoodModel

    # Create model instance
    model = model_class()

    # Generate distributions
    batch_size = 8
    latent_dim = 5
    cell_state = torch.zeros((batch_size, latent_dim))
    dists = model(test_adata, cell_state=cell_state, batch_size=batch_size)

    # Check output
    assert isinstance(dists, dict)
    assert "obs_counts" in dists
    assert isinstance(dists["obs_counts"], pyro.distributions.Poisson)

    # Check distribution parameters
    assert dists["obs_counts"].batch_shape == (batch_size, test_adata.n_vars)

    # Check rate parameter
    rate = dists["obs_counts"].rate
    assert rate.shape == (batch_size, test_adata.n_vars)
    assert torch.all(rate > 0)  # Rate should be positive


def test_likelihood_model_with_gene_offset(test_adata):
    """Test likelihood models with gene offset."""
    # Get models from registry
    poisson_model = LikelihoodModelRegistry.get("poisson")()
    legacy_model = LikelihoodModelRegistry.get("legacy")()

    # Generate test data
    batch_size = 8
    latent_dim = 5
    cell_state = torch.zeros((batch_size, latent_dim))
    gene_offset = (
        torch.ones((batch_size, test_adata.n_vars)) * 2.0
    )  # Double the offset

    # Generate distributions
    poisson_dists = poisson_model(
        test_adata, cell_state=cell_state, gene_offset=gene_offset, batch_size=batch_size
    )
    legacy_dists = legacy_model(
        test_adata, cell_state=cell_state, gene_offset=gene_offset, batch_size=batch_size
    )

    # Check that offset affects the rate
    poisson_rate = poisson_dists["obs_counts"].rate
    legacy_rate = legacy_dists["obs_counts"].rate

    # Generate distributions without offset for comparison
    poisson_dists_no_offset = poisson_model(test_adata, cell_state=cell_state, batch_size=batch_size)
    legacy_dists_no_offset = legacy_model(test_adata, cell_state=cell_state, batch_size=batch_size)

    poisson_rate_no_offset = poisson_dists_no_offset["obs_counts"].rate
    legacy_rate_no_offset = legacy_dists_no_offset["obs_counts"].rate

    # Check that rates with offset are approximately twice the rates without offset
    # (allowing for numerical precision issues)
    assert torch.allclose(poisson_rate, poisson_rate_no_offset * 2.0, rtol=1e-5)
    assert torch.allclose(legacy_rate, legacy_rate_no_offset * 2.0, rtol=1e-5)


def test_likelihood_registry():
    """Test likelihood model registry."""
    # Check that models are registered
    assert "poisson" in LikelihoodModelRegistry.list_available()
    assert "legacy" in LikelihoodModelRegistry.list_available()

    # Check retrieved models
    poisson_class = LikelihoodModelRegistry.get("poisson")
    assert poisson_class == PoissonLikelihoodModel

    legacy_class = LikelihoodModelRegistry.get("legacy")
    assert legacy_class == LegacyLikelihoodModel

    # Create instances using the registry
    poisson_model = LikelihoodModelRegistry.create("poisson")
    legacy_model = LikelihoodModelRegistry.create("legacy")

    assert isinstance(poisson_model, PoissonLikelihoodModel)
    assert isinstance(legacy_model, LegacyLikelihoodModel)
