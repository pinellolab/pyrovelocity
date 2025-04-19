"""Tests for likelihood models."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jaxtyping import Array, Float
import numpy as np
from anndata._core.anndata import AnnData

from pyrovelocity.models.modular.components.likelihoods import (
    NegativeBinomialLikelihoodModel,
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
    LikelihoodModelRegistry._registry["negative_binomial"] = NegativeBinomialLikelihoodModel
    
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
    return AnnData(X=X)


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
    cell_state = jnp.zeros((batch_size, latent_dim))
    dists = model(test_adata, cell_state=cell_state)

    # Check output
    assert isinstance(dists, dict)
    assert "obs_counts" in dists
    assert isinstance(dists["obs_counts"], dist.Poisson)

    # Check distribution parameters
    assert dists["obs_counts"].batch_shape == (batch_size, test_adata.n_vars)

    # Check rate parameter
    rate = dists["obs_counts"].rate
    assert rate.shape == (batch_size, test_adata.n_vars)
    assert jnp.all(rate > 0)  # Rate should be positive


def test_negative_binomial_likelihood_model(test_adata):
    """Test NegativeBinomialLikelihoodModel."""
    # Get model from registry
    model_class = LikelihoodModelRegistry.get("negative_binomial")
    assert model_class == NegativeBinomialLikelihoodModel

    # Create model instance
    model = model_class()

    # Generate distributions
    batch_size = 8
    latent_dim = 5
    cell_state = jnp.zeros((batch_size, latent_dim))
    dists = model(test_adata, cell_state=cell_state)

    # Check output
    assert isinstance(dists, dict)
    assert "obs_counts" in dists
    assert isinstance(dists["obs_counts"], dist.GammaPoisson)

    # Check distribution parameters
    assert dists["obs_counts"].batch_shape == (batch_size, test_adata.n_vars)

    # Check concentration and rate parameters
    concentration = dists["obs_counts"].concentration
    rate = dists["obs_counts"].rate
    # The concentration parameter might have shape (n_genes,) or (1, n_genes)
    # Just check that it contains the right number of elements
    assert test_adata.n_vars in concentration.shape
    assert rate.shape == (batch_size, test_adata.n_vars)
    assert jnp.all(concentration > 0)  # Concentration should be positive
    assert jnp.all(rate > 0)  # Rate should be positive


def test_likelihood_model_with_gene_offset(test_adata):
    """Test likelihood models with gene offset."""
    # Get models from registry
    poisson_model = LikelihoodModelRegistry.get("poisson")()
    nb_model = LikelihoodModelRegistry.get("negative_binomial")()

    # Generate test data
    batch_size = 8
    latent_dim = 5
    cell_state = jnp.zeros((batch_size, latent_dim))
    gene_offset = (
        jnp.ones((batch_size, test_adata.n_vars)) * 2.0
    )  # Double the offset

    # Generate distributions
    poisson_dists = poisson_model(
        test_adata, cell_state=cell_state, gene_offset=gene_offset
    )
    nb_dists = nb_model(
        test_adata, cell_state=cell_state, gene_offset=gene_offset
    )

    # Check that offset affects the rate
    poisson_rate = poisson_dists["obs_counts"].rate
    nb_rate_param = nb_dists["obs_counts"].rate

    # Generate distributions without offset for comparison
    poisson_dists_no_offset = poisson_model(test_adata, cell_state=cell_state)
    nb_dists_no_offset = nb_model(test_adata, cell_state=cell_state)

    poisson_rate_no_offset = poisson_dists_no_offset["obs_counts"].rate
    nb_rate_param_no_offset = nb_dists_no_offset["obs_counts"].rate

    # Check that rates with offset are approximately twice the rates without offset
    # (allowing for numerical precision issues)
    assert jnp.allclose(poisson_rate, poisson_rate_no_offset * 2.0, rtol=1e-5)
    assert jnp.allclose(nb_rate_param_no_offset, nb_rate_param * 2.0, rtol=1e-5)


def test_likelihood_registry():
    """Test likelihood model registry."""
    # Check that models are registered
    assert "poisson" in LikelihoodModelRegistry.list_available()
    assert "negative_binomial" in LikelihoodModelRegistry.list_available()

    # Check retrieved models
    poisson_class = LikelihoodModelRegistry.get("poisson")
    assert poisson_class == PoissonLikelihoodModel

    nb_class = LikelihoodModelRegistry.get("negative_binomial")
    assert nb_class == NegativeBinomialLikelihoodModel

    # Create instances using the registry
    poisson_model = LikelihoodModelRegistry.create("poisson")
    nb_model = LikelihoodModelRegistry.create("negative_binomial")

    assert isinstance(poisson_model, PoissonLikelihoodModel)
    assert isinstance(nb_model, NegativeBinomialLikelihoodModel)
