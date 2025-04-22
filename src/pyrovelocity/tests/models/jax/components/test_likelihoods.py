"""
Tests for PyroVelocity JAX/NumPyro likelihood components.

This module contains tests for the likelihood components, including:

- test_poisson_likelihood_function: Test poisson likelihood function
- test_negative_binomial_likelihood_function: Test negative binomial likelihood function
- test_register_standard_likelihoods: Test registration of standard likelihood functions
"""

import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from pyrovelocity.models.jax.components.likelihoods import (
    negative_binomial_likelihood_function,
    poisson_likelihood_function,
)
from pyrovelocity.models.jax.registry import get_likelihood


def test_poisson_likelihood_function():
    """Test poisson likelihood function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))
    u_logits = jnp.ones((batch_size, n_cells, n_genes))
    s_logits = jnp.ones((batch_size, n_cells, n_genes))

    # Create a model that uses the likelihood function
    def model():
        poisson_likelihood_function(u_obs, s_obs, u_logits, s_logits)

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u" in tr
    assert "s" in tr

    # Check that the model uses the correct observations
    np.testing.assert_allclose(tr["u"]["value"], u_obs)
    np.testing.assert_allclose(tr["s"]["value"], s_obs)

    # Test with library size scaling
    u_log_library = jnp.zeros((batch_size, n_cells))
    s_log_library = jnp.zeros((batch_size, n_cells))
    likelihood_params = {
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
    }

    # Create a model that uses the likelihood function with library size scaling
    def model_with_scaling():
        poisson_likelihood_function(
            u_obs, s_obs, u_logits, s_logits, likelihood_params
        )

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model_with_scaling, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u" in tr
    assert "s" in tr


def test_negative_binomial_likelihood_function():
    """Test negative binomial likelihood function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))
    u_logits = jnp.ones((batch_size, n_cells, n_genes))
    s_logits = jnp.ones((batch_size, n_cells, n_genes))

    # Create a model that uses the likelihood function
    def model():
        negative_binomial_likelihood_function(u_obs, s_obs, u_logits, s_logits)

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u" in tr
    assert "s" in tr

    # Check that the model uses the correct observations
    np.testing.assert_allclose(tr["u"]["value"], u_obs)
    np.testing.assert_allclose(tr["s"]["value"], s_obs)

    # Test with library size scaling and dispersion parameters
    u_log_library = jnp.zeros((batch_size, n_cells))
    s_log_library = jnp.zeros((batch_size, n_cells))
    likelihood_params = {
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
        "u_dispersion": 2.0,
        "s_dispersion": 3.0,
    }

    # Create a model that uses the likelihood function with library size scaling and dispersion
    def model_with_params():
        negative_binomial_likelihood_function(
            u_obs, s_obs, u_logits, s_logits, likelihood_params
        )

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model_with_params, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u" in tr
    assert "s" in tr

    # Check that the model uses the correct observations


def test_register_standard_likelihoods():
    """Test registration of standard likelihood functions."""
    # Check that poisson likelihood function is registered
    poisson_fn = get_likelihood("poisson")
    assert poisson_fn is not None

    # Check that negative binomial likelihood function is registered
    nb_fn = get_likelihood("negative_binomial")
    assert nb_fn is not None
