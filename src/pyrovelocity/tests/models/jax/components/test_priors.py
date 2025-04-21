"""
Tests for PyroVelocity JAX/NumPyro prior components.

This module contains tests for the prior components, including:

- test_lognormal_prior_function: Test lognormal prior function
- test_informative_prior_function: Test informative prior function
- test_register_standard_priors: Test registration of standard prior functions
"""

import jax
import jax.numpy as jnp

from pyrovelocity.models.jax.components.priors import (
    informative_prior_function,
    lognormal_prior_function,
)
from pyrovelocity.models.jax.registry import get_prior


def test_lognormal_prior_function():
    """Test lognormal prior function."""
    # Create test data
    key = jax.random.PRNGKey(0)
    num_genes = 10

    # Call function with default parameters
    params = lognormal_prior_function(key, num_genes)

    # Check that all required parameters are present
    assert "alpha" in params
    assert "beta" in params
    assert "gamma" in params

    # Check shapes
    assert params["alpha"].shape == (num_genes,)
    assert params["beta"].shape == (num_genes,)
    assert params["gamma"].shape == (num_genes,)

    # Check that values are positive (lognormal distribution)
    assert jnp.all(params["alpha"] > 0)
    assert jnp.all(params["beta"] > 0)
    assert jnp.all(params["gamma"] > 0)

    # Call function with custom parameters
    prior_params = {
        "alpha_loc": 0.0,
        "alpha_scale": 0.5,
        "beta_loc": -1.0,
        "beta_scale": 0.5,
        "gamma_loc": -1.5,
        "gamma_scale": 0.5,
    }
    params = lognormal_prior_function(key, num_genes, prior_params)

    # Check that all required parameters are present
    assert "alpha" in params
    assert "beta" in params
    assert "gamma" in params

    # Check shapes
    assert params["alpha"].shape == (num_genes,)
    assert params["beta"].shape == (num_genes,)
    assert params["gamma"].shape == (num_genes,)

    # Check that values are positive (lognormal distribution)
    assert jnp.all(params["alpha"] > 0)
    assert jnp.all(params["beta"] > 0)
    assert jnp.all(params["gamma"] > 0)


def test_informative_prior_function():
    """Test informative prior function."""
    # Create test data
    key = jax.random.PRNGKey(0)
    num_genes = 10

    # Call function with default parameters
    params = informative_prior_function(key, num_genes)

    # Check that all required parameters are present
    assert "alpha" in params
    assert "beta" in params
    assert "gamma" in params

    # Check shapes
    assert params["alpha"].shape == (num_genes,)
    assert params["beta"].shape == (num_genes,)
    assert params["gamma"].shape == (num_genes,)

    # Check that values are positive (lognormal distribution)
    assert jnp.all(params["alpha"] > 0)
    assert jnp.all(params["beta"] > 0)
    assert jnp.all(params["gamma"] > 0)

    # Call function with custom parameters
    prior_params = {
        "alpha_loc": 0.0,
        "alpha_scale": 0.5,
        "beta_loc": -1.0,
        "beta_scale": 0.5,
        "gamma_loc": -1.5,
        "gamma_scale": 0.5,
    }
    params = informative_prior_function(key, num_genes, prior_params)

    # Check that all required parameters are present
    assert "alpha" in params
    assert "beta" in params
    assert "gamma" in params

    # Check shapes
    assert params["alpha"].shape == (num_genes,)
    assert params["beta"].shape == (num_genes,)
    assert params["gamma"].shape == (num_genes,)

    # Check that values are positive (lognormal distribution)
    assert jnp.all(params["alpha"] > 0)
    assert jnp.all(params["beta"] > 0)
    assert jnp.all(params["gamma"] > 0)


def test_register_standard_priors():
    """Test registration of standard prior functions."""
    # Check that lognormal prior function is registered
    lognormal_fn = get_prior("lognormal")
    assert lognormal_fn is not None

    # Check that informative prior function is registered
    informative_fn = get_prior("informative")
    assert informative_fn is not None
