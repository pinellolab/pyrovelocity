"""
Tests for PyroVelocity JAX/NumPyro observation components.

This module contains tests for the observation components, including:

- test_standard_observation_function: Test standard observation function
- test_normalized_observation_function: Test normalized observation function
- test_register_standard_observations: Test registration of standard observation functions
"""

import jax.numpy as jnp
import numpy as np

from pyrovelocity.models.jax.components.observations import (
    normalized_observation_function,
    standard_observation_function,
)
from pyrovelocity.models.jax.registry import get_observation


def test_standard_observation_function():
    """Test standard observation function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))

    # Call function with default parameters
    u_transformed, s_transformed = standard_observation_function(u_obs, s_obs)

    # Check shapes
    assert u_transformed.shape == (batch_size, n_cells, n_genes)
    assert s_transformed.shape == (batch_size, n_cells, n_genes)

    # Check that values are unchanged by default
    np.testing.assert_allclose(u_transformed, u_obs)
    np.testing.assert_allclose(s_transformed, s_obs)

    # Test with log1p transformation
    observation_params = {"log1p": True}
    u_transformed, s_transformed = standard_observation_function(
        u_obs, s_obs, observation_params
    )

    # Check that values are log1p transformed
    np.testing.assert_allclose(u_transformed, jnp.log1p(u_obs))
    np.testing.assert_allclose(s_transformed, jnp.log1p(s_obs))

    # Test with normalization
    observation_params = {"normalize": True}
    u_transformed, s_transformed = standard_observation_function(
        u_obs, s_obs, observation_params
    )

    # Check that values are normalized (approximately)
    u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
    s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)
    np.testing.assert_allclose(u_transformed, u_obs / u_size_factor, rtol=1e-5)
    np.testing.assert_allclose(s_transformed, s_obs / s_size_factor, rtol=1e-5)

    # Test with both log1p and normalization
    observation_params = {"log1p": True, "normalize": True}
    u_transformed, s_transformed = standard_observation_function(
        u_obs, s_obs, observation_params
    )

    # Check that values are log1p transformed and normalized (approximately)
    u_log1p = jnp.log1p(u_obs)
    s_log1p = jnp.log1p(s_obs)
    u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
    s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        u_transformed, u_log1p / u_size_factor, rtol=1e-5
    )
    np.testing.assert_allclose(
        s_transformed, s_log1p / s_size_factor, rtol=1e-5
    )


def test_normalized_observation_function():
    """Test normalized observation function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))

    # Call function with default parameters
    u_transformed, s_transformed = normalized_observation_function(u_obs, s_obs)

    # Check shapes
    assert u_transformed.shape == (batch_size, n_cells, n_genes)
    assert s_transformed.shape == (batch_size, n_cells, n_genes)

    # Check that values are normalized (approximately)
    u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
    s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)
    np.testing.assert_allclose(u_transformed, u_obs / u_size_factor, rtol=1e-5)
    np.testing.assert_allclose(s_transformed, s_obs / s_size_factor, rtol=1e-5)

    # Test with scaling
    observation_params = {"scale_factor": 2.0}
    u_transformed, s_transformed = normalized_observation_function(
        u_obs, s_obs, observation_params
    )

    # Check that values are normalized and scaled (approximately)
    u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
    s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        u_transformed, u_obs / u_size_factor * 2.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        s_transformed, s_obs / s_size_factor * 2.0, rtol=1e-5
    )


def test_register_standard_observations():
    """Test registration of standard observation functions."""
    # Check that standard observation function is registered
    standard_fn = get_observation("standard")
    assert standard_fn is not None

    # Check that normalized observation function is registered
    normalized_fn = get_observation("normalized")
    assert normalized_fn is not None
