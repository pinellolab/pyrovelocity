"""
Tests for PyroVelocity JAX/NumPyro guide components.

This module contains tests for the guide components, including:

- test_auto_normal_guide_factory: Test auto normal guide factory
- test_auto_delta_guide_factory: Test auto delta guide factory
- test_custom_guide_factory: Test custom guide factory
- test_register_standard_guides: Test registration of standard guide functions
"""

import jax.numpy as jnp
import numpyro
from numpyro.distributions import Normal

from pyrovelocity.models.jax.components.guides import (
    auto_delta_guide_factory,
    auto_normal_guide_factory,
    custom_guide_factory,
)
from pyrovelocity.models.jax.registry import get_guide


# Define a simple model for testing guides
def simple_model(u_obs, s_obs):
    """Simple model for testing guides."""
    # Get dimensions
    batch_size, n_cells, n_genes = u_obs.shape

    # Sample parameters
    with numpyro.plate("gene", n_genes):
        numpyro.sample("alpha", Normal(0.0, 1.0))
        numpyro.sample("beta", Normal(0.0, 1.0))
        numpyro.sample("gamma", Normal(0.0, 1.0))

    # Sample latent time
    with numpyro.plate("cell", n_cells):
        numpyro.sample("tau", Normal(0.0, 1.0))

    # Compute expected counts (simplified)
    u_expected = jnp.ones_like(u_obs)
    s_expected = jnp.ones_like(s_obs)

    # Sample observations
    with numpyro.plate("batch", batch_size):
        numpyro.sample("u", Normal(0.0, 1.0).to_event(2), obs=u_obs)
        numpyro.sample("s", Normal(0.0, 1.0).to_event(2), obs=s_obs)


def test_auto_normal_guide_factory():
    """Test auto normal guide factory."""
    # Create guide
    guide = auto_normal_guide_factory(simple_model)

    # Check that guide is callable
    assert callable(guide)


def test_auto_delta_guide_factory():
    """Test auto delta guide factory."""
    # Create guide
    guide = auto_delta_guide_factory(simple_model)

    # Check that guide is callable
    assert callable(guide)


def test_custom_guide_factory():
    """Test custom guide factory."""
    # Create test data
    n_cells = 2
    n_genes = 3

    # Create guide with custom parameters
    guide_params = {
        "num_genes": n_genes,
        "num_cells": n_cells,
        "alpha_loc": 0.0,
        "alpha_scale": 1.0,
        "beta_loc": 0.0,
        "beta_scale": 1.0,
        "gamma_loc": 0.0,
        "gamma_scale": 1.0,
        "tau_loc": 0.0,
        "tau_scale": 1.0,
    }
    guide = custom_guide_factory(simple_model, guide_params)

    # Check that guide is callable
    assert callable(guide)


def test_register_standard_guides():
    """Test registration of standard guide functions."""
    # Check that guide factories are registered
    auto_fn = get_guide("auto")
    assert auto_fn is not None

    auto_normal_fn = get_guide("auto_normal")
    assert auto_normal_fn is not None

    auto_delta_fn = get_guide("auto_delta")
    assert auto_delta_fn is not None

    custom_fn = get_guide("custom")
    assert custom_fn is not None
