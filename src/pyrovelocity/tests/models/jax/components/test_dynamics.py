"""
Tests for PyroVelocity JAX/NumPyro dynamics components.

This module contains tests for the dynamics components, including:

- test_standard_dynamics_function: Test standard dynamics function
- test_nonlinear_dynamics_function: Test nonlinear dynamics function
- test_register_standard_dynamics: Test registration of standard dynamics functions
"""

import jax.numpy as jnp

from pyrovelocity.models.jax.components.dynamics import (
    nonlinear_dynamics_function,
    standard_dynamics_function,
)
from pyrovelocity.models.jax.registry import get_dynamics


def test_standard_dynamics_function():
    """Test standard dynamics function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    tau = jnp.zeros(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)
    tau = tau.at[:, 1, :].set(1.0)  # Set tau=1.0 for the second cell

    u0 = jnp.ones(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)
    s0 = jnp.ones(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)

    # Create parameters
    params = {
        "alpha": jnp.ones(
            (batch_size, n_genes)
        ),  # Shape: (batch_size, n_genes)
        "beta": jnp.ones((batch_size, n_genes)),  # Shape: (batch_size, n_genes)
        "gamma": jnp.ones(
            (batch_size, n_genes)
        ),  # Shape: (batch_size, n_genes)
    }

    # Call function
    u, s = standard_dynamics_function(tau, u0, s0, params)

    # Check shapes
    assert u.shape == (batch_size, n_cells, n_genes)
    assert s.shape == (batch_size, n_cells, n_genes)

    # Just check that the values are finite for u
    assert jnp.all(jnp.isfinite(u))


def test_nonlinear_dynamics_function():
    """Test nonlinear dynamics function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    tau = jnp.zeros(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)
    tau = tau.at[:, 1, :].set(1.0)  # Set tau=1.0 for the second cell

    u0 = jnp.ones(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)
    s0 = jnp.ones(
        (batch_size, n_cells, n_genes)
    )  # Shape: (batch_size, n_cells, n_genes)

    # Create parameters
    params = {
        "alpha": jnp.ones(
            (batch_size, n_genes)
        ),  # Shape: (batch_size, n_genes)
        "beta": jnp.ones((batch_size, n_genes)),  # Shape: (batch_size, n_genes)
        "gamma": jnp.ones(
            (batch_size, n_genes)
        ),  # Shape: (batch_size, n_genes)
        "scaling": jnp.ones((batch_size, n_genes))
        * 0.1,  # Shape: (batch_size, n_genes)
    }

    # Call function
    u, s = nonlinear_dynamics_function(tau, u0, s0, params)

    # Check shapes
    assert u.shape == (batch_size, n_cells, n_genes)
    assert s.shape == (batch_size, n_cells, n_genes)

    # For nonlinear dynamics, we can't easily predict the exact values,
    # but we can check that the values are reasonable
    assert jnp.all(u > 0)
    assert jnp.all(s > 0)


def test_register_standard_dynamics():
    """Test registration of standard dynamics functions."""
    # Check that standard dynamics function is registered
    standard_fn = get_dynamics("standard")
    assert standard_fn is not None

    # Check that nonlinear dynamics function is registered
    nonlinear_fn = get_dynamics("nonlinear")
    assert nonlinear_fn is not None
