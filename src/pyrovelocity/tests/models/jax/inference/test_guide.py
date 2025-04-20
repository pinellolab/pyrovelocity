"""
Tests for the guide components.
"""

import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pyrovelocity.models.jax.inference.guide import (
    auto_normal_guide,
    auto_delta_guide,
    custom_guide,
    create_guide,
)


# Define a simple model for testing
def simple_model(x=None):
    """Simple model for testing."""
    # Sample parameters
    alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
    
    # Sample observations
    with numpyro.plate("data", 10):
        if x is not None:
            numpyro.sample("x_obs", dist.Poisson(alpha * beta), obs=x)
    
    # Return expected values
    return {
        "expected": alpha * beta,
    }


def test_auto_normal_guide():
    """Test auto normal guide."""
    # Create guide
    guide = auto_normal_guide(simple_model)
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Check that guide is an AutoNormal instance
    assert isinstance(guide, numpyro.infer.autoguide.AutoNormal)


def test_auto_delta_guide():
    """Test auto delta guide."""
    # Create guide
    guide = auto_delta_guide(simple_model)
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Check that guide is an AutoDelta instance
    assert isinstance(guide, numpyro.infer.autoguide.AutoDelta)


def test_custom_guide():
    """Test custom guide."""
    # Create guide with default parameters
    guide = custom_guide(simple_model)
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Create guide with custom parameters
    init_params = {
        "alpha_loc": jnp.zeros(1),
        "beta_loc": jnp.zeros(1),
        "gamma_loc": jnp.zeros(1),
    }
    guide = custom_guide(simple_model, init_params=init_params)
    
    # Check that guide is a callable
    assert callable(guide)


def test_create_guide():
    """Test creating a guide."""
    # Create auto normal guide
    guide = create_guide(simple_model, guide_type="auto_normal")
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Check that guide is an AutoNormal instance
    assert isinstance(guide, numpyro.infer.autoguide.AutoNormal)
    
    # Create auto delta guide
    guide = create_guide(simple_model, guide_type="auto_delta")
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Check that guide is an AutoDelta instance
    assert isinstance(guide, numpyro.infer.autoguide.AutoDelta)
    
    # Create custom guide
    init_params = {
        "alpha_loc": jnp.zeros(1),
        "beta_loc": jnp.zeros(1),
        "gamma_loc": jnp.zeros(1),
    }
    guide = create_guide(simple_model, guide_type="custom", init_params=init_params)
    
    # Check that guide is a callable
    assert callable(guide)
    
    # Check that an error is raised for unknown guide type
    with pytest.raises(ValueError):
        create_guide(simple_model, guide_type="unknown")