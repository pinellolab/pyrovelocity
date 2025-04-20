"""
Tests for the SVI inference components.
"""

import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pyrovelocity.models.jax.inference.svi import (
    run_svi_inference,
    create_svi,
    create_optimizer,
)
from pyrovelocity.models.jax.inference.guide import auto_normal_guide
from pyrovelocity.models.jax.core.state import InferenceConfig


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


def test_create_optimizer():
    """Test creating an optimizer."""
    # Get Adam optimizer
    optimizer = create_optimizer("adam", learning_rate=0.01)
    
    # Check that optimizer is a callable
    assert callable(optimizer)
    
    # Get SGD optimizer
    optimizer = create_optimizer("sgd", learning_rate=0.01)
    
    # Check that optimizer is a callable
    assert callable(optimizer)
    
    # Check that an error is raised for unknown optimizer
    with pytest.raises(ValueError):
        create_optimizer("unknown", learning_rate=0.01)


def test_create_svi():
    """Test creating an SVI object."""
    # Create guide
    guide = auto_normal_guide(simple_model)
    
    # Create SVI object with Adam optimizer
    svi = create_svi(simple_model, guide, optimizer="adam", learning_rate=0.01)
    
    # Check that SVI object has expected attributes
    assert hasattr(svi, "update")
    assert hasattr(svi, "evaluate")
    assert hasattr(svi, "get_params")
    
    # Create SVI object with SGD optimizer
    svi = create_svi(simple_model, guide, optimizer="sgd", learning_rate=0.01)
    
    # Check that SVI object has expected attributes
    assert hasattr(svi, "update")
    assert hasattr(svi, "evaluate")
    assert hasattr(svi, "get_params")


def test_run_svi_inference():
    """Test running SVI inference."""
    # Create data
    key = jax.random.PRNGKey(0)
    x = jnp.ones(10)
    
    # Create guide
    guide = auto_normal_guide(simple_model)
    
    # Create inference config
    config = InferenceConfig(
        method="svi",
        guide_type="auto_normal",
        optimizer="adam",
        learning_rate=0.1,
        num_epochs=10,
        batch_size=None,
        clip_norm=None,
        early_stopping=False,
        early_stopping_patience=None,
    )
    
    # Run SVI inference
    state, posterior_samples = run_svi_inference(
        model=simple_model,
        guide=guide,
        args=(),
        kwargs={"x": x},
        config=config,
        key=key,
    )
    
    # Check that state has expected attributes
    assert hasattr(state, "params")
    assert hasattr(state, "loss_history")
    assert len(state.loss_history) == 10  # 10 epochs
    
    # Check that posterior_samples is a dictionary
    assert isinstance(posterior_samples, dict)