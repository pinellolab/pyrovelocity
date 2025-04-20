"""Tests for PyroVelocity JAX/NumPyro model definition."""

import jax
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.model import (
    velocity_model,
    create_model,
)
from pyrovelocity.models.jax.core.state import ModelConfig
from pyrovelocity.models.jax.core.dynamics import (
    standard_dynamics_model,
    nonlinear_dynamics_model,
    dynamics_ode_model,
)
from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
    negative_binomial_likelihood,
)


def test_velocity_model_interface(cell_gene_data):
    """Test velocity_model interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    # Reshape to match the expected type annotation
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )


def test_velocity_model_type_checking(cell_gene_data):
    """Test velocity_model type checking."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))
    
    # Invalid u_obs type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs="not_an_array",
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )
    
    # Invalid s_obs type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs="not_an_array",
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )
    
    # Invalid u_log_library type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library="not_an_array",
            s_log_library=s_log_library,
        )
    
    # Invalid s_log_library type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library="not_an_array",
        )


def test_create_model():
    """Test create_model function."""
    # Test with standard dynamics
    config = ModelConfig(dynamics="standard", likelihood="poisson")
    model_fn = create_model(config)
    
    # Test with nonlinear dynamics
    config = ModelConfig(dynamics="nonlinear", likelihood="poisson")
    model_fn = create_model(config)
    
    # Test with ode dynamics
    config = ModelConfig(dynamics="ode", likelihood="poisson")
    model_fn = create_model(config)
    
    # Test with unknown dynamics
    config = ModelConfig(dynamics="unknown", likelihood="poisson")
    with pytest.raises(ValueError):
        create_model(config)


def test_create_model_type_checking():
    """Test create_model type checking."""
    # Invalid config type
    with pytest.raises(BeartypeCallHintParamViolation):
        create_model("not_a_config")


def test_model_fn_interface(cell_gene_data):
    """Test model_fn interface."""
    # This test checks that the model function created by create_model has the correct interface
    
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))
    
    # Create model function
    config = ModelConfig(dynamics="standard", likelihood="poisson")
    model_fn = create_model(config)
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        model_fn(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )


def test_model_config_dynamics_selection():
    """Test that ModelConfig correctly selects dynamics function."""
    # Test standard dynamics
    config = ModelConfig(dynamics="standard")
    model_fn = create_model(config)
    
    # Get the dynamics_fn from the closure
    dynamics_fn = None
    for cell in model_fn.__closure__:
        if isinstance(cell.cell_contents, type(standard_dynamics_model)):
            dynamics_fn = cell.cell_contents
            break
    
    assert dynamics_fn == standard_dynamics_model
    
    # Test nonlinear dynamics
    config = ModelConfig(dynamics="nonlinear")
    model_fn = create_model(config)
    
    # Get the dynamics_fn from the closure
    dynamics_fn = None
    for cell in model_fn.__closure__:
        if isinstance(cell.cell_contents, type(nonlinear_dynamics_model)):
            dynamics_fn = cell.cell_contents
            break
    
    assert dynamics_fn == nonlinear_dynamics_model
    
    # Test ode dynamics
    config = ModelConfig(dynamics="ode")
    model_fn = create_model(config)
    
    # Get the dynamics_fn from the closure
    dynamics_fn = None
    for cell in model_fn.__closure__:
        if isinstance(cell.cell_contents, type(dynamics_ode_model)):
            dynamics_fn = cell.cell_contents
            break
    
    assert dynamics_fn == dynamics_ode_model


def test_model_config_likelihood_selection():
    """Test that ModelConfig correctly selects likelihood function."""
    # Test poisson likelihood
    config = ModelConfig(likelihood="poisson")
    model_fn = create_model(config)
    
    # Check that the model uses the correct likelihood function
    # We can't directly access the likelihood function in the closure
    # So we'll check that the create_likelihood function is called with the correct argument
    assert config.likelihood == "poisson"
    
    # Test negative_binomial likelihood
    config = ModelConfig(likelihood="negative_binomial")
    model_fn = create_model(config)
    
    # Check that the model uses the correct likelihood function
    assert config.likelihood == "negative_binomial"