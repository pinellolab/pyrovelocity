"""Tests for PyroVelocity JAX/NumPyro likelihood models."""

import jax
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
    negative_binomial_likelihood,
    create_likelihood,
)


def test_poisson_likelihood_interface():
    """Test poisson_likelihood interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
    }
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        poisson_likelihood(ut, st, scaling_params)


def test_poisson_likelihood_type_checking():
    """Test poisson_likelihood type checking."""
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
    }
    
    # Invalid ut type
    with pytest.raises(BeartypeCallHintParamViolation):
        poisson_likelihood("not_an_array", st, scaling_params)
    
    # Invalid st type
    with pytest.raises(BeartypeCallHintParamViolation):
        poisson_likelihood(ut, "not_an_array", scaling_params)
    
    # Invalid scaling_params type
    with pytest.raises(BeartypeCallHintParamViolation):
        poisson_likelihood(ut, st, "not_a_dict")


def test_negative_binomial_likelihood_interface():
    """Test negative_binomial_likelihood interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
        "u_dispersion": jnp.array([0.1, 0.2, 0.3]),
        "s_dispersion": jnp.array([0.05, 0.1, 0.15]),
    }
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        negative_binomial_likelihood(ut, st, scaling_params)


def test_negative_binomial_likelihood_type_checking():
    """Test negative_binomial_likelihood type checking."""
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
        "u_dispersion": jnp.array([0.1, 0.2, 0.3]),
        "s_dispersion": jnp.array([0.05, 0.1, 0.15]),
    }
    
    # Invalid ut type
    with pytest.raises(BeartypeCallHintParamViolation):
        negative_binomial_likelihood("not_an_array", st, scaling_params)
    
    # Invalid st type
    with pytest.raises(BeartypeCallHintParamViolation):
        negative_binomial_likelihood(ut, "not_an_array", scaling_params)
    
    # Invalid scaling_params type
    with pytest.raises(BeartypeCallHintParamViolation):
        negative_binomial_likelihood(ut, st, "not_a_dict")


def test_create_likelihood():
    """Test create_likelihood function."""
    # Test with poisson likelihood
    likelihood_fn = create_likelihood("poisson")
    assert likelihood_fn == poisson_likelihood
    
    # Test with negative_binomial likelihood
    likelihood_fn = create_likelihood("negative_binomial")
    assert likelihood_fn == negative_binomial_likelihood
    
    # Test with unknown likelihood
    with pytest.raises(ValueError):
        create_likelihood("unknown_likelihood")


def test_create_likelihood_type_checking():
    """Test create_likelihood type checking."""
    # Invalid likelihood_type type
    with pytest.raises(BeartypeCallHintParamViolation):
        create_likelihood(123)