"""Tests for PyroVelocity JAX/NumPyro likelihood models."""

import jax
import jax.numpy as jnp
import pytest
import numpyro.distributions as dist
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
    negative_binomial_likelihood,
    create_likelihood,
)


def test_poisson_likelihood_interface():
    """Test poisson_likelihood interface."""
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
    }
    
    # Call the function
    u_dist, s_dist = poisson_likelihood(ut, st, scaling_params)
    
    # Check that the result is a tuple of distributions
    assert isinstance(u_dist, dist.Distribution)
    assert isinstance(s_dist, dist.Distribution)
    
    # Check that the distributions are Poisson
    assert isinstance(u_dist, dist.Poisson)
    assert isinstance(s_dist, dist.Poisson)
    
    # Check that the rates are correctly computed
    # In the updated implementation, the rates are 2D arrays with shape (batch, features)
    # where batch is the number of cells and features is the number of genes
    expected_u_rate = jnp.exp(scaling_params["u_log_library"][:, jnp.newaxis]) * ut
    expected_s_rate = jnp.exp(scaling_params["s_log_library"][:, jnp.newaxis]) * st
    
    assert jnp.allclose(u_dist.rate, expected_u_rate)
    assert jnp.allclose(s_dist.rate, expected_s_rate)


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
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([1.0, 2.0, 3.0]),
        "s_log_library": jnp.array([0.5, 1.0, 1.5]),
        "u_dispersion": jnp.array([0.1, 0.2, 0.3]),
        "s_dispersion": jnp.array([0.05, 0.1, 0.15]),
    }
    
    # Call the function
    u_dist, s_dist = negative_binomial_likelihood(ut, st, scaling_params)
    
    # Check that the result is a tuple of distributions
    assert isinstance(u_dist, dist.Distribution)
    assert isinstance(s_dist, dist.Distribution)
    
    # Check that the distributions are GammaPoisson (Negative Binomial)
    assert isinstance(u_dist, dist.GammaPoisson)
    assert isinstance(s_dist, dist.GammaPoisson)
    
    # Check that the parameters are correctly computed
    # In the updated implementation, the rates are 2D arrays with shape (batch, features)
    u_rate = jnp.exp(scaling_params["u_log_library"][:, jnp.newaxis]) * ut
    s_rate = jnp.exp(scaling_params["s_log_library"][:, jnp.newaxis]) * st
    u_disp = scaling_params["u_dispersion"]
    s_disp = scaling_params["s_dispersion"]
    
    # Check concentration parameter (1/dispersion)
    assert jnp.allclose(u_dist.concentration, 1.0 / u_disp)
    assert jnp.allclose(s_dist.concentration, 1.0 / s_disp)
    
    # Check rate parameter (1/(dispersion * mean))
    expected_u_rate = 1.0 / (u_disp * u_rate)
    expected_s_rate = 1.0 / (s_disp * s_rate)
    assert jnp.allclose(u_dist.rate, expected_u_rate)
    assert jnp.allclose(s_dist.rate, expected_s_rate)


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


def test_poisson_likelihood_default_scaling():
    """Test poisson_likelihood with default scaling parameters."""
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    
    # Call the function with minimal scaling_params
    # We need to provide at least empty arrays for log_library
    scaling_params = {
        "u_log_library": jnp.zeros(3),
        "s_log_library": jnp.zeros(3),
    }
    u_dist, s_dist = poisson_likelihood(ut, st, scaling_params)
    
    # Check that the distributions use the data directly as rates (with broadcasting)
    expected_u_rate = jnp.exp(scaling_params["u_log_library"][:, jnp.newaxis]) * ut
    expected_s_rate = jnp.exp(scaling_params["s_log_library"][:, jnp.newaxis]) * st
    assert jnp.allclose(u_dist.rate, expected_u_rate)
    assert jnp.allclose(s_dist.rate, expected_s_rate)


def test_negative_binomial_likelihood_default_scaling():
    """Test negative_binomial_likelihood with default scaling parameters."""
    # Prepare test inputs
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    
    # Call the function with minimal scaling_params
    # We need to provide at least empty arrays for log_library
    scaling_params = {
        "u_log_library": jnp.zeros(3),
        "s_log_library": jnp.zeros(3),
    }
    u_dist, s_dist = negative_binomial_likelihood(ut, st, scaling_params)
    
    # Check that the parameters are correctly computed with defaults
    # Default dispersion is 1.0, default log_library is 0.0
    
    # Check concentration parameter (1/dispersion)
    assert jnp.allclose(u_dist.concentration, 1.0)
    assert jnp.allclose(s_dist.concentration, 1.0)
    
    # Check rate parameter (1/(dispersion * mean))
    expected_u_rate = 1.0 / (jnp.ones_like(u_dist.rate) * jnp.exp(scaling_params["u_log_library"][:, jnp.newaxis]) * ut)
    expected_s_rate = 1.0 / (jnp.ones_like(s_dist.rate) * jnp.exp(scaling_params["s_log_library"][:, jnp.newaxis]) * st)
    assert jnp.allclose(u_dist.rate, expected_u_rate)
    assert jnp.allclose(s_dist.rate, expected_s_rate)


def test_likelihood_sampling():
    """Test that we can sample from the likelihood distributions."""
    # Prepare test inputs
    key = jax.random.PRNGKey(42)
    ut = jnp.array([10.0, 20.0, 30.0])
    st = jnp.array([5.0, 10.0, 15.0])
    scaling_params = {
        "u_log_library": jnp.array([0.0, 0.0, 0.0]),
        "s_log_library": jnp.array([0.0, 0.0, 0.0]),
    }
    
    # Get distributions
    u_dist, s_dist = poisson_likelihood(ut, st, scaling_params)
    
    # Sample from distributions
    key1, key2 = jax.random.split(key)
    u_samples = u_dist.sample(key1, sample_shape=(10,))
    s_samples = s_dist.sample(key2, sample_shape=(10,))
    
    # Check shapes - with our broadcasting changes, the shape is now (10, 3, 3)
    # because the distribution has shape (3, 3) and we sample 10 times
    assert u_samples.shape[0] == 10  # Sample dimension
    assert u_samples.shape[1:] == u_dist.batch_shape  # Distribution batch shape
    assert s_samples.shape[0] == 10
    assert s_samples.shape[1:] == s_dist.batch_shape
    
    # Check that samples are non-negative integers
    assert jnp.all(u_samples >= 0)
    assert jnp.all(s_samples >= 0)