"""Tests for PyroVelocity JAX/NumPyro prior distributions."""

import jax
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.priors import (
    lognormal_prior,
    informative_prior,
    sample_prior_parameters,
)
from pyrovelocity.models.jax.core.utils import create_key


def test_lognormal_prior_interface():
    """Test lognormal_prior interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)
    loc = 0.0
    scale = 1.0
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        lognormal_prior(key, shape, loc, scale)


def test_lognormal_prior_type_checking():
    """Test lognormal_prior type checking."""
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)
    loc = 0.0
    scale = 1.0
    
    # Invalid key type
    with pytest.raises(BeartypeCallHintParamViolation):
        lognormal_prior("not_a_key", shape, loc, scale)
    
    # Invalid shape type
    with pytest.raises(BeartypeCallHintParamViolation):
        lognormal_prior(key, "not_a_shape", loc, scale)
    
    # Invalid loc type
    with pytest.raises(BeartypeCallHintParamViolation):
        lognormal_prior(key, shape, "not_a_float", scale)
    
    # Invalid scale type
    with pytest.raises(BeartypeCallHintParamViolation):
        lognormal_prior(key, shape, loc, "not_a_float")


def test_informative_prior_interface():
    """Test informative_prior interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)
    prior_params = {
        "alpha_loc": 0.0,
        "alpha_scale": 1.0,
        "beta_loc": 0.0,
        "beta_scale": 1.0,
        "gamma_loc": 0.0,
        "gamma_scale": 1.0,
    }
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        informative_prior(key, shape, prior_params)


def test_informative_prior_type_checking():
    """Test informative_prior type checking."""
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)
    prior_params = {
        "alpha_loc": 0.0,
        "alpha_scale": 1.0,
        "beta_loc": 0.0,
        "beta_scale": 1.0,
        "gamma_loc": 0.0,
        "gamma_scale": 1.0,
    }
    
    # Invalid key type
    with pytest.raises(BeartypeCallHintParamViolation):
        informative_prior("not_a_key", shape, prior_params)
    
    # Invalid shape type
    with pytest.raises(BeartypeCallHintParamViolation):
        informative_prior(key, "not_a_shape", prior_params)
    
    # Invalid prior_params type
    with pytest.raises(BeartypeCallHintParamViolation):
        informative_prior(key, shape, "not_a_dict")


def test_sample_prior_parameters_interface():
    """Test sample_prior_parameters interface."""
    # This test only checks that the function has the correct interface
    # The actual implementation will be tested in a future phase
    
    # Prepare test inputs
    key = create_key(42)
    num_genes = 3
    prior_type = "lognormal"
    
    # Check that the function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        sample_prior_parameters(key, num_genes, prior_type)


def test_sample_prior_parameters_type_checking():
    """Test sample_prior_parameters type checking."""
    # Prepare test inputs
    key = create_key(42)
    num_genes = 3
    prior_type = "lognormal"
    
    # Invalid key type
    with pytest.raises(BeartypeCallHintParamViolation):
        sample_prior_parameters("not_a_key", num_genes, prior_type)
    
    # Invalid num_genes type
    with pytest.raises(BeartypeCallHintParamViolation):
        sample_prior_parameters(key, "not_an_int", prior_type)
    
    # Invalid prior_type type
    with pytest.raises(BeartypeCallHintParamViolation):
        sample_prior_parameters(key, num_genes, 123)