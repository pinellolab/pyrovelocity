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
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)
    loc = 0.0
    scale = 1.0

    # Call the function
    result = lognormal_prior(key, shape, loc, scale)

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict)
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result

    # Check that the values have the expected shape
    assert result["alpha"].shape == shape
    assert result["beta"].shape == shape
    assert result["gamma"].shape == shape

    # Check that the values are positive (log-normal distribution)
    assert jnp.all(result["alpha"] > 0)
    assert jnp.all(result["beta"] > 0)
    assert jnp.all(result["gamma"] > 0)


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

    # Call the function
    result = informative_prior(key, shape, prior_params)

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict)
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result

    # Check that the values have the expected shape
    assert result["alpha"].shape == shape
    assert result["beta"].shape == shape
    assert result["gamma"].shape == shape

    # Check that the values are positive (log-normal distribution)
    assert jnp.all(result["alpha"] > 0)
    assert jnp.all(result["beta"] > 0)
    assert jnp.all(result["gamma"] > 0)


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
    # Prepare test inputs
    key = create_key(42)
    num_genes = 3

    # Test with lognormal prior
    result_lognormal = sample_prior_parameters(
        key=key, num_genes=num_genes, prior_type="lognormal"
    )

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result_lognormal, dict)
    assert "alpha" in result_lognormal
    assert "beta" in result_lognormal
    assert "gamma" in result_lognormal

    # Check that the values have the expected shape
    assert result_lognormal["alpha"].shape == (num_genes,)
    assert result_lognormal["beta"].shape == (num_genes,)
    assert result_lognormal["gamma"].shape == (num_genes,)

    # Test with informative prior
    prior_params = {
        "alpha_loc": 0.0,
        "alpha_scale": 1.0,
        "beta_loc": 0.0,
        "beta_scale": 1.0,
        "gamma_loc": 0.0,
        "gamma_scale": 1.0,
    }
    result_informative = sample_prior_parameters(
        key, num_genes, "informative", prior_params
    )

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result_informative, dict)
    assert "alpha" in result_informative
    assert "beta" in result_informative
    assert "gamma" in result_informative

    # Check that the values have the expected shape
    assert result_informative["alpha"].shape == (num_genes,)
    assert result_informative["beta"].shape == (num_genes,)
    assert result_informative["gamma"].shape == (num_genes,)


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


def test_sample_prior_parameters_unknown_prior():
    """Test sample_prior_parameters with unknown prior type."""
    # Prepare test inputs
    key = create_key(42)
    num_genes = 3

    # Check that the function raises ValueError for unknown prior type
    with pytest.raises(ValueError):
        sample_prior_parameters(key, num_genes, "unknown_prior")


def test_prior_deterministic_with_same_key():
    """Test that priors are deterministic with the same key."""
    # Prepare test inputs
    key = create_key(42)
    shape = (3,)

    # Call the function twice with the same key
    result1 = lognormal_prior(key, shape)
    result2 = lognormal_prior(key, shape)

    # Check that the results are identical
    assert jnp.array_equal(result1["alpha"], result2["alpha"])
    assert jnp.array_equal(result1["beta"], result2["beta"])
    assert jnp.array_equal(result1["gamma"], result2["gamma"])


def test_prior_different_with_different_keys():
    """Test that priors are different with different keys."""
    # Prepare test inputs
    key1 = create_key(42)
    key2 = create_key(43)
    shape = (3,)

    # Call the function with different keys
    result1 = lognormal_prior(key1, shape)
    result2 = lognormal_prior(key2, shape)

    # Check that the results are different
    assert not jnp.array_equal(result1["alpha"], result2["alpha"])
    assert not jnp.array_equal(result1["beta"], result2["beta"])
    assert not jnp.array_equal(result1["gamma"], result2["gamma"])
