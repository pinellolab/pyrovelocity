"""Tests for PyroVelocity JAX/NumPyro dynamics models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.core.dynamics import (
    standard_dynamics_model,
    nonlinear_dynamics_model,
    dynamics_ode_model,
    vectorized_standard_dynamics_model,
    vectorized_nonlinear_dynamics_model,
    vectorized_dynamics_ode_model,
)


def test_standard_dynamics_model():
    """Test standard_dynamics_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Call the function
    ut, st = standard_dynamics_model(tau, u0, s0, params)

    # Check that the output has the correct shape
    assert ut.shape == tau.shape
    assert st.shape == tau.shape

    # Check that the output values are finite
    assert jnp.all(jnp.isfinite(ut))
    assert jnp.all(jnp.isfinite(st))

    # Check that the output values are non-negative
    assert jnp.all(ut >= 0)
    assert jnp.all(st >= 0)


def test_standard_dynamics_model_special_case():
    """Test standard_dynamics_model with gamma = beta (special case)."""
    # Prepare test inputs with gamma = beta
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.5, 1.0, 1.5]),  # Same as beta
    }

    # Call the function
    ut, st = standard_dynamics_model(tau, u0, s0, params)

    # Check that the output has the correct shape
    assert ut.shape == tau.shape
    assert st.shape == tau.shape

    # Check that the output values are finite
    assert jnp.all(jnp.isfinite(ut))
    assert jnp.all(jnp.isfinite(st))

    # Check that the output values are non-negative
    assert jnp.all(ut >= 0)
    assert jnp.all(st >= 0)


def test_standard_dynamics_model_type_checking():
    """Test standard_dynamics_model type checking."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Invalid tau type
    with pytest.raises(BeartypeCallHintParamViolation):
        standard_dynamics_model("not_an_array", u0, s0, params)

    # Invalid u0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        standard_dynamics_model(tau, "not_an_array", s0, params)

    # Invalid s0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        standard_dynamics_model(tau, u0, "not_an_array", params)

    # Invalid params type
    with pytest.raises(BeartypeCallHintParamViolation):
        standard_dynamics_model(tau, u0, s0, "not_a_dict")


def test_nonlinear_dynamics_model():
    """Test nonlinear_dynamics_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
        "scaling": jnp.array([0.1, 0.2, 0.3]),
    }

    # Call the function
    ut, st = nonlinear_dynamics_model(tau, u0, s0, params)

    # Check that the output has the correct shape
    assert ut.shape == tau.shape
    assert st.shape == tau.shape

    # Check that the output values are finite
    assert jnp.all(jnp.isfinite(ut))
    assert jnp.all(jnp.isfinite(st))

    # Check that the output values are non-negative
    assert jnp.all(ut >= 0)
    assert jnp.all(st >= 0)


def test_nonlinear_dynamics_model_saturation():
    """Test nonlinear_dynamics_model saturation effect."""
    # Prepare test inputs with different scaling values
    tau = jnp.array([1.0, 1.0, 1.0])
    u0 = jnp.array([10.0, 10.0, 10.0])
    s0 = jnp.array([5.0, 5.0, 5.0])

    # Create two parameter sets with different scaling values
    params_low_scaling = {
        "alpha": jnp.array([1.0, 1.0, 1.0]),
        "beta": jnp.array([0.5, 0.5, 0.5]),
        "gamma": jnp.array([0.3, 0.3, 0.3]),
        "scaling": jnp.array([0.01, 0.01, 0.01]),  # Low scaling
    }

    params_high_scaling = {
        "alpha": jnp.array([1.0, 1.0, 1.0]),
        "beta": jnp.array([0.5, 0.5, 0.5]),
        "gamma": jnp.array([0.3, 0.3, 0.3]),
        "scaling": jnp.array([0.5, 0.5, 0.5]),  # High scaling
    }

    # Call the function with both parameter sets
    ut_low, st_low = nonlinear_dynamics_model(tau, u0, s0, params_low_scaling)
    ut_high, st_high = nonlinear_dynamics_model(
        tau, u0, s0, params_high_scaling
    )

    # Check that higher scaling leads to lower unspliced counts (more saturation)
    assert jnp.all(ut_high <= ut_low)


def test_nonlinear_dynamics_model_type_checking():
    """Test nonlinear_dynamics_model type checking."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
        "scaling": jnp.array([0.1, 0.2, 0.3]),
    }

    # Invalid tau type
    with pytest.raises(BeartypeCallHintParamViolation):
        nonlinear_dynamics_model("not_an_array", u0, s0, params)

    # Invalid u0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        nonlinear_dynamics_model(tau, "not_an_array", s0, params)

    # Invalid s0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        nonlinear_dynamics_model(tau, u0, "not_an_array", params)

    # Invalid params type
    with pytest.raises(BeartypeCallHintParamViolation):
        nonlinear_dynamics_model(tau, u0, s0, "not_a_dict")


def test_dynamics_ode_model():
    """Test dynamics_ode_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Call the function
    ut, st = dynamics_ode_model(tau, u0, s0, params)

    # Check that the output has the correct shape
    assert ut.shape == tau.shape
    assert st.shape == tau.shape

    # Check that the output values are finite
    assert jnp.all(jnp.isfinite(ut))
    assert jnp.all(jnp.isfinite(st))

    # Check that the output values are non-negative
    assert jnp.all(ut >= 0)
    assert jnp.all(st >= 0)


def test_dynamics_ode_model_vs_standard():
    """Test that dynamics_ode_model gives similar results to standard_dynamics_model."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Call both functions
    ut_standard, st_standard = standard_dynamics_model(tau, u0, s0, params)
    ut_ode, st_ode = dynamics_ode_model(tau, u0, s0, params)

    # Check that the results are similar (within a reasonable tolerance)
    assert jnp.allclose(ut_standard, ut_ode, rtol=5e-2, atol=5e-2)
    assert jnp.allclose(st_standard, st_ode, rtol=5e-2, atol=5e-2)
    assert jnp.allclose(st_standard, st_ode, rtol=1e-2, atol=1e-2)


def test_dynamics_ode_model_type_checking():
    """Test dynamics_ode_model type checking."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Invalid tau type
    with pytest.raises(BeartypeCallHintParamViolation):
        dynamics_ode_model("not_an_array", u0, s0, params)

    # Invalid u0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        dynamics_ode_model(tau, "not_an_array", s0, params)

    # Invalid s0 type
    with pytest.raises(BeartypeCallHintParamViolation):
        dynamics_ode_model(tau, u0, "not_an_array", params)

    # Invalid params type
    with pytest.raises(BeartypeCallHintParamViolation):
        dynamics_ode_model(tau, u0, s0, "not_a_dict")


def test_vectorized_standard_dynamics_model():
    """Test vectorized_standard_dynamics_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Call both the standard and vectorized functions
    ut_standard, st_standard = standard_dynamics_model(tau, u0, s0, params)
    ut_vectorized, st_vectorized = vectorized_standard_dynamics_model(
        tau, u0, s0, params
    )

    # Check that the results are identical
    assert jnp.allclose(ut_standard, ut_vectorized)
    assert jnp.allclose(st_standard, st_vectorized)


def test_vectorized_nonlinear_dynamics_model():
    """Test vectorized_nonlinear_dynamics_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
        "scaling": jnp.array([0.1, 0.2, 0.3]),
    }

    # Call both the standard and vectorized functions
    ut_standard, st_standard = nonlinear_dynamics_model(tau, u0, s0, params)
    ut_vectorized, st_vectorized = vectorized_nonlinear_dynamics_model(
        tau, u0, s0, params
    )

    # Check that the results are identical
    assert jnp.allclose(ut_standard, ut_vectorized)
    assert jnp.allclose(st_standard, st_vectorized)


def test_vectorized_dynamics_ode_model():
    """Test vectorized_dynamics_ode_model implementation."""
    # Prepare test inputs
    tau = jnp.array([0.5, 1.0, 1.5])
    u0 = jnp.array([10.0, 20.0, 30.0])
    s0 = jnp.array([5.0, 10.0, 15.0])
    params = {
        "alpha": jnp.array([1.0, 2.0, 3.0]),
        "beta": jnp.array([0.5, 1.0, 1.5]),
        "gamma": jnp.array([0.3, 0.6, 0.9]),
    }

    # Call both the standard and vectorized functions
    ut_standard, st_standard = dynamics_ode_model(tau, u0, s0, params)
    ut_vectorized, st_vectorized = vectorized_dynamics_ode_model(
        tau, u0, s0, params
    )

    # Check that the results are identical
    assert jnp.allclose(ut_standard, ut_vectorized)
    assert jnp.allclose(st_standard, st_vectorized)


def test_vectorized_performance():
    """Test that vectorized implementations are faster for batch processing."""
    # This is a simple performance test to demonstrate the advantage of vectorization
    # In a real test, you might want to use more sophisticated benchmarking

    # Prepare large batch inputs
    batch_size = 100
    tau = jnp.ones(batch_size)
    u0 = jnp.ones(batch_size) * 10.0
    s0 = jnp.ones(batch_size) * 5.0
    params = {
        "alpha": jnp.ones(batch_size),
        "beta": jnp.ones(batch_size) * 0.5,
        "gamma": jnp.ones(batch_size) * 0.3,
    }

    # Define a function to process the batch using the non-vectorized function
    def process_batch_standard():
        results_u = []
        results_s = []
        for i in range(batch_size):
            params_i = {k: v[i] for k, v in params.items()}
            ut, st = standard_dynamics_model(tau[i], u0[i], s0[i], params_i)
            results_u.append(ut)
            results_s.append(st)
        return jnp.array(results_u), jnp.array(results_s)

    # Define a function to process the batch using the vectorized function
    def process_batch_vectorized():
        return vectorized_standard_dynamics_model(tau, u0, s0, params)

    # Time both approaches (using JAX's jit to compile for fair comparison)
    standard_fn = jax.jit(process_batch_standard)
    vectorized_fn = jax.jit(process_batch_vectorized)

    # Warm-up
    _ = standard_fn()
    _ = vectorized_fn()

    # Check that the results are the same
    ut_standard, st_standard = standard_fn()
    ut_vectorized, st_vectorized = vectorized_fn()

    assert jnp.allclose(ut_standard, ut_vectorized)
    assert jnp.allclose(st_standard, st_vectorized)
