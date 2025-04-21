"""
Tests for PyroVelocity JAX/NumPyro adapter utilities.

This module contains tests for the adapter utilities, including:

- test_convert_tensor: Test conversion of PyTorch tensor to JAX array
- test_convert_array: Test conversion of JAX array to PyTorch tensor
- test_convert_parameters: Test conversion of parameters between frameworks
- test_convert_model_state: Test conversion of model state between frameworks
"""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from pyrovelocity.models.jax.adapters.torch_to_jax import (
    convert_tensor,
    convert_parameters,
    convert_model_state,
)
from pyrovelocity.models.jax.adapters.jax_to_torch import (
    convert_array,
    convert_parameters as convert_parameters_to_torch,
    convert_model_state as convert_model_state_to_torch,
)
from pyrovelocity.models.jax.core.state import VelocityModelState


def test_convert_tensor():
    """Test conversion of PyTorch tensor to JAX array."""
    # Create a PyTorch tensor
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Convert to JAX array
    jax_array = convert_tensor(torch_tensor)

    # Check type
    assert isinstance(jax_array, jnp.ndarray)

    # Check values
    np.testing.assert_allclose(jax_array, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_convert_tensor_with_gradients():
    """Test conversion of PyTorch tensor with gradients to JAX array."""
    # Create a PyTorch tensor with gradients
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Convert to JAX array
    jax_array = convert_tensor(torch_tensor)

    # Check type
    assert isinstance(jax_array, jnp.ndarray)

    # Check values
    np.testing.assert_allclose(jax_array, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_convert_array():
    """Test conversion of JAX array to PyTorch tensor."""
    # Create a JAX array
    jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    # Convert to PyTorch tensor
    torch_tensor = convert_array(jax_array)

    # Check type
    assert isinstance(torch_tensor, torch.Tensor)

    # Check values
    np.testing.assert_allclose(
        torch_tensor.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_convert_parameters():
    """Test conversion of parameters from PyTorch to JAX."""
    # Create PyTorch parameters
    torch_params = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": torch.tensor([0.1, 0.2]),
    }

    # Convert to JAX parameters
    jax_params = convert_parameters(torch_params)

    # Check types
    assert isinstance(jax_params, dict)
    assert all(isinstance(param, jnp.ndarray) for param in jax_params.values())

    # Check keys
    assert set(jax_params.keys()) == set(torch_params.keys())

    # Check values
    np.testing.assert_allclose(
        jax_params["weight"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    np.testing.assert_allclose(jax_params["bias"], np.array([0.1, 0.2]))


def test_convert_parameters_to_torch():
    """Test conversion of parameters from JAX to PyTorch."""
    # Create JAX parameters
    jax_params = {
        "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "bias": jnp.array([0.1, 0.2]),
    }

    # Convert to PyTorch parameters
    torch_params = convert_parameters_to_torch(jax_params)

    # Check types
    assert isinstance(torch_params, dict)
    assert all(
        isinstance(param, torch.Tensor) for param in torch_params.values()
    )

    # Check keys
    assert set(torch_params.keys()) == set(jax_params.keys())

    # Check values
    np.testing.assert_allclose(
        torch_params["weight"].numpy(), np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    np.testing.assert_allclose(
        torch_params["bias"].numpy(), np.array([0.1, 0.2])
    )


def test_convert_model_state():
    """Test conversion of model state from PyTorch to JAX."""
    # Create PyTorch model state
    torch_model_state = {
        "parameters": {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.1, 0.2]),
        },
        "dynamics_output": (
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        ),
        "observations": {
            "counts": torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
        },
        "distributions": None,
    }

    # Convert to JAX model state
    jax_model_state = convert_model_state(torch_model_state)

    # Check type
    assert isinstance(jax_model_state, VelocityModelState)

    # Check parameters
    assert isinstance(jax_model_state.parameters, dict)
    assert all(
        isinstance(param, jnp.ndarray)
        for param in jax_model_state.parameters.values()
    )
    np.testing.assert_allclose(
        jax_model_state.parameters["weight"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    np.testing.assert_allclose(
        jax_model_state.parameters["bias"], np.array([0.1, 0.2])
    )

    # Check dynamics output
    assert isinstance(jax_model_state.dynamics_output, tuple)
    assert len(jax_model_state.dynamics_output) == 2
    np.testing.assert_allclose(
        jax_model_state.dynamics_output[0], np.array([[5.0, 6.0], [7.0, 8.0]])
    )
    np.testing.assert_allclose(
        jax_model_state.dynamics_output[1],
        np.array([[9.0, 10.0], [11.0, 12.0]]),
    )

    # Check observations
    assert isinstance(jax_model_state.observations, dict)
    assert all(
        isinstance(obs, jnp.ndarray)
        for obs in jax_model_state.observations.values()
    )
    np.testing.assert_allclose(
        jax_model_state.observations["counts"],
        np.array([[13.0, 14.0], [15.0, 16.0]]),
    )

    # Check distributions
    assert jax_model_state.distributions is None


def test_convert_model_state_to_torch():
    """Test conversion of model state from JAX to PyTorch."""
    # Create JAX model state
    jax_model_state = VelocityModelState(
        parameters={
            "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": jnp.array([0.1, 0.2]),
        },
        dynamics_output=(
            jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            jnp.array([[9.0, 10.0], [11.0, 12.0]]),
        ),
        observations={
            "counts": jnp.array([[13.0, 14.0], [15.0, 16.0]]),
        },
        distributions=None,
    )

    # Convert to PyTorch model state
    torch_model_state = convert_model_state_to_torch(jax_model_state)

    # Check type
    assert isinstance(torch_model_state, dict)

    # Check parameters
    assert isinstance(torch_model_state["parameters"], dict)
    assert all(
        isinstance(param, torch.Tensor)
        for param in torch_model_state["parameters"].values()
    )
    np.testing.assert_allclose(
        torch_model_state["parameters"]["weight"].numpy(),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    np.testing.assert_allclose(
        torch_model_state["parameters"]["bias"].numpy(), np.array([0.1, 0.2])
    )

    # Check dynamics output
    assert isinstance(torch_model_state["dynamics_output"], tuple)
    assert len(torch_model_state["dynamics_output"]) == 2
    np.testing.assert_allclose(
        torch_model_state["dynamics_output"][0].numpy(),
        np.array([[5.0, 6.0], [7.0, 8.0]]),
    )
    np.testing.assert_allclose(
        torch_model_state["dynamics_output"][1].numpy(),
        np.array([[9.0, 10.0], [11.0, 12.0]]),
    )

    # Check observations
    assert isinstance(torch_model_state["observations"], dict)
    assert all(
        isinstance(obs, torch.Tensor)
        for obs in torch_model_state["observations"].values()
    )
    np.testing.assert_allclose(
        torch_model_state["observations"]["counts"].numpy(),
        np.array([[13.0, 14.0], [15.0, 16.0]]),
    )

    # Check distributions
    assert torch_model_state["distributions"] is None


def test_roundtrip_conversion():
    """Test roundtrip conversion of parameters between PyTorch and JAX."""
    # Create original PyTorch parameters
    original_torch_params = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": torch.tensor([0.1, 0.2]),
    }

    # Convert to JAX parameters
    jax_params = convert_parameters(original_torch_params)

    # Convert back to PyTorch parameters
    roundtrip_torch_params = convert_parameters_to_torch(jax_params)

    # Check types
    assert isinstance(roundtrip_torch_params, dict)
    assert all(
        isinstance(param, torch.Tensor)
        for param in roundtrip_torch_params.values()
    )

    # Check keys
    assert set(roundtrip_torch_params.keys()) == set(
        original_torch_params.keys()
    )

    # Check values
    for key in original_torch_params:
        np.testing.assert_allclose(
            roundtrip_torch_params[key].numpy(),
            original_torch_params[key].numpy(),
        )
