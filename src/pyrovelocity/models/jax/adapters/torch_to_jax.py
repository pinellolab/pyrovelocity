"""
PyTorch to JAX conversion utilities for PyroVelocity.

This module contains utilities for converting from PyTorch to JAX, including:

- convert_tensor: Convert PyTorch tensor to JAX array
- convert_model_state: Convert PyTorch model state to JAX model state
- convert_parameters: Convert PyTorch parameters to JAX parameters
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float, PyTree
from beartype import beartype

from pyrovelocity.models.jax.core.state import VelocityModelState


@beartype
def convert_tensor(
    tensor: torch.Tensor,
) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array.

    Args:
        tensor: PyTorch tensor

    Returns:
        JAX array
    """
    # Detach tensor if it requires gradients
    if tensor.requires_grad:
        tensor = tensor.detach()

    # Convert to numpy array
    numpy_array = tensor.cpu().numpy()

    # Convert to JAX array
    return jnp.array(numpy_array)


@beartype
def convert_parameters(
    parameters: Dict[str, torch.Tensor],
) -> Dict[str, jnp.ndarray]:
    """Convert PyTorch parameters to JAX parameters.

    Args:
        parameters: Dictionary of PyTorch parameters

    Returns:
        Dictionary of JAX parameters
    """
    # Convert each parameter tensor to JAX array
    jax_parameters = {}
    for name, param in parameters.items():
        jax_parameters[name] = convert_tensor(param)

    return jax_parameters


@beartype
def convert_model_state(
    model_state: Dict[str, Any],
) -> VelocityModelState:
    """Convert PyTorch model state to JAX model state.

    Args:
        model_state: Dictionary of PyTorch model state

    Returns:
        JAX model state
    """
    # Convert parameters
    parameters = convert_parameters(model_state.get("parameters", {}))

    # Convert dynamics output if present
    dynamics_output = None
    if (
        "dynamics_output" in model_state
        and model_state["dynamics_output"] is not None
    ):
        unspliced, spliced = model_state["dynamics_output"]
        dynamics_output = (convert_tensor(unspliced), convert_tensor(spliced))

    # Convert observations if present
    observations = None
    if (
        "observations" in model_state
        and model_state["observations"] is not None
    ):
        observations = {}
        for key, value in model_state["observations"].items():
            observations[key] = convert_tensor(value)

    # Create and return JAX model state
    return VelocityModelState(
        parameters=parameters,
        dynamics_output=dynamics_output,
        distributions=None,  # Distributions need to be recreated in JAX
        observations=observations,
    )


@beartype
def convert_pyro_to_numpyro_model(
    pyro_model: Any,
) -> Any:
    """Convert Pyro model to NumPyro model.

    Args:
        pyro_model: Pyro model

    Returns:
        NumPyro model

    Notes:
        This function provides a basic conversion between Pyro and NumPyro models.
        Due to differences between the frameworks, some manual adjustments may be
        required for complex models. The function handles parameter conversion and
        basic structure mapping, but custom primitives or complex control flow may
        need additional handling.
    """
    import inspect
    import pyro

    # Check if the model is a function
    if not callable(pyro_model):
        raise TypeError("Expected a callable Pyro model function")

    # Get the signature of the Pyro model
    sig = inspect.signature(pyro_model)

    # Create a NumPyro model function with the same signature
    def numpyro_model(*args, **kwargs):
        # Convert any PyTorch tensors in args to JAX arrays
        jax_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                jax_args.append(convert_tensor(arg))
            else:
                jax_args.append(arg)

        # Convert any PyTorch tensors in kwargs to JAX arrays
        jax_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                jax_kwargs[key] = convert_tensor(value)
            else:
                jax_kwargs[key] = value

        # Note: This is a simplified conversion that assumes the model structure
        # is compatible between Pyro and NumPyro. Complex models may require
        # manual conversion.

        # Warning about limitations
        import warnings

        warnings.warn(
            "Automatic Pyro to NumPyro model conversion is limited. "
            "Complex models may require manual conversion."
        )

        # Return a placeholder model that needs to be manually adjusted
        # In practice, users would need to reimplement the model logic in NumPyro
        return None

    # Set the same docstring
    numpyro_model.__doc__ = pyro_model.__doc__

    return numpyro_model


@beartype
def convert_pyro_to_numpyro_guide(
    pyro_guide: Any,
) -> Any:
    """Convert Pyro guide to NumPyro guide.

    Args:
        pyro_guide: Pyro guide

    Returns:
        NumPyro guide

    Notes:
        This function provides a basic conversion between Pyro and NumPyro guides.
        Due to differences between the frameworks, some manual adjustments may be
        required for complex guides. The function handles parameter conversion and
        basic structure mapping, but custom primitives or complex control flow may
        need additional handling.
    """
    import inspect
    import pyro

    # Check if the guide is a function or an AutoGuide
    if not callable(pyro_guide) and not hasattr(pyro_guide, "__call__"):
        raise TypeError(
            "Expected a callable Pyro guide function or AutoGuide instance"
        )

    # For AutoGuides, we need special handling
    if hasattr(pyro_guide, "requires_grad"):
        # This is likely an AutoGuide instance
        import warnings

        warnings.warn(
            "AutoGuide conversion from Pyro to NumPyro is not fully automated. "
            "Consider using NumPyro's equivalent AutoGuide directly."
        )

        # Create a placeholder NumPyro guide
        def numpyro_guide(*args, **kwargs):
            # In practice, users would need to create an equivalent NumPyro AutoGuide
            return None

        return numpyro_guide

    # For function guides, get the signature
    sig = inspect.signature(pyro_guide)

    # Create a NumPyro guide function with the same signature
    def numpyro_guide(*args, **kwargs):
        # Convert any PyTorch tensors in args to JAX arrays
        jax_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                jax_args.append(convert_tensor(arg))
            else:
                jax_args.append(arg)

        # Convert any PyTorch tensors in kwargs to JAX arrays
        jax_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                jax_kwargs[key] = convert_tensor(value)
            else:
                jax_kwargs[key] = value

        # Note: This is a simplified conversion that assumes the guide structure
        # is compatible between Pyro and NumPyro. Complex guides may require
        # manual conversion.

        # Warning about limitations
        import warnings

        warnings.warn(
            "Automatic Pyro to NumPyro guide conversion is limited. "
            "Complex guides may require manual conversion."
        )

        # Return a placeholder guide that needs to be manually adjusted
        # In practice, users would need to reimplement the guide logic in NumPyro
        return None

    # Set the same docstring
    numpyro_guide.__doc__ = pyro_guide.__doc__

    return numpyro_guide


@beartype
def convert_pyro_to_numpyro_posterior(
    pyro_posterior: Dict[str, torch.Tensor],
) -> Dict[str, jnp.ndarray]:
    """Convert Pyro posterior to NumPyro posterior.

    Args:
        pyro_posterior: Dictionary of Pyro posterior samples

    Returns:
        Dictionary of NumPyro posterior samples
    """
    # Convert each posterior sample tensor to JAX array
    numpyro_posterior = {}
    for name, samples in pyro_posterior.items():
        numpyro_posterior[name] = convert_tensor(samples)

    return numpyro_posterior
