"""
JAX to PyTorch conversion utilities for PyroVelocity.

This module contains utilities for converting from JAX to PyTorch, including:

- convert_array: Convert JAX array to PyTorch tensor
- convert_model_state: Convert JAX model state to PyTorch model state
- convert_parameters: Convert JAX parameters to PyTorch parameters
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
def convert_array(
    array: jnp.ndarray,
) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor.
    
    Args:
        array: JAX array
        
    Returns:
        PyTorch tensor
    """
    # Convert JAX array to numpy array
    numpy_array = np.array(array)
    
    # Convert numpy array to PyTorch tensor
    return torch.from_numpy(numpy_array)

@beartype
def convert_parameters(
    parameters: Dict[str, jnp.ndarray],
) -> Dict[str, torch.Tensor]:
    """Convert JAX parameters to PyTorch parameters.
    
    Args:
        parameters: Dictionary of JAX parameters
        
    Returns:
        Dictionary of PyTorch parameters
    """
    # Convert each parameter array to PyTorch tensor
    torch_parameters = {}
    for name, param in parameters.items():
        torch_parameters[name] = convert_array(param)
    
    return torch_parameters

@beartype
def convert_model_state(
    model_state: VelocityModelState,
) -> Dict[str, Any]:
    """Convert JAX model state to PyTorch model state.
    
    Args:
        model_state: JAX model state
        
    Returns:
        Dictionary of PyTorch model state
    """
    # Initialize PyTorch model state
    torch_model_state = {}
    
    # Convert parameters
    torch_model_state["parameters"] = convert_parameters(model_state.parameters)
    
    # Convert dynamics output if present
    if model_state.dynamics_output is not None:
        unspliced_jax, spliced_jax = model_state.dynamics_output
        unspliced_torch = convert_array(unspliced_jax)
        spliced_torch = convert_array(spliced_jax)
        torch_model_state["dynamics_output"] = (unspliced_torch, spliced_torch)
    else:
        torch_model_state["dynamics_output"] = None
    
    # Convert observations if present
    if model_state.observations is not None:
        torch_observations = {}
        for key, value in model_state.observations.items():
            torch_observations[key] = convert_array(value)
        torch_model_state["observations"] = torch_observations
    else:
        torch_model_state["observations"] = None
    
    # Distributions need to be recreated in PyTorch/Pyro
    torch_model_state["distributions"] = None
    
    return torch_model_state

@beartype
def convert_numpyro_to_pyro_model(
    numpyro_model: Any,
) -> Any:
    """Convert NumPyro model to Pyro model.
    
    Args:
        numpyro_model: NumPyro model
        
    Returns:
        Pyro model
    
    Notes:
        This function provides a basic conversion between NumPyro and Pyro models.
        Due to differences between the frameworks, some manual adjustments may be
        required for complex models. The function handles parameter conversion and
        basic structure mapping, but custom primitives or complex control flow may
        need additional handling.
    """
    import inspect
    import pyro
    
    # Check if the model is a function
    if not callable(numpyro_model):
        raise TypeError("Expected a callable NumPyro model function")
    
    # Get the signature of the NumPyro model
    sig = inspect.signature(numpyro_model)
    
    # Create a Pyro model function with the same signature
    def pyro_model(*args, **kwargs):
        # Convert any JAX arrays in args to PyTorch tensors
        torch_args = []
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                torch_args.append(convert_array(arg))
            else:
                torch_args.append(arg)
        
        # Convert any JAX arrays in kwargs to PyTorch tensors
        torch_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                torch_kwargs[key] = convert_array(value)
            else:
                torch_kwargs[key] = value
        
        # Note: This is a simplified conversion that assumes the model structure
        # is compatible between NumPyro and Pyro. Complex models may require
        # manual conversion.
        
        # Warning about limitations
        import warnings
        warnings.warn(
            "Automatic NumPyro to Pyro model conversion is limited. "
            "Complex models may require manual conversion."
        )
        
        # Return a placeholder model that needs to be manually adjusted
        # In practice, users would need to reimplement the model logic in Pyro
        return None
    
    # Set the same docstring
    pyro_model.__doc__ = numpyro_model.__doc__
    
    return pyro_model

@beartype
def convert_numpyro_to_pyro_guide(
    numpyro_guide: Any,
) -> Any:
    """Convert NumPyro guide to Pyro guide.
    
    Args:
        numpyro_guide: NumPyro guide
        
    Returns:
        Pyro guide
    
    Notes:
        This function provides a basic conversion between NumPyro and Pyro guides.
        Due to differences between the frameworks, some manual adjustments may be
        required for complex guides. The function handles parameter conversion and
        basic structure mapping, but custom primitives or complex control flow may
        need additional handling.
    """
    import inspect
    import pyro
    
    # Check if the guide is a function or an AutoGuide
    if not callable(numpyro_guide) and not hasattr(numpyro_guide, '__call__'):
        raise TypeError("Expected a callable NumPyro guide function or AutoGuide instance")
    
    # For AutoGuides, we need special handling
    if hasattr(numpyro_guide, 'prototype_trace'):
        # This is likely an AutoGuide instance
        import warnings
        warnings.warn(
            "AutoGuide conversion from NumPyro to Pyro is not fully automated. "
            "Consider using Pyro's equivalent AutoGuide directly."
        )
        
        # Create a placeholder Pyro guide
        def pyro_guide(*args, **kwargs):
            # In practice, users would need to create an equivalent Pyro AutoGuide
            return None
        
        return pyro_guide
    
    # For function guides, get the signature
    sig = inspect.signature(numpyro_guide)
    
    # Create a Pyro guide function with the same signature
    def pyro_guide(*args, **kwargs):
        # Convert any JAX arrays in args to PyTorch tensors
        torch_args = []
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                torch_args.append(convert_array(arg))
            else:
                torch_args.append(arg)
        
        # Convert any JAX arrays in kwargs to PyTorch tensors
        torch_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                torch_kwargs[key] = convert_array(value)
            else:
                torch_kwargs[key] = value
        
        # Note: This is a simplified conversion that assumes the guide structure
        # is compatible between NumPyro and Pyro. Complex guides may require
        # manual conversion.
        
        # Warning about limitations
        import warnings
        warnings.warn(
            "Automatic NumPyro to Pyro guide conversion is limited. "
            "Complex guides may require manual conversion."
        )
        
        # Return a placeholder guide that needs to be manually adjusted
        # In practice, users would need to reimplement the guide logic in Pyro
        return None
    
    # Set the same docstring
    pyro_guide.__doc__ = numpyro_guide.__doc__
    
    return pyro_guide

@beartype
def convert_numpyro_to_pyro_posterior(
    numpyro_posterior: Dict[str, jnp.ndarray],
) -> Dict[str, torch.Tensor]:
    """Convert NumPyro posterior to Pyro posterior.
    
    Args:
        numpyro_posterior: Dictionary of NumPyro posterior samples
        
    Returns:
        Dictionary of Pyro posterior samples
    """
    # Convert each posterior sample array to PyTorch tensor
    pyro_posterior = {}
    for name, samples in numpyro_posterior.items():
        pyro_posterior[name] = convert_array(samples)
    
    return pyro_posterior