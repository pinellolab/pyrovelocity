"""
Type definitions for observation functions in PyroVelocity JAX/NumPyro implementation.

This module defines the ObservationFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for observation functions
ObservationFunction = Callable[
    [
        Float[Array, "batch_size n_cells n_genes"],  # u_obs (raw unspliced)
        Float[Array, "batch_size n_cells n_genes"],  # s_obs (raw spliced)
        Optional[Dict[str, Any]],                    # observation_params
    ],
    Tuple[
        Float[Array, "batch_size n_cells n_genes"],  # u_transformed
        Float[Array, "batch_size n_cells n_genes"],  # s_transformed
    ]
]


def validate_observation_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the ObservationFunction interface.
    
    This function checks if the provided function has the correct signature
    and return type to be used as an observation function in PyroVelocity.
    
    Args:
        fn: Function to validate
        
    Returns:
        True if the function conforms to the ObservationFunction interface
        
    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Observation function must be callable")
    
    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters
    
    # Check parameter count
    if len(params) != 3:
        raise TypeError(f"Observation function must have 3 parameters, got {len(params)}")
    
    # Check parameter names
    param_names = list(params.keys())
    expected_names = ["u_obs", "s_obs", "observation_params"]
    for i, name in enumerate(param_names):
        if name != expected_names[i]:
            raise TypeError(f"Parameter {i+1} should be named '{expected_names[i]}', got '{name}'")
    
    # Check if the third parameter is optional
    if params["observation_params"].default == inspect.Parameter.empty:
        raise TypeError("Parameter 'observation_params' should be optional")
    
    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation == inspect.Signature.empty:
        raise TypeError("Observation function must have a return type annotation")
    
    # Check if the function is bearable as an ObservationFunction
    if not is_bearable(fn, ObservationFunction):
        raise TypeError("Function does not conform to ObservationFunction interface")
    
    return True
