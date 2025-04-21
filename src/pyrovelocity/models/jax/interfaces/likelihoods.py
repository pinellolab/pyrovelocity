"""
Type definitions for likelihood functions in PyroVelocity JAX/NumPyro implementation.

This module defines the LikelihoodFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for likelihood functions
LikelihoodFunction = Callable[
    [
        Float[Array, "batch_size n_cells n_genes"],  # u_obs (observed unspliced)
        Float[Array, "batch_size n_cells n_genes"],  # s_obs (observed spliced)
        Float[Array, "batch_size n_cells n_genes"],  # u_logits (expected unspliced)
        Float[Array, "batch_size n_cells n_genes"],  # s_logits (expected spliced)
        Optional[Dict[str, Any]],                    # likelihood_params
    ],
    None                                            # PyroEffect (implicit)
]


def validate_likelihood_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the LikelihoodFunction interface.
    
    This function checks if the provided function has the correct signature
    to be used as a likelihood function in PyroVelocity. Likelihood functions
    are expected to use numpyro.sample inside their implementation.
    
    Args:
        fn: Function to validate
        
    Returns:
        True if the function conforms to the LikelihoodFunction interface
        
    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Likelihood function must be callable")
    
    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters
    
    # Check parameter count
    if len(params) != 5:
        raise TypeError(f"Likelihood function must have 5 parameters, got {len(params)}")
    
    # Check parameter names
    param_names = list(params.keys())
    expected_names = ["u_obs", "s_obs", "u_logits", "s_logits", "likelihood_params"]
    for i, name in enumerate(param_names):
        if name != expected_names[i]:
            raise TypeError(f"Parameter {i+1} should be named '{expected_names[i]}', got '{name}'")
    
    # Check if the fifth parameter is optional
    if params["likelihood_params"].default == inspect.Parameter.empty:
        raise TypeError("Parameter 'likelihood_params' should be optional")
    
    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation != None and return_annotation != type(None):
        raise TypeError("Likelihood function must have None as return type annotation")
    
    # Check if the function is bearable as a LikelihoodFunction
    if not is_bearable(fn, LikelihoodFunction):
        raise TypeError("Function does not conform to LikelihoodFunction interface")
    
    return True
