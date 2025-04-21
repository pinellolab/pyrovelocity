"""
Type definitions for guide factory functions in PyroVelocity JAX/NumPyro implementation.

This module defines the GuideFactoryFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for guide factory functions
GuideFactoryFunction = Callable[
    [
        Callable,                                    # model function
        Optional[Dict[str, Any]],                    # guide_params
    ],
    Callable                                         # guide function
]


def validate_guide_factory_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the GuideFactoryFunction interface.
    
    This function checks if the provided function has the correct signature
    and return type to be used as a guide factory function in PyroVelocity.
    
    Args:
        fn: Function to validate
        
    Returns:
        True if the function conforms to the GuideFactoryFunction interface
        
    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Guide factory function must be callable")
    
    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters
    
    # Check parameter count
    if len(params) != 2:
        raise TypeError(f"Guide factory function must have 2 parameters, got {len(params)}")
    
    # Check parameter names
    param_names = list(params.keys())
    expected_names = ["model", "guide_params"]
    for i, name in enumerate(param_names):
        if name != expected_names[i]:
            raise TypeError(f"Parameter {i+1} should be named '{expected_names[i]}', got '{name}'")
    
    # Check if the second parameter is optional
    if params["guide_params"].default == inspect.Parameter.empty:
        raise TypeError("Parameter 'guide_params' should be optional")
    
    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation == inspect.Signature.empty:
        raise TypeError("Guide factory function must have a return type annotation")
    
    # Check if the function is bearable as a GuideFactoryFunction
    if not is_bearable(fn, GuideFactoryFunction):
        raise TypeError("Function does not conform to GuideFactoryFunction interface")
    
    return True
