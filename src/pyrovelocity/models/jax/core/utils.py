"""
Utility functions for PyroVelocity JAX/NumPyro implementation.

This module contains utility functions for JAX, including random state management
and type definitions.
"""

from typing import Dict, Tuple, Optional, Any, Callable, Union
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jaxtyping import Array, Float, Int, Bool, PyTree, ArrayLike
from beartype import beartype

# Type definitions
KeyArray = jnp.ndarray  # JAX random key type

@beartype
def create_key(seed: int) -> KeyArray:
    """Create a JAX random key from a seed.
    
    Args:
        seed: Integer seed for random number generation
        
    Returns:
        JAX random key
    """
    return jax.random.PRNGKey(seed)

@beartype
def split_key(key: KeyArray, num: int = 2) -> Tuple[KeyArray, ...]:
    """Split a JAX random key into multiple keys.
    
    Args:
        key: JAX random key
        num: Number of keys to split into
        
    Returns:
        Tuple of JAX random keys
    """
    return tuple(jax.random.split(key, num))

@beartype
def set_platform_device(platform: str = "cpu") -> None:
    """Set the JAX platform device.
    
    Args:
        platform: Platform device ("cpu", "gpu", or "tpu")
    """
    jax.config.update("jax_platform_name", platform)

@beartype
def enable_x64() -> None:
    """Enable 64-bit floating point precision in JAX."""
    jax.config.update("jax_enable_x64", True)

@beartype
def disable_x64() -> None:
    """Disable 64-bit floating point precision in JAX."""
    jax.config.update("jax_enable_x64", False)

@beartype
def get_device_count() -> int:
    """Get the number of available devices.
    
    Returns:
        Number of available devices
    """
    return jax.device_count()

@beartype
def get_devices() -> list:
    """Get the list of available devices.
    
    Returns:
        List of available devices
    """
    return jax.devices()

@beartype
def check_array_shape(array: ArrayLike, expected_shape: Tuple[int, ...]) -> bool:
    """Check if an array has the expected shape.
    
    Args:
        array: JAX array
        expected_shape: Expected shape
        
    Returns:
        True if the array has the expected shape, False otherwise
    """
    return array.shape == expected_shape

@beartype
def check_array_dtype(array: ArrayLike, expected_dtype: Any) -> bool:
    """Check if an array has the expected dtype.
    
    Args:
        array: JAX array
        expected_dtype: Expected dtype
        
    Returns:
        True if the array has the expected dtype, False otherwise
    """
    return array.dtype == expected_dtype

@beartype
def ensure_array(array: Union[ArrayLike, list, tuple]) -> jnp.ndarray:
    """Ensure that the input is a JAX array.
    
    Args:
        array: Input array or array-like object
        
    Returns:
        JAX array
    """
    # Handle different types of inputs
    if isinstance(array, (list, tuple)):
        return jnp.array(array)
    # Check if it's a numpy array (without using type annotations)
    if 'numpy' in str(type(array)):
        return jnp.asarray(array)
    # Default case for JAX arrays
    return jnp.asarray(array)