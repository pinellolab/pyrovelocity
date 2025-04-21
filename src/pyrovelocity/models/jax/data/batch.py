"""
Batch processing utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for functional batch processing, including:

- create_batch_iterator: Create a functional batch iterator
- random_batch_indices: Generate random batch indices
- batch_data: Batch data using JAX's vmap
"""

from typing import Dict, Tuple, Optional, Any, List, Union, Iterator, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from beartype import beartype


@beartype
def random_batch_indices(
    key: jnp.ndarray,
    num_items: int,
    batch_size: int,
) -> jnp.ndarray:
    """Generate random batch indices.

    Args:
        key: JAX random key
        num_items: Number of items
        batch_size: Batch size

    Returns:
        JAX array of batch indices
    """
    # Ensure batch_size is not larger than num_items
    batch_size = min(batch_size, num_items)

    # Generate random indices without replacement
    indices = jax.random.choice(
        key, jnp.arange(num_items), shape=(batch_size,), replace=False
    )

    return indices


@beartype
def create_batch_iterator(
    data: Dict[str, jnp.ndarray],
    batch_size: int,
    key: jnp.ndarray,
    shuffle: bool = True,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """Create a functional batch iterator.

    Args:
        data: Dictionary of data arrays
        batch_size: Batch size
        key: JAX random key
        shuffle: Whether to shuffle the data

    Returns:
        Iterator of batched data dictionaries
    """
    # Get the number of items from the first array in the dictionary
    num_items = next(iter(data.values())).shape[0]

    # Create indices for batching
    if shuffle:
        # Generate a permutation of indices
        indices = jax.random.permutation(key, jnp.arange(num_items))
    else:
        # Use sequential indices
        indices = jnp.arange(num_items)

    # Calculate the number of batches
    num_batches = (num_items + batch_size - 1) // batch_size

    # Create and yield batches
    for i in range(num_batches):
        # Get the indices for this batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_items)
        batch_indices = indices[start_idx:end_idx]

        # Create the batch
        batch = batch_data(data, batch_indices)

        yield batch


@beartype
def batch_data(
    data: Dict[str, jnp.ndarray],
    indices: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Batch data using indices.

    Args:
        data: Dictionary of data arrays
        indices: Batch indices

    Returns:
        Dictionary of batched data arrays
    """
    # Create a new dictionary with batched data
    batched_data = {}

    # Apply indexing to each array in the dictionary
    for key, array in data.items():
        # Check if the first dimension of the array matches the number of items
        if array.shape[0] == indices.shape[0]:
            # If the array already has the same first dimension as indices,
            # it's likely already a batch, so we include it as is
            batched_data[key] = array
        else:
            # Otherwise, index into the first dimension
            batched_data[key] = array[indices]

    return batched_data


@beartype
def vmap_batch_function(
    fn: Callable,
    in_axes: Union[int, Dict[str, int]] = 0,
    out_axes: Union[int, Dict[str, int]] = 0,
) -> Callable:
    """Apply a function to each element in a batch using JAX's vmap.

    Args:
        fn: Function to apply
        in_axes: Input axes specification for vmap
        out_axes: Output axes specification for vmap

    Returns:
        Batched function
    """
    # Apply JAX's vmap to the function
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
