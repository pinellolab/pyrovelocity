"""
Standard observation functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard observation functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry import register_observation


def standard_observation_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    observation_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """
    Standard observation function for RNA velocity.

    This function applies standard preprocessing to observed RNA counts.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        observation_params: Dictionary of observation parameters

    Returns:
        Tuple of (transformed unspliced counts, transformed spliced counts)
    """
    if observation_params is None:
        observation_params = {}

    # Apply log1p transformation if specified
    log1p = observation_params.get("log1p", False)
    if log1p:
        u_transformed = jnp.log1p(u_obs)
        s_transformed = jnp.log1p(s_obs)
    else:
        u_transformed = u_obs
        s_transformed = s_obs

    # Apply normalization if specified
    normalize = observation_params.get("normalize", False)
    if normalize:
        # Compute size factors
        u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
        s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)

        # Normalize
        u_transformed = u_transformed / (u_size_factor + 1e-6)
        s_transformed = s_transformed / (s_size_factor + 1e-6)

    return u_transformed, s_transformed


def normalized_observation_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    observation_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """
    Normalized observation function for RNA velocity.

    This function applies normalization to observed RNA counts.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        observation_params: Dictionary of observation parameters

    Returns:
        Tuple of (normalized unspliced counts, normalized spliced counts)
    """
    if observation_params is None:
        observation_params = {}

    # Compute size factors
    u_size_factor = jnp.sum(u_obs, axis=-1, keepdims=True)
    s_size_factor = jnp.sum(s_obs, axis=-1, keepdims=True)

    # Normalize
    u_normalized = u_obs / (u_size_factor + 1e-6)
    s_normalized = s_obs / (s_size_factor + 1e-6)

    # Apply scaling if specified
    scale_factor = observation_params.get("scale_factor", 1.0)
    u_normalized = u_normalized * scale_factor
    s_normalized = s_normalized * scale_factor

    return u_normalized, s_normalized


def register_standard_observations():
    """Register standard observation functions."""
    register_observation("standard", standard_observation_function)
    register_observation("normalized", normalized_observation_function)
