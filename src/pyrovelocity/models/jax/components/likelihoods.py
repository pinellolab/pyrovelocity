"""
Standard likelihood functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard likelihood functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry import register_likelihood


def poisson_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Poisson likelihood function for RNA velocity.

    This function samples observed RNA counts from Poisson distributions.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        u_logits: Expected unspliced counts
        s_logits: Expected spliced counts
        likelihood_params: Dictionary of likelihood parameters
    """
    if likelihood_params is None:
        likelihood_params = {}

    # Get library size scaling
    u_log_library = likelihood_params.get("u_log_library")
    s_log_library = likelihood_params.get("s_log_library")

    # Apply library size scaling if provided
    if u_log_library is not None:
        u_logits = u_logits * jnp.exp(u_log_library[:, :, jnp.newaxis])
    if s_log_library is not None:
        s_logits = s_logits * jnp.exp(s_log_library[:, :, jnp.newaxis])

    # Ensure positive values for Poisson distribution
    u_logits = jnp.maximum(u_logits, 1e-6)
    s_logits = jnp.maximum(s_logits, 1e-6)

    # Sample from Poisson distribution
    numpyro.sample("u", dist.Poisson(u_logits).to_event(2), obs=u_obs)
    numpyro.sample("s", dist.Poisson(s_logits).to_event(2), obs=s_obs)


def negative_binomial_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Negative binomial likelihood function for RNA velocity.

    This function samples observed RNA counts from negative binomial distributions.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        u_logits: Expected unspliced counts
        s_logits: Expected spliced counts
        likelihood_params: Dictionary of likelihood parameters
    """
    if likelihood_params is None:
        likelihood_params = {}

    # Get library size scaling
    u_log_library = likelihood_params.get("u_log_library")
    s_log_library = likelihood_params.get("s_log_library")

    # Apply library size scaling if provided
    if u_log_library is not None:
        u_logits = u_logits * jnp.exp(u_log_library[:, :, jnp.newaxis])
    if s_log_library is not None:
        s_logits = s_logits * jnp.exp(s_log_library[:, :, jnp.newaxis])

    # Get dispersion parameters
    u_dispersion = likelihood_params.get("u_dispersion", 1.0)
    s_dispersion = likelihood_params.get("s_dispersion", 1.0)

    # Ensure positive values for NegativeBinomial distribution
    u_logits = jnp.maximum(u_logits, 1e-6)
    s_logits = jnp.maximum(s_logits, 1e-6)

    # Sample from negative binomial distribution
    numpyro.sample(
        "u",
        dist.NegativeBinomial2(u_logits, u_dispersion).to_event(2),
        obs=u_obs,
    )
    numpyro.sample(
        "s",
        dist.NegativeBinomial2(s_logits, s_dispersion).to_event(2),
        obs=s_obs,
    )


def register_standard_likelihoods():
    """Register standard likelihood functions."""
    register_likelihood("poisson", poisson_likelihood_function)
    register_likelihood("negative_binomial", negative_binomial_likelihood_function)
