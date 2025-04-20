"""
Likelihood model implementations for RNA counts.

This module contains likelihood model implementations for RNA counts, including:

- poisson_likelihood: Poisson likelihood for RNA counts
- negative_binomial_likelihood: Negative binomial likelihood for RNA counts
"""

from typing import Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from beartype import beartype

@beartype
def poisson_likelihood(
    ut: Float[Array, "..."],
    st: Float[Array, "..."],
    scaling_params: Dict[str, Float[Array, "..."]]
) -> Tuple[dist.Distribution, dist.Distribution]:
    """Poisson likelihood for RNA counts.
    
    Args:
        ut: Unspliced RNA counts
        st: Spliced RNA counts
        scaling_params: Dictionary of scaling parameters
        
    Returns:
        Tuple of (unspliced_distribution, spliced_distribution)
    """
    # Extract scaling parameters
    u_log_library = scaling_params.get("u_log_library", jnp.zeros_like(jnp.sum(ut, axis=-1)))
    s_log_library = scaling_params.get("s_log_library", jnp.zeros_like(jnp.sum(st, axis=-1)))
    
    # Reshape log library sizes for broadcasting
    # If ut has shape (batch, genes), u_log_library should have shape (batch,)
    # We need to reshape to (batch, 1) for proper broadcasting
    u_log_library_expanded = u_log_library[:, jnp.newaxis]
    s_log_library_expanded = s_log_library[:, jnp.newaxis]
    
    # Apply library size scaling
    u_rate = jnp.exp(u_log_library_expanded) * ut
    s_rate = jnp.exp(s_log_library_expanded) * st
    
    # Create Poisson distributions
    u_dist = dist.Poisson(rate=u_rate)
    s_dist = dist.Poisson(rate=s_rate)
    
    return u_dist, s_dist

@beartype
def negative_binomial_likelihood(
    ut: Float[Array, "..."],
    st: Float[Array, "..."],
    scaling_params: Dict[str, Float[Array, "..."]]
) -> Tuple[dist.Distribution, dist.Distribution]:
    """Negative binomial likelihood for RNA counts.
    
    Args:
        ut: Unspliced RNA counts
        st: Spliced RNA counts
        scaling_params: Dictionary of scaling parameters
        
    Returns:
        Tuple of (unspliced_distribution, spliced_distribution)
    """
    # Extract scaling parameters
    u_log_library = scaling_params.get("u_log_library", jnp.zeros_like(jnp.sum(ut, axis=-1)))
    s_log_library = scaling_params.get("s_log_library", jnp.zeros_like(jnp.sum(st, axis=-1)))
    u_dispersion = scaling_params.get("u_dispersion", jnp.ones_like(ut))
    s_dispersion = scaling_params.get("s_dispersion", jnp.ones_like(st))
    
    # Reshape log library sizes for broadcasting
    # If ut has shape (batch, genes), u_log_library should have shape (batch,)
    # We need to reshape to (batch, 1) for proper broadcasting
    u_log_library_expanded = u_log_library[:, jnp.newaxis]
    s_log_library_expanded = s_log_library[:, jnp.newaxis]
    
    # Apply library size scaling
    u_rate = jnp.exp(u_log_library_expanded) * ut
    s_rate = jnp.exp(s_log_library_expanded) * st
    
    # Ensure positive dispersion values
    u_dispersion = jnp.maximum(u_dispersion, 1e-6)
    s_dispersion = jnp.maximum(s_dispersion, 1e-6)
    
    # Create Negative Binomial distributions
    # Using the parameterization with mean and dispersion
    u_dist = dist.GammaPoisson(concentration=1.0/u_dispersion, rate=1.0/(u_dispersion * u_rate))
    s_dist = dist.GammaPoisson(concentration=1.0/s_dispersion, rate=1.0/(s_dispersion * s_rate))
    
    return u_dist, s_dist

@beartype
def create_likelihood(
    likelihood_type: str = "poisson"
) -> Callable:
    """Create a likelihood function based on the specified type.
    
    Args:
        likelihood_type: Type of likelihood ("poisson" or "negative_binomial")
        
    Returns:
        Likelihood function
    """
    if likelihood_type == "poisson":
        return poisson_likelihood
    elif likelihood_type == "negative_binomial":
        return negative_binomial_likelihood
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood_type}")