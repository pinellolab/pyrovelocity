"""
Prior distribution implementations for RNA velocity parameters.

This module contains prior distribution implementations for RNA velocity
parameters, including:

- lognormal_prior: Log-normal prior for RNA velocity parameters
- informative_prior: Informative prior for RNA velocity parameters
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, ArrayLike, Float


@beartype
def lognormal_prior(
    key: jnp.ndarray,
    shape: Tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
) -> Dict[str, Float[Array, "..."]]:
    """Sample parameters from log-normal priors.

    Args:
        key: JAX random key
        shape: Shape of the parameters
        loc: Location parameter for log-normal distribution
        scale: Scale parameter for log-normal distribution

    Returns:
        Dictionary of parameter samples
    """
    # Split the key for different parameters
    key_alpha, key_beta, key_gamma = jax.random.split(key, 3)

    # Sample from log-normal distributions
    alpha = jnp.exp(jax.random.normal(key_alpha, shape) * scale + loc)
    beta = jnp.exp(jax.random.normal(key_beta, shape) * scale + loc)
    gamma = jnp.exp(jax.random.normal(key_gamma, shape) * scale + loc)

    # Return dictionary of parameters
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


@beartype
def informative_prior(
    key: jnp.ndarray, shape: Tuple[int, ...], prior_params: Dict[str, Any]
) -> Dict[str, Float[Array, "..."]]:
    """Sample parameters from informative priors.

    Args:
        key: JAX random key
        shape: Shape of the parameters
        prior_params: Dictionary of prior parameters

    Returns:
        Dictionary of parameter samples
    """
    # Split the key for different parameters
    key_alpha, key_beta, key_gamma = jax.random.split(key, 3)

    # Extract prior parameters with defaults
    alpha_loc = prior_params.get("alpha_loc", 0.0)
    alpha_scale = prior_params.get("alpha_scale", 1.0)
    beta_loc = prior_params.get("beta_loc", 0.0)
    beta_scale = prior_params.get("beta_scale", 1.0)
    gamma_loc = prior_params.get("gamma_loc", 0.0)
    gamma_scale = prior_params.get("gamma_scale", 1.0)

    # Sample from log-normal distributions with parameter-specific settings
    alpha = jnp.exp(
        jax.random.normal(key_alpha, shape) * alpha_scale + alpha_loc
    )
    beta = jnp.exp(jax.random.normal(key_beta, shape) * beta_scale + beta_loc)
    gamma = jnp.exp(
        jax.random.normal(key_gamma, shape) * gamma_scale + gamma_loc
    )

    # Return dictionary of parameters
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


@beartype
def sample_prior_parameters(
    key: Optional[ArrayLike],
    num_genes: int,
    prior_type: str = "lognormal",
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "gene"]]:
    """Sample prior parameters for RNA velocity model.

    Args:
        key: JAX random key
        num_genes: Number of genes
        prior_type: Type of prior ("lognormal" or "informative")
        prior_params: Optional dictionary of prior parameters

    Returns:
        Dictionary of parameter samples
    """
    # Define shape for gene-specific parameters
    shape = (num_genes,)

    if key is None:
        key = jax.random.PRNGKey(0)

    # Use default prior parameters if not provided
    if prior_params is None:
        prior_params = {}

    # Sample parameters based on prior type
    if prior_type == "lognormal":
        # Default log-normal prior
        loc = prior_params.get("loc", 0.0)
        scale = prior_params.get("scale", 1.0)
        return lognormal_prior(key, shape, loc, scale)
    elif prior_type == "informative":
        # Informative prior with parameter-specific settings
        return informative_prior(key, shape, prior_params)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")
