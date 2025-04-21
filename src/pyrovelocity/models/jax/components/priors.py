"""
Standard prior functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard prior functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry import register_prior


def lognormal_prior_function(
    key: jnp.ndarray,
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "n_genes"]]:
    """
    Log-normal prior function for RNA velocity parameters.
    
    This function samples RNA velocity parameters from log-normal distributions.
    
    Args:
        key: Random key
        num_genes: Number of genes
        prior_params: Dictionary of prior parameters
        
    Returns:
        Dictionary of sampled parameters
    """
    if prior_params is None:
        prior_params = {}
    
    alpha_loc = prior_params.get("alpha_loc", -0.5)
    alpha_scale = prior_params.get("alpha_scale", 1.0)
    beta_loc = prior_params.get("beta_loc", -0.5)
    beta_scale = prior_params.get("beta_scale", 1.0)
    gamma_loc = prior_params.get("gamma_loc", -0.5)
    gamma_scale = prior_params.get("gamma_scale", 1.0)
    
    key1, key2, key3 = jax.random.split(key, 3)
    
    alpha = jnp.exp(jax.random.normal(key1, (num_genes,)) * alpha_scale + alpha_loc)
    beta = jnp.exp(jax.random.normal(key2, (num_genes,)) * beta_scale + beta_loc)
    gamma = jnp.exp(jax.random.normal(key3, (num_genes,)) * gamma_scale + gamma_loc)
    
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def informative_prior_function(
    key: jnp.ndarray,
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "n_genes"]]:
    """
    Informative prior function for RNA velocity parameters.
    
    This function samples RNA velocity parameters from informative distributions
    based on prior knowledge.
    
    Args:
        key: Random key
        num_genes: Number of genes
        prior_params: Dictionary of prior parameters
        
    Returns:
        Dictionary of sampled parameters
    """
    if prior_params is None:
        prior_params = {}
    
    # Default values based on typical RNA velocity parameters
    alpha_loc = prior_params.get("alpha_loc", 0.0)
    alpha_scale = prior_params.get("alpha_scale", 0.5)
    beta_loc = prior_params.get("beta_loc", -1.0)
    beta_scale = prior_params.get("beta_scale", 0.5)
    gamma_loc = prior_params.get("gamma_loc", -1.5)
    gamma_scale = prior_params.get("gamma_scale", 0.5)
    
    key1, key2, key3 = jax.random.split(key, 3)
    
    alpha = jnp.exp(jax.random.normal(key1, (num_genes,)) * alpha_scale + alpha_loc)
    beta = jnp.exp(jax.random.normal(key2, (num_genes,)) * beta_scale + beta_loc)
    gamma = jnp.exp(jax.random.normal(key3, (num_genes,)) * gamma_scale + gamma_loc)
    
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def register_standard_priors():
    """Register standard prior functions."""
    register_prior("lognormal", lognormal_prior_function)
    register_prior("informative", informative_prior_function)
