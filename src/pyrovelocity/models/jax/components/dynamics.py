"""
Standard dynamics functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard dynamics functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry import register_dynamics


@jax.jit
def standard_dynamics_function(
    tau: Float[Array, "batch_size n_cells n_genes"],
    u0: Float[Array, "batch_size n_cells n_genes"],
    s0: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[Float[Array, "batch_size n_cells n_genes"], Float[Array, "batch_size n_cells n_genes"]]:
    """
    Standard RNA velocity dynamics function.
    
    This function implements the standard RNA velocity model:
    
    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s
    
    Args:
        tau: Time parameter
        u0: Initial unspliced RNA
        s0: Initial spliced RNA
        params: Dictionary of parameters (alpha, beta, gamma)
        
    Returns:
        Tuple of (unspliced, spliced) RNA counts
    """
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    
    # Compute dynamics
    ut = u0 * jnp.exp(-beta * tau) + (alpha / beta) * (1 - jnp.exp(-beta * tau))
    st = s0 * jnp.exp(-gamma * tau) + (beta * u0 / (gamma - beta)) * \
         (jnp.exp(-beta * tau) - jnp.exp(-gamma * tau))
    
    return ut, st


@jax.jit
def nonlinear_dynamics_function(
    tau: Float[Array, "batch_size n_cells n_genes"],
    u0: Float[Array, "batch_size n_cells n_genes"],
    s0: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[Float[Array, "batch_size n_cells n_genes"], Float[Array, "batch_size n_cells n_genes"]]:
    """
    Nonlinear RNA velocity dynamics function with saturation.
    
    This function implements a nonlinear RNA velocity model with saturation:
    
    du/dt = alpha * (1 / (1 + scaling * u)) - beta * u
    ds/dt = beta * u - gamma * s
    
    Args:
        tau: Time parameter
        u0: Initial unspliced RNA
        s0: Initial spliced RNA
        params: Dictionary of parameters (alpha, beta, gamma, scaling)
        
    Returns:
        Tuple of (unspliced, spliced) RNA counts
    """
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    scaling = params.get("scaling", jnp.ones_like(alpha) * 0.1)
    
    # For nonlinear dynamics, we use a simple Euler integration
    # This is a simplified approximation for testing purposes
    dt = 0.01
    steps = jnp.ceil(tau / dt).astype(jnp.int32)
    
    # Initialize state
    u = u0
    s = s0
    
    # Integrate
    for _ in range(100):  # Fixed number of steps for simplicity
        # Compute derivatives
        du_dt = alpha * (1.0 / (1.0 + scaling * u)) - beta * u
        ds_dt = beta * u - gamma * s
        
        # Update state
        u = u + du_dt * dt
        s = s + ds_dt * dt
    
    return u, s


def register_standard_dynamics():
    """Register standard dynamics functions."""
    register_dynamics("standard", standard_dynamics_function)
    register_dynamics("nonlinear", nonlinear_dynamics_function)
