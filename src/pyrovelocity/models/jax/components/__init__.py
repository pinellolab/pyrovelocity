"""
Standard components for PyroVelocity JAX/NumPyro implementation.

This module registers standard components for the JAX implementation of PyroVelocity.
"""

from pyrovelocity.models.jax.components.dynamics import register_standard_dynamics
from pyrovelocity.models.jax.components.priors import register_standard_priors
from pyrovelocity.models.jax.components.likelihoods import register_standard_likelihoods
from pyrovelocity.models.jax.components.observations import register_standard_observations
from pyrovelocity.models.jax.components.guides import register_standard_guides


def register_standard_components():
    """Register all standard components."""
    register_standard_dynamics()
    register_standard_priors()
    register_standard_likelihoods()
    register_standard_observations()
    register_standard_guides()


# Register standard components when the module is imported
register_standard_components()
