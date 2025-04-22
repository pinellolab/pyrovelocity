"""
Standard guide factory functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard guide factory functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional

import numpyro
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from pyrovelocity.models.jax.registry import register_guide


def auto_normal_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Auto normal guide factory function.

    This function creates an AutoNormal guide for the given model.

    Args:
        model: Model function
        guide_params: Dictionary of guide parameters

    Returns:
        Guide function
    """
    if guide_params is None:
        guide_params = {}

    # Get guide parameters
    init_loc_fn = guide_params.get("init_loc_fn", None)
    init_scale = guide_params.get("init_scale", 0.1)

    # Create guide
    return AutoNormal(
        model,
        init_loc_fn=init_loc_fn,
        init_scale=init_scale,
    )


def auto_delta_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Auto delta guide factory function.

    This function creates an AutoDelta guide for the given model.

    Args:
        model: Model function
        guide_params: Dictionary of guide parameters

    Returns:
        Guide function
    """
    if guide_params is None:
        guide_params = {}

    # Get guide parameters
    init_loc_fn = guide_params.get("init_loc_fn", None)

    # Create guide
    return AutoDelta(
        model,
        init_loc_fn=init_loc_fn,
    )


def custom_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Custom guide factory function.

    This function creates a custom guide for the given model.

    Args:
        model: Model function
        guide_params: Dictionary of guide parameters

    Returns:
        Guide function
    """
    if guide_params is None:
        guide_params = {}

    # Define custom guide
    def guide(*args, **kwargs):
        # Sample parameters
        with numpyro.plate("gene", guide_params.get("num_genes", 1)):
            alpha_loc = numpyro.param(
                "alpha_loc", guide_params.get("alpha_loc", 0.0)
            )
            alpha_scale = numpyro.param(
                "alpha_scale",
                guide_params.get("alpha_scale", 1.0),
                constraint=numpyro.distributions.constraints.positive,
            )
            numpyro.sample(
                "alpha", numpyro.distributions.Normal(alpha_loc, alpha_scale)
            )

            beta_loc = numpyro.param(
                "beta_loc", guide_params.get("beta_loc", 0.0)
            )
            beta_scale = numpyro.param(
                "beta_scale",
                guide_params.get("beta_scale", 1.0),
                constraint=numpyro.distributions.constraints.positive,
            )
            numpyro.sample(
                "beta", numpyro.distributions.Normal(beta_loc, beta_scale)
            )

            gamma_loc = numpyro.param(
                "gamma_loc", guide_params.get("gamma_loc", 0.0)
            )
            gamma_scale = numpyro.param(
                "gamma_scale",
                guide_params.get("gamma_scale", 1.0),
                constraint=numpyro.distributions.constraints.positive,
            )
            numpyro.sample(
                "gamma", numpyro.distributions.Normal(gamma_loc, gamma_scale)
            )

        # Sample latent time
        with numpyro.plate("cell", guide_params.get("num_cells", 1)):
            tau_loc = numpyro.param("tau_loc", guide_params.get("tau_loc", 0.0))
            tau_scale = numpyro.param(
                "tau_scale",
                guide_params.get("tau_scale", 1.0),
                constraint=numpyro.distributions.constraints.positive,
            )
            numpyro.sample(
                "tau", numpyro.distributions.Normal(tau_loc, tau_scale)
            )

    return guide


def register_standard_guides():
    """Register standard guide factory functions."""
    register_guide("auto", auto_normal_guide_factory)
    register_guide("auto_normal", auto_normal_guide_factory)
    register_guide("auto_delta", auto_delta_guide_factory)
    register_guide("custom", custom_guide_factory)
