"""
NumPyro model definition for RNA velocity.

This module contains the NumPyro model definition for RNA velocity, including:

- velocity_model: Main NumPyro model for RNA velocity
- create_model: Factory function for creating models with different components
"""

from typing import Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from beartype import beartype

from pyrovelocity.models.jax.core.dynamics import (
    standard_dynamics_model,
    nonlinear_dynamics_model,
    dynamics_ode_model,
)
from pyrovelocity.models.jax.core.priors import (
    lognormal_prior,
    informative_prior,
    sample_prior_parameters,
)
from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
    negative_binomial_likelihood,
    create_likelihood,
)
from pyrovelocity.models.jax.core.state import (
    ModelConfig,
)

@beartype
def velocity_model(
    u_obs: Float[Array, "cell gene"],
    s_obs: Float[Array, "cell gene"],
    u_log_library: Optional[Float[Array, "cell"]] = None,
    s_log_library: Optional[Float[Array, "cell"]] = None,
    dynamics_fn: Callable = standard_dynamics_model,
    likelihood_fn: Callable = poisson_likelihood,
    prior_fn: Callable = sample_prior_parameters,
    latent_time: bool = True,
    include_prior: bool = True,
) -> Dict[str, Float[Array, "..."]]:
    """Main NumPyro model for RNA velocity.
    
    Args:
        u_obs: Observed unspliced RNA counts
        s_obs: Observed spliced RNA counts
        u_log_library: Log library size for unspliced RNA
        s_log_library: Log library size for spliced RNA
        dynamics_fn: Dynamics function
        likelihood_fn: Likelihood function
        prior_fn: Prior function
        latent_time: Whether to use latent time
        include_prior: Whether to include prior in the model
        
    Returns:
        Dictionary of model outputs
    """
    # Placeholder for future implementation
    raise NotImplementedError("This function will be implemented in a future phase.")

@beartype
def create_model(
    config: ModelConfig,
) -> Callable:
    """Factory function for creating models with different components.
    
    Args:
        config: Model configuration
        
    Returns:
        Model function
    """
    # Select dynamics function
    if config.dynamics == "standard":
        dynamics_fn = standard_dynamics_model
    elif config.dynamics == "nonlinear":
        dynamics_fn = nonlinear_dynamics_model
    elif config.dynamics == "ode":
        dynamics_fn = dynamics_ode_model
    else:
        raise ValueError(f"Unknown dynamics type: {config.dynamics}")
    
    # Select likelihood function
    likelihood_fn = create_likelihood(config.likelihood)
    
    # Create model function
    def model_fn(
        u_obs: Float[Array, "cell gene"],
        s_obs: Float[Array, "cell gene"],
        u_log_library: Optional[Float[Array, "cell"]] = None,
        s_log_library: Optional[Float[Array, "cell"]] = None,
    ) -> Dict[str, Float[Array, "..."]]:
        return velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            dynamics_fn=dynamics_fn,
            likelihood_fn=likelihood_fn,
            prior_fn=lambda key, num_genes: sample_prior_parameters(
                key, num_genes, config.prior
            ),
            latent_time=config.latent_time,
            include_prior=config.include_prior,
        )
    
    return model_fn