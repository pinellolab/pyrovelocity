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
    # Get dimensions
    num_cells, num_genes = u_obs.shape
    
    # Create default log library sizes if not provided
    if u_log_library is None:
        u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    if s_log_library is None:
        s_log_library = jnp.log(jnp.sum(s_obs, axis=1))
    
    # Sample model parameters
    with numpyro.plate("gene", num_genes):
        # Sample RNA velocity parameters from prior
        if include_prior:
            # Use the prior function to sample parameters
            key = numpyro.prng_key()
            params = prior_fn(key, num_genes)
            
            # Register parameters with the model
            alpha = numpyro.sample("alpha", dist.Delta(params["alpha"]))
            beta = numpyro.sample("beta", dist.Delta(params["beta"]))
            gamma = numpyro.sample("gamma", dist.Delta(params["gamma"]))
        else:
            # Sample parameters directly
            alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
            beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
            gamma = numpyro.sample("gamma", dist.LogNormal(0.0, 1.0))
    
    # Sample latent time for each cell
    if latent_time:
        with numpyro.plate("cell", num_cells):
            tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))
    else:
        # Use fixed time points if latent_time is False
        tau = jnp.linspace(0.0, 1.0, num_cells)
        numpyro.deterministic("tau", tau)
    
    # Compute RNA dynamics
    params = {"alpha": alpha, "beta": beta, "gamma": gamma}
    
    # Initial conditions (steady state)
    u0 = alpha / beta
    s0 = alpha / gamma
    
    # Reshape parameters for broadcasting
    # tau has shape (num_cells,), parameters have shape (num_genes,)
    # We need to reshape to allow proper broadcasting in the dynamics function
    tau_expanded = tau[:, jnp.newaxis]  # Shape: (num_cells, 1)
    u0_expanded = u0[jnp.newaxis, :]    # Shape: (1, num_genes)
    s0_expanded = s0[jnp.newaxis, :]    # Shape: (1, num_genes)
    
    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha[jnp.newaxis, :],  # Shape: (1, num_genes)
        "beta": beta[jnp.newaxis, :],    # Shape: (1, num_genes)
        "gamma": gamma[jnp.newaxis, :]   # Shape: (1, num_genes)
    }
    
    # Add scaling parameter if it exists in the parameters
    if "scaling" in params:
        expanded_params["scaling"] = params["scaling"][jnp.newaxis, :]
    
    # Apply dynamics model to get expected counts
    u_expected, s_expected = dynamics_fn(tau_expanded, u0_expanded, s0_expanded, expanded_params)
    
    # Register expected counts with the model
    numpyro.deterministic("u_expected", u_expected)
    numpyro.deterministic("s_expected", s_expected)
    
    # Create scaling parameters for likelihood
    scaling_params = {
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
    }
    
    # Get likelihood distributions
    u_dist, s_dist = likelihood_fn(u_expected, s_expected, scaling_params)
    
    # Observe data
    with numpyro.plate("cell_gene", num_cells * num_genes, dim=-1):
        # Reshape observations to match plate dimension
        u_obs_flat = u_obs.reshape(-1)
        s_obs_flat = s_obs.reshape(-1)
        
        # Reshape expected counts to match plate dimension
        u_expected_flat = u_expected.reshape(-1)
        s_expected_flat = s_expected.reshape(-1)
        
        # Create likelihood distributions for flattened data
        u_dist_flat, s_dist_flat = likelihood_fn(u_expected_flat, s_expected_flat, scaling_params)
        
        # Observe data
        numpyro.sample("u_obs", u_dist_flat, obs=u_obs_flat)
        numpyro.sample("s_obs", s_dist_flat, obs=s_obs_flat)
    
    # Return model outputs
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "tau": tau,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }

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
    
    # Create a properly typed wrapper function for the prior
    def typed_prior_fn(key: jnp.ndarray, num_genes: int) -> Dict[str, Float[Array, "gene"]]:
        """Wrapper for sample_prior_parameters with proper type annotations.
        
        This ensures that a valid key is always passed to sample_prior_parameters.
        If key is None, it creates a new deterministic key.
        
        Args:
            key: JAX random key (can be None)
            num_genes: Number of genes
            
        Returns:
            Dictionary of parameter samples
        """
        # Handle the case where key is None
        if key is None:
            # Create a deterministic key for reproducibility
            key = jax.random.PRNGKey(0)
            
        # Now we can safely call sample_prior_parameters with a valid key
        return sample_prior_parameters(key, num_genes, config.prior)
    
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
            prior_fn=typed_prior_fn,  # Use the properly typed wrapper function
            latent_time=config.latent_time,
            include_prior=config.include_prior,
        )
    
    return model_fn