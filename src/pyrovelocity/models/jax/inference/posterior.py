"""
Posterior analysis utilities for PyroVelocity JAX/NumPyro implementation.

This module contains posterior analysis utilities, including:

- sample_posterior: Sample from the posterior
- posterior_predictive: Generate posterior predictive samples
- compute_velocity: Compute RNA velocity from posterior samples
- compute_uncertainty: Compute uncertainty in RNA velocity
- analyze_posterior: Analyze posterior samples from either SVI or MCMC
- create_inference_data: Create ArviZ InferenceData object from posterior samples
- format_anndata_output: Format results into AnnData object
"""

from typing import Dict, Tuple, Optional, Any, List, Union, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import arviz as az
import anndata
from jaxtyping import Array, Float, PyTree
from beartype import beartype

from pyrovelocity.models.jax.core.state import InferenceState
from pyrovelocity.models.jax.core.dynamics import standard_dynamics_model

@beartype
def sample_posterior(
    inference_state: InferenceState,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Sample from the posterior.
    
    Args:
        inference_state: Inference state
        num_samples: Number of samples
        key: JAX random key
        
    Returns:
        Dictionary of posterior samples
    """
    # Get posterior samples from inference state
    posterior_samples = inference_state.posterior_samples
    
    # If we already have the requested number of samples, return them
    if all(v.shape[0] >= num_samples for v in posterior_samples.values()):
        # Truncate to the requested number of samples if needed
        return {k: v[:num_samples] for k, v in posterior_samples.items()}
    
    # If we need more samples, we need to resample
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Resample from the posterior
    resampled_samples = {}
    for param_name, param_samples in posterior_samples.items():
        # Get the number of existing samples
        n_existing = param_samples.shape[0]
        
        # Generate indices for resampling with replacement
        indices = jax.random.randint(
            key,
            shape=(num_samples,),
            minval=0,
            maxval=n_existing
        )
        
        # Resample
        resampled_samples[param_name] = param_samples[indices]
        
        # Get a new key for the next parameter
        key, _ = jax.random.split(key)
    
    return resampled_samples

@beartype
def posterior_predictive(
    model: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    args: Tuple,
    kwargs: Dict[str, Any],
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
    return_sites: Optional[List[str]] = None,
    parallel: bool = True,
) -> Dict[str, jnp.ndarray]:
    """Generate posterior predictive samples.
    
    Args:
        model: NumPyro model function
        posterior_samples: Dictionary of posterior samples
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        num_samples: Number of samples
        key: JAX random key
        return_sites: Names of sites to return
        parallel: Whether to run in parallel
        
    Returns:
        Dictionary of posterior predictive samples
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Subsample if we have more samples than requested
    if all(v.shape[0] > num_samples for v in posterior_samples.values()):
        subkey, key = jax.random.split(key)
        idx = jax.random.choice(subkey, posterior_samples[list(posterior_samples.keys())[0]].shape[0],
                               shape=(num_samples,), replace=False)
        posterior_samples = {k: v[idx] for k, v in posterior_samples.items()}
    
    # Create predictive object
    predictive = numpyro.infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=return_sites,
        parallel=parallel,
    )
    
    # Generate posterior predictive samples
    return predictive(key, *args, **kwargs)

@beartype
def compute_velocity(
    posterior_samples: Dict[str, jnp.ndarray],
    dynamics_fn: Callable = standard_dynamics_model,
) -> Dict[str, jnp.ndarray]:
    """Compute RNA velocity from posterior samples.
    
    Args:
        posterior_samples: Dictionary of posterior samples
        dynamics_fn: Dynamics function
        
    Returns:
        Dictionary of velocity results
    """
    # Extract parameters
    alpha = posterior_samples["alpha"]  # Shape: (num_samples, num_genes)
    beta = posterior_samples["beta"]    # Shape: (num_samples, num_genes)
    gamma = posterior_samples["gamma"]  # Shape: (num_samples, num_genes)
    tau = posterior_samples["tau"]      # Shape: (num_samples, num_cells)
    
    # Get dimensions
    num_samples, num_genes = alpha.shape
    _, num_cells = tau.shape
    
    # Reshape parameters for broadcasting
    alpha_expanded = alpha[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    beta_expanded = beta[:, :, jnp.newaxis]    # Shape: (num_samples, num_genes, 1)
    gamma_expanded = gamma[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    tau_expanded = tau[:, jnp.newaxis, :]      # Shape: (num_samples, 1, num_cells)
    
    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha_expanded,
        "beta": beta_expanded,
        "gamma": gamma_expanded
    }
    
    # Add scaling parameter if it exists in the parameters
    if "scaling" in posterior_samples:
        scaling = posterior_samples["scaling"]
        expanded_params["scaling"] = scaling[:, :, jnp.newaxis]
    
    # Initial conditions (steady state)
    u0 = alpha / beta
    s0 = alpha / gamma
    
    # Reshape for broadcasting
    u0_expanded = u0[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    s0_expanded = s0[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    
    # Apply dynamics model to get expected counts
    u_expected, s_expected = dynamics_fn(tau_expanded, u0_expanded, s0_expanded, expanded_params)
    
    # Compute velocity as time derivative of spliced counts
    # For standard dynamics: ds/dt = beta*u - gamma*s
    velocity = beta_expanded * u_expected - gamma_expanded * s_expected
    
    # Compute acceleration as second time derivative
    # For standard dynamics: d²s/dt² = beta*du/dt - gamma*ds/dt
    # du/dt = alpha - beta*u
    du_dt = alpha_expanded - beta_expanded * u_expected
    acceleration = beta_expanded * du_dt - gamma_expanded * velocity
    
    # Return results
    return {
        "u_expected": u_expected,
        "s_expected": s_expected,
        "velocity": velocity,
        "acceleration": acceleration,
    }

@beartype
def compute_uncertainty(
    velocity_samples: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Compute uncertainty in RNA velocity.
    
    Args:
        velocity_samples: Dictionary of velocity samples
        
    Returns:
        Dictionary of uncertainty measures
    """
    # Initialize uncertainty measures
    uncertainty = {}
    
    # Compute mean and standard deviation for each quantity
    for key, samples in velocity_samples.items():
        # Compute mean across samples (first dimension)
        mean = jnp.mean(samples, axis=0)
        uncertainty[f"{key}_mean"] = mean
        
        # Compute standard deviation across samples
        std = jnp.std(samples, axis=0)
        uncertainty[f"{key}_std"] = std
        
        # Compute coefficient of variation (CV = std/mean)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        cv = std / (jnp.abs(mean) + epsilon)
        uncertainty[f"{key}_cv"] = cv
        
        # Compute quantiles
        q10 = jnp.quantile(samples, 0.1, axis=0)
        q90 = jnp.quantile(samples, 0.9, axis=0)
        uncertainty[f"{key}_q10"] = q10
        uncertainty[f"{key}_q90"] = q90
        
        # Compute credible interval width
        ci_width = q90 - q10
        uncertainty[f"{key}_ci_width"] = ci_width
        
        # Compute normalized credible interval width
        norm_ci_width = ci_width / (jnp.abs(mean) + epsilon)
        uncertainty[f"{key}_norm_ci_width"] = norm_ci_width
    
    # For velocity specifically, compute additional measures
    if "velocity" in velocity_samples:
        velocity = velocity_samples["velocity"]
        
        # Compute probability of positive velocity
        prob_positive = jnp.mean(velocity > 0, axis=0)
        uncertainty["velocity_prob_positive"] = prob_positive
        
        # Compute velocity confidence (1 - 2*|0.5 - prob_positive|)
        # This is 1 when prob_positive is 0 or 1 (certain) and 0 when prob_positive is 0.5 (uncertain)
        velocity_confidence = 1 - 2 * jnp.abs(0.5 - prob_positive)
        uncertainty["velocity_confidence"] = velocity_confidence
    
    return uncertainty

@beartype
def analyze_posterior(
    inference_state: InferenceState,
    model: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    dynamics_fn: Callable = standard_dynamics_model,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze posterior samples from either SVI or MCMC.
    
    Args:
        inference_state: Inference state
        model: NumPyro model function
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        dynamics_fn: Dynamics function
        num_samples: Number of samples
        key: JAX random key
        
    Returns:
        Dictionary of analysis results
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Split key for different operations
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Sample from the posterior
    posterior_samples = sample_posterior(
        inference_state=inference_state,
        num_samples=num_samples,
        key=subkey1,
    )
    
    # Generate posterior predictive samples
    posterior_predictive_samples = posterior_predictive(
        model=model,
        posterior_samples=posterior_samples,
        args=args,
        kwargs=kwargs,
        num_samples=num_samples,
        key=subkey2,
    )
    
    # Compute velocity
    velocity_samples = compute_velocity(
        posterior_samples=posterior_samples,
        dynamics_fn=dynamics_fn,
    )
    
    # Compute uncertainty
    uncertainty = compute_uncertainty(
        velocity_samples=velocity_samples,
    )
    
    # Create inference data for ArviZ
    inference_data = create_inference_data(
        posterior_samples=posterior_samples,
        posterior_predictive_samples=posterior_predictive_samples,
        observed_data={
            "u_obs": kwargs.get("u_obs", args[0]),
            "s_obs": kwargs.get("s_obs", args[1]),
        },
    )
    
    # Combine all results
    results = {
        "posterior_samples": posterior_samples,
        "posterior_predictive": posterior_predictive_samples,
        "velocity": velocity_samples,
        "uncertainty": uncertainty,
        "inference_data": inference_data,
    }
    
    # Add diagnostics if available
    if inference_state.diagnostics is not None:
        results["diagnostics"] = inference_state.diagnostics
    
    return results

@beartype
def create_inference_data(
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive_samples: Optional[Dict[str, jnp.ndarray]] = None,
    observed_data: Optional[Dict[str, jnp.ndarray]] = None,
) -> az.InferenceData:
    """Create ArviZ InferenceData object from posterior samples.
    
    Args:
        posterior_samples: Dictionary of posterior samples
        posterior_predictive_samples: Optional dictionary of posterior predictive samples
        observed_data: Optional dictionary of observed data
        
    Returns:
        ArviZ InferenceData object
    """
    # Convert JAX arrays to NumPy arrays
    posterior_dict = {k: jnp.array(v) for k, v in posterior_samples.items()}
    
    # Create dictionary for InferenceData
    inference_dict = {}
    
    # Add posterior samples
    inference_dict["posterior"] = posterior_dict
    
    # Add posterior predictive samples if provided
    if posterior_predictive_samples is not None:
        pp_dict = {k: jnp.array(v) for k, v in posterior_predictive_samples.items()}
        inference_dict["posterior_predictive"] = pp_dict
    
    # Add observed data if provided
    if observed_data is not None:
        obs_dict = {k: jnp.array(v) for k, v in observed_data.items()}
        inference_dict["observed_data"] = obs_dict
    
    # Create InferenceData object
    return az.from_dict(**inference_dict)

@beartype
def format_anndata_output(
    adata: anndata.AnnData,
    results: Dict[str, Any],
    model_name: str = "velocity_model",
) -> anndata.AnnData:
    """Format results into AnnData object compatible with src/pyrovelocity/plots.
    
    Args:
        adata: AnnData object
        results: Dictionary of results
        model_name: Name of the model
        
    Returns:
        Updated AnnData object
    """
    # Create a copy of the AnnData object to avoid modifying the original
    adata_copy = adata.copy()
    
    # Extract uncertainty results
    uncertainty = results.get("uncertainty", {})
    
    # Add mean velocity to cell-specific annotations
    if "velocity_mean" in uncertainty:
        adata_copy.layers[f"{model_name}_velocity"] = jnp.array(uncertainty["velocity_mean"]).T
    
    # Add velocity confidence to cell-specific annotations
    if "velocity_confidence" in uncertainty:
        adata_copy.obs[f"{model_name}_velocity_confidence"] = jnp.array(uncertainty["velocity_confidence"])
    
    # Add velocity probability to cell-specific annotations
    if "velocity_prob_positive" in uncertainty:
        adata_copy.obs[f"{model_name}_velocity_probability"] = jnp.array(uncertainty["velocity_prob_positive"])
    
    # Add expected unspliced and spliced counts to layers
    if "u_expected_mean" in uncertainty and "s_expected_mean" in uncertainty:
        adata_copy.layers[f"{model_name}_u_expected"] = jnp.array(uncertainty["u_expected_mean"]).T
        adata_copy.layers[f"{model_name}_s_expected"] = jnp.array(uncertainty["s_expected_mean"]).T
    
    # Add uncertainty measures to var annotations
    for key in uncertainty:
        if key.endswith("_cv") or key.endswith("_norm_ci_width"):
            # These are gene-specific measures
            adata_copy.var[f"{model_name}_{key}"] = jnp.array(uncertainty[key])
    
    # Add model parameters to var annotations
    posterior_samples = results.get("posterior_samples", {})
    for param_name in ["alpha", "beta", "gamma"]:
        if param_name in posterior_samples:
            # Compute mean across samples
            param_mean = jnp.mean(posterior_samples[param_name], axis=0)
            adata_copy.var[f"{model_name}_{param_name}"] = jnp.array(param_mean)
    
    # Add latent time to obs annotations
    if "tau" in posterior_samples:
        tau_mean = jnp.mean(posterior_samples["tau"], axis=0)
        adata_copy.obs[f"{model_name}_latent_time"] = jnp.array(tau_mean)
    
    # Store the model name in uns
    if "uns" not in adata_copy or adata_copy.uns is None:
        adata_copy.uns = {}
    
    if "velocity_models" not in adata_copy.uns:
        adata_copy.uns["velocity_models"] = []
    
    if model_name not in adata_copy.uns["velocity_models"]:
        adata_copy.uns["velocity_models"].append(model_name)
    
    # Store additional model information
    adata_copy.uns[f"{model_name}_model_type"] = "jax_numpyro"
    
    return adata_copy