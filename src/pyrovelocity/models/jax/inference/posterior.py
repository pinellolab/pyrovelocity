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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, PyTree

from pyrovelocity.models.jax.core.dynamics import standard_dynamics_model
from pyrovelocity.models.jax.core.state import InferenceState
from pyrovelocity.models.jax.factory.config import ModelConfig
from pyrovelocity.models.jax.factory.factory import create_model


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
            key, shape=(num_samples,), minval=0, maxval=n_existing
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
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
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
        idx = jax.random.choice(
            subkey,
            posterior_samples[list(posterior_samples.keys())[0]].shape[0],
            shape=(num_samples,),
            replace=False,
        )
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
    if kwargs is None:
        kwargs = {}
    samples = predictive(key, *args, **kwargs)

    # If the samples don't include the original parameters, add them
    for param_name, param_samples in posterior_samples.items():
        if param_name not in samples:
            # Take only the first num_samples samples if we have more
            if param_samples.shape[0] > num_samples:
                samples[param_name] = param_samples[:num_samples]
            else:
                samples[param_name] = param_samples

    return samples


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
    alpha = posterior_samples[
        "alpha"
    ]  # Shape: (num_samples, num_genes) or (num_samples,)
    beta = posterior_samples[
        "beta"
    ]  # Shape: (num_samples, num_genes) or (num_samples,)
    gamma = posterior_samples[
        "gamma"
    ]  # Shape: (num_samples, num_genes) or (num_samples,)
    tau = posterior_samples[
        "tau"
    ]  # Shape: (num_samples, num_cells) or (num_samples,)

    # Get dimensions
    num_samples = alpha.shape[0]

    # Handle both 1D and 2D parameter cases
    if len(alpha.shape) == 1:
        # Single gene case - reshape to 2D
        alpha = alpha.reshape(-1, 1)  # Shape: (num_samples, 1)
        beta = beta.reshape(-1, 1)  # Shape: (num_samples, 1)
        gamma = gamma.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_genes = 1
    else:
        # Multiple genes case
        num_genes = alpha.shape[1]

    # Handle both 1D and 2D tau cases
    if len(tau.shape) == 1:
        # Single cell case - reshape to 2D
        tau = tau.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_cells = 1
    else:
        # Multiple cells case
        num_cells = tau.shape[1]

    # Reshape parameters for broadcasting
    alpha_expanded = alpha[
        :, :, jnp.newaxis
    ]  # Shape: (num_samples, num_genes, 1)
    beta_expanded = beta[
        :, :, jnp.newaxis
    ]  # Shape: (num_samples, num_genes, 1)
    gamma_expanded = gamma[
        :, :, jnp.newaxis
    ]  # Shape: (num_samples, num_genes, 1)
    tau_expanded = tau[:, jnp.newaxis, :]  # Shape: (num_samples, 1, num_cells)

    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha_expanded,
        "beta": beta_expanded,
        "gamma": gamma_expanded,
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
    dynamics_result = dynamics_fn(
        tau_expanded, u0_expanded, s0_expanded, expanded_params
    )

    # Handle both tuple and single return value cases
    if isinstance(dynamics_result, tuple) and len(dynamics_result) == 2:
        u_expected, s_expected = dynamics_result
    else:
        # If only one value is returned, assume it's the spliced counts
        # and compute unspliced counts from the parameters
        s_expected = dynamics_result
        # For standard dynamics, u = (ds/dt + gamma*s) / beta
        u_expected = (
            beta_expanded * u0_expanded * jnp.exp(-beta_expanded * tau_expanded)
        )

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
    model: Union[Callable, Dict[str, Any], ModelConfig],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    dynamics_fn: Callable = standard_dynamics_model,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze posterior samples from either SVI or MCMC.

    This function analyzes posterior samples from an inference state, computes velocity,
    uncertainty, and generates posterior predictive samples. It supports both direct model
    functions and model configurations through the factory system.

    Args:
        inference_state: Inference state containing posterior samples
        model: Either a NumPyro model function or a model configuration dictionary
        args: Positional arguments for the model (optional)
        kwargs: Keyword arguments for the model (optional)
        dynamics_fn: Dynamics function to use for velocity computation
        num_samples: Number of posterior samples to use
        key: JAX random key (optional)

    Returns:
        Dictionary of analysis results containing:
        - posterior_samples: Samples from the posterior distribution
        - posterior_predictive: Posterior predictive samples
        - velocity: Computed velocity samples
        - uncertainty: Uncertainty measures for velocity
        - inference_data: ArviZ InferenceData object
        - diagnostics: Inference diagnostics (if available)
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Split key for different operations
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Sample from the posterior
    posterior_samples = sample_posterior(
        inference_state=inference_state,
        num_samples=num_samples,
        key=subkey1,
    )

    # Handle model configuration by creating a model function
    model_fn = model
    if isinstance(model, (dict, ModelConfig)):
        model_fn = create_model(model)

    # Generate posterior predictive samples
    posterior_predictive_samples = posterior_predictive(
        model=model_fn,
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
    observed_data = {}
    if args and len(args) > 0:
        observed_data["u_obs"] = args[0]
    if args and len(args) > 1:
        observed_data["s_obs"] = args[1]

    # Override with kwargs if provided
    if "u_obs" in kwargs:
        observed_data["u_obs"] = kwargs["u_obs"]
    if "s_obs" in kwargs:
        observed_data["s_obs"] = kwargs["s_obs"]

    inference_data = create_inference_data(
        posterior_samples=posterior_samples,
        posterior_predictive_samples=posterior_predictive_samples,
        observed_data=observed_data,
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
) -> Any:  # Using Any instead of az.InferenceData to avoid beartype issues
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
        pp_dict = {
            k: jnp.array(v) for k, v in posterior_predictive_samples.items()
        }
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

    This function takes the results from analyze_posterior and formats them into an
    AnnData object for visualization and further analysis. It handles different shapes
    and formats of the input data, ensuring compatibility with the plotting functions.

    Args:
        adata: AnnData object with cells as rows and genes as columns
        results: Dictionary of results from analyze_posterior
        model_name: Name of the model to use as a prefix for annotations

    Returns:
        Updated AnnData object with velocity information
    """
    # Create a copy of the AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    # Validate AnnData structure
    if adata_copy.n_obs == 0 or adata_copy.n_vars == 0:
        raise ValueError("AnnData object is empty")

    # Extract uncertainty results
    uncertainty = results.get("uncertainty", {})
    if not uncertainty:
        raise ValueError("Results dictionary does not contain uncertainty measures")

    # Extract posterior samples
    posterior_samples = results.get("posterior_samples", {})
    if not posterior_samples:
        raise ValueError("Results dictionary does not contain posterior samples")

    # Extract velocity samples
    velocity_samples = results.get("velocity", {})
    if not velocity_samples:
        # Try to compute velocity if not provided
        try:
            from pyrovelocity.models.jax.core.dynamics import (
                standard_dynamics_model,
            )
            velocity_samples = compute_velocity(
                posterior_samples=posterior_samples,
                dynamics_fn=standard_dynamics_model,
            )
        except Exception as e:
            raise ValueError(f"Could not compute velocity: {e}")

    # Get dimensions
    n_cells = adata_copy.n_obs
    n_genes = adata_copy.n_vars

    # Add mean velocity to cell-specific annotations
    if "velocity_mean" in uncertainty:
        # Handle both 1D and 2D cases
        velocity = jnp.array(uncertainty["velocity_mean"])

        # Check shape and reshape if necessary
        if len(velocity.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            velocity = velocity.reshape(1, -1)
        elif velocity.shape[0] == n_genes and velocity.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            velocity = velocity.T
        elif velocity.shape[0] == 1 and velocity.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            velocity = jnp.broadcast_to(velocity, (n_cells, n_genes))

        # Ensure the shape matches AnnData dimensions
        if velocity.shape[1] != n_genes:
            # If the number of genes doesn't match, try to reshape
            if velocity.size == n_genes:
                velocity = velocity.reshape(1, n_genes)
                velocity = jnp.broadcast_to(velocity, (n_cells, n_genes))
            else:
                raise ValueError(
                    f"Velocity shape {velocity.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in layers
        adata_copy.layers[f"{model_name}_velocity"] = jnp.array(velocity)

    # Add velocity confidence to cell-specific annotations
    if "velocity_confidence" in uncertainty:
        # Handle both 1D and 2D cases
        confidence = jnp.array(uncertainty["velocity_confidence"])

        # Check shape and reshape if necessary
        if len(confidence.shape) > 1:
            if confidence.shape[0] == 1:
                # If shape is (1, n_cells), reshape to (n_cells,)
                confidence = confidence.reshape(-1)
            elif confidence.shape[1] == 1:
                # If shape is (n_cells, 1), reshape to (n_cells,)
                confidence = confidence.reshape(-1)
            elif confidence.shape[0] == n_genes and confidence.shape[1] == n_cells:
                # If shape is (n_genes, n_cells), take mean across genes
                confidence = jnp.mean(confidence, axis=0)
            elif confidence.shape[0] == n_cells and confidence.shape[1] == n_genes:
                # If shape is (n_cells, n_genes), take mean across genes
                confidence = jnp.mean(confidence, axis=1)

        # Ensure the shape matches AnnData dimensions
        if len(confidence.shape) == 1 and confidence.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if confidence.size == 1:
                # If scalar, broadcast to all cells
                confidence = jnp.broadcast_to(confidence, (n_cells,))
            elif confidence.size == n_genes:
                # If gene-wise, take mean
                confidence = jnp.mean(confidence) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Confidence shape {confidence.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in obs
        adata_copy.obs[f"{model_name}_velocity_confidence"] = jnp.array(confidence)

    # Add velocity probability to cell-specific annotations
    if "velocity_prob_positive" in uncertainty:
        # Handle both 1D and 2D cases
        probability = jnp.array(uncertainty["velocity_prob_positive"])

        # Check shape and reshape if necessary
        if len(probability.shape) > 1:
            if probability.shape[0] == 1:
                # If shape is (1, n_cells), reshape to (n_cells,)
                probability = probability.reshape(-1)
            elif probability.shape[1] == 1:
                # If shape is (n_cells, 1), reshape to (n_cells,)
                probability = probability.reshape(-1)
            elif probability.shape[0] == n_genes and probability.shape[1] == n_cells:
                # If shape is (n_genes, n_cells), take mean across genes
                probability = jnp.mean(probability, axis=0)
            elif probability.shape[0] == n_cells and probability.shape[1] == n_genes:
                # If shape is (n_cells, n_genes), take mean across genes
                probability = jnp.mean(probability, axis=1)

        # Ensure the shape matches AnnData dimensions
        if len(probability.shape) == 1 and probability.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if probability.size == 1:
                # If scalar, broadcast to all cells
                probability = jnp.broadcast_to(probability, (n_cells,))
            elif probability.size == n_genes:
                # If gene-wise, take mean
                probability = jnp.mean(probability) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Probability shape {probability.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in obs
        adata_copy.obs[f"{model_name}_velocity_probability"] = jnp.array(probability)

    # Add expected unspliced and spliced counts to layers
    if "u_expected_mean" in uncertainty and "s_expected_mean" in uncertainty:
        # Handle both 1D and 2D cases for u_expected
        u_expected = jnp.array(uncertainty["u_expected_mean"])
        s_expected = jnp.array(uncertainty["s_expected_mean"])

        # Check shape and reshape if necessary for u_expected
        if len(u_expected.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            u_expected = u_expected.reshape(1, -1)
        elif u_expected.shape[0] == n_genes and u_expected.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            u_expected = u_expected.T
        elif u_expected.shape[0] == 1 and u_expected.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            u_expected = jnp.broadcast_to(u_expected, (n_cells, n_genes))

        # Check shape and reshape if necessary for s_expected
        if len(s_expected.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            s_expected = s_expected.reshape(1, -1)
        elif s_expected.shape[0] == n_genes and s_expected.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            s_expected = s_expected.T
        elif s_expected.shape[0] == 1 and s_expected.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            s_expected = jnp.broadcast_to(s_expected, (n_cells, n_genes))

        # Ensure the shapes match AnnData dimensions
        if u_expected.shape[1] != n_genes or s_expected.shape[1] != n_genes:
            # If the number of genes doesn't match, try to reshape
            if u_expected.size == n_genes and s_expected.size == n_genes:
                u_expected = u_expected.reshape(1, n_genes)
                s_expected = s_expected.reshape(1, n_genes)
                u_expected = jnp.broadcast_to(u_expected, (n_cells, n_genes))
                s_expected = jnp.broadcast_to(s_expected, (n_cells, n_genes))
            else:
                raise ValueError(
                    f"Expected counts shapes {u_expected.shape}, {s_expected.shape} do not match "
                    f"AnnData dimensions ({n_cells}, {n_genes})"
                )

        # Store in layers
        adata_copy.layers[f"{model_name}_u_expected"] = jnp.array(u_expected)
        adata_copy.layers[f"{model_name}_s_expected"] = jnp.array(s_expected)

    # Add uncertainty measures to var annotations
    for key in uncertainty:
        if key.endswith("_cv") or key.endswith("_norm_ci_width"):
            # These are gene-specific measures
            value = jnp.array(uncertainty[key])

            # Handle both 1D and 2D cases
            if len(value.shape) > 1:
                if value.shape[0] == 1:
                    # If shape is (1, n_genes), reshape to (n_genes,)
                    value = value[0]
                elif value.shape[1] == 1:
                    # If shape is (n_genes, 1), reshape to (n_genes,)
                    value = value.reshape(-1)
                elif value.shape[0] == n_cells and value.shape[1] == n_genes:
                    # If shape is (n_cells, n_genes), take mean across cells
                    value = jnp.mean(value, axis=0)
                elif value.shape[0] == n_genes and value.shape[1] == n_cells:
                    # If shape is (n_genes, n_cells), take mean across cells
                    value = jnp.mean(value, axis=1)

            # Ensure the shape matches AnnData dimensions
            if len(value.shape) == 1 and value.shape[0] != n_genes:
                # If the number of genes doesn't match, try to reshape or broadcast
                if value.size == 1:
                    # If scalar, broadcast to all genes
                    value = jnp.broadcast_to(value, (n_genes,))
                elif value.size == n_cells:
                    # If cell-wise, take mean
                    value = jnp.mean(value) * jnp.ones(n_genes)
                else:
                    raise ValueError(
                        f"Uncertainty measure {key} shape {value.shape} does not match "
                        f"AnnData dimensions ({n_cells}, {n_genes})"
                    )

            # Store in var
            adata_copy.var[f"{model_name}_{key}"] = jnp.array(value)

    # Add model parameters to var annotations
    for param_name in ["alpha", "beta", "gamma"]:
        if param_name in posterior_samples:
            # Compute mean across samples
            param_samples = jnp.array(posterior_samples[param_name])
            param_mean = jnp.mean(param_samples, axis=0)

            # Handle both scalar and vector cases
            if not isinstance(param_mean, jnp.ndarray) or param_mean.ndim == 0:
                # If scalar, broadcast to all genes
                param_mean = jnp.array([param_mean])
                param_mean = jnp.broadcast_to(param_mean, (n_genes,))
            elif param_mean.shape[0] != n_genes:
                # If the number of genes doesn't match, try to reshape or broadcast
                if param_mean.size == 1:
                    # If scalar, broadcast to all genes
                    param_mean = jnp.broadcast_to(param_mean, (n_genes,))
                elif param_mean.size == n_cells:
                    # If cell-wise, take mean
                    param_mean = jnp.mean(param_mean) * jnp.ones(n_genes)
                else:
                    raise ValueError(
                        f"Parameter {param_name} shape {param_mean.shape} does not match "
                        f"AnnData dimensions ({n_cells}, {n_genes})"
                    )

            # Store in var
            adata_copy.var[f"{model_name}_{param_name}"] = jnp.array(param_mean)

    # Add latent time to obs annotations
    if "tau" in posterior_samples:
        # Compute mean across samples
        tau_samples = jnp.array(posterior_samples["tau"])
        tau_mean = jnp.mean(tau_samples, axis=0)

        # Handle both scalar and vector cases
        if not isinstance(tau_mean, jnp.ndarray) or tau_mean.ndim == 0:
            # If scalar, broadcast to all cells
            tau_mean = jnp.array([tau_mean])
            tau_mean = jnp.broadcast_to(tau_mean, (n_cells,))
        elif tau_mean.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if tau_mean.size == 1:
                # If scalar, broadcast to all cells
                tau_mean = jnp.broadcast_to(tau_mean, (n_cells,))
            elif tau_mean.size == n_genes:
                # If gene-wise, take mean
                tau_mean = jnp.mean(tau_mean) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Latent time shape {tau_mean.shape} does not match "
                    f"AnnData dimensions ({n_cells}, {n_genes})"
                )

        # Store in obs
        adata_copy.obs[f"{model_name}_latent_time"] = jnp.array(tau_mean)

    # Store the model name in uns
    # Initialize uns if it doesn't exist or is None
    if not hasattr(adata_copy, "uns") or adata_copy.uns is None:
        adata_copy.uns = {}

    # Initialize velocity_models if it doesn't exist
    if "velocity_models" not in adata_copy.uns:
        adata_copy.uns["velocity_models"] = []

    # Add model_name to velocity_models if it's not already there
    if model_name not in adata_copy.uns["velocity_models"]:
        adata_copy.uns["velocity_models"].append(model_name)

    # Store additional model information
    adata_copy.uns[f"{model_name}_model_type"] = "jax_numpyro"

    # Store model parameters in uns
    adata_copy.uns[f"{model_name}_params"] = {
        param: jnp.array(posterior_samples[param]).tolist()
        for param in ["alpha", "beta", "gamma"]
        if param in posterior_samples
    }

    return adata_copy
