"""
Model comparison utilities for PyroVelocity JAX/NumPyro implementation.

This module provides utilities for comparing different models, including:

- compute_log_likelihood: Compute log likelihood for a model
- compute_waic: Compute WAIC (Widely Applicable Information Criterion)
- compute_loo: Compute LOO (Leave-One-Out cross-validation)
- compare_models: Compare multiple models using information criteria
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from numpyro.handlers import substitute, trace

from pyrovelocity.models.jax.core.state import InferenceState


@beartype
def compute_log_likelihood(
    model: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Any:
    """
    Compute log likelihood for a model.

    Args:
        model: NumPyro model function
        posterior_samples: Dictionary of posterior samples
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        num_samples: Number of samples to use
        key: JAX random key

    Returns:
        Array of log likelihood values with shape (num_samples,)
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Create a log likelihood computation function
    def log_likelihood_fn(params):
        # Create a model trace with the given parameters
        with substitute(data=params):
            model_trace = trace(model).get_trace(*args, **kwargs)

        # Extract all sample sites that have an observed value
        log_likelihood = 0.0
        for site_name, site in model_trace.items():
            if site["type"] == "sample" and site["is_observed"]:
                # Get the distribution and observed value
                dist = site["fn"]
                obs = site["value"]

                # Compute log probability
                log_prob = dist.log_prob(obs)

                # Sum over all dimensions except batch
                while len(log_prob.shape) > 0:
                    log_prob = jnp.sum(log_prob, axis=-1)

                # Add to total log likelihood
                log_likelihood = log_likelihood + log_prob

        return log_likelihood

    # Vectorize the log likelihood function
    vmap_log_likelihood = jax.vmap(log_likelihood_fn)

    # Prepare parameters for each sample
    param_samples = {}
    for param_name, param_values in posterior_samples.items():
        # Take only the first num_samples samples if we have more
        if param_values.shape[0] > num_samples:
            param_samples[param_name] = param_values[:num_samples]
        else:
            param_samples[param_name] = param_values

    # Compute log likelihood for each sample
    log_likelihoods = vmap_log_likelihood(param_samples)

    return log_likelihoods


@beartype
def compute_waic(
    log_likelihoods: jnp.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute WAIC (Widely Applicable Information Criterion).

    Args:
        log_likelihoods: Array of log likelihood values with shape (num_samples,)

    Returns:
        Tuple of (waic, waic_se, p_waic) where:
        - waic: WAIC value
        - waic_se: Standard error of WAIC
        - p_waic: Effective number of parameters
    """
    # Compute log pointwise predictive density
    lppd = jnp.log(jnp.mean(jnp.exp(log_likelihoods), axis=0))

    # Compute p_waic (effective number of parameters)
    p_waic = jnp.var(log_likelihoods, axis=0)

    # Compute WAIC
    waic = -2 * (lppd - p_waic)

    # Compute standard error of WAIC
    waic_se = jnp.sqrt(jnp.var(waic) * log_likelihoods.shape[0])

    # Return WAIC, standard error, and effective number of parameters
    return float(waic), float(waic_se), float(p_waic)


@beartype
def compute_loo(
    log_likelihoods: jnp.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute LOO (Leave-One-Out cross-validation).

    Args:
        log_likelihoods: Array of log likelihood values with shape (num_samples,)

    Returns:
        Tuple of (loo, loo_se, p_loo) where:
        - loo: LOO value
        - loo_se: Standard error of LOO
        - p_loo: Effective number of parameters
    """
    # Compute Pareto smoothed importance sampling (PSIS)
    # This is a simplified version that doesn't do the full PSIS
    # For a more accurate implementation, use arviz.loo

    # Compute log weights
    log_weights = log_likelihoods - jnp.max(log_likelihoods, axis=0)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights, axis=0)

    # Compute effective sample size (not used in this simplified implementation)
    # ess = 1.0 / jnp.sum(weights**2, axis=0)

    # Compute LOO
    loo = -2 * jnp.sum(jnp.log(jnp.mean(jnp.exp(log_likelihoods), axis=0)))

    # Compute p_loo (effective number of parameters)
    lppd = jnp.sum(jnp.log(jnp.mean(jnp.exp(log_likelihoods), axis=0)))
    p_loo = lppd - jnp.sum(jnp.mean(log_likelihoods, axis=0))

    # Compute standard error of LOO
    loo_se = jnp.sqrt(
        log_likelihoods.shape[0]
        * jnp.var(-2 * jnp.log(jnp.mean(jnp.exp(log_likelihoods), axis=0)))
    )

    # Return LOO, standard error, and effective number of parameters
    return float(loo), float(loo_se), float(p_loo)


def compare_models(
    models: Dict[str, Tuple[Callable, InferenceState]],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models using information criteria.

    Args:
        models: Dictionary mapping model names to tuples of (model_fn, inference_state)
        args: Positional arguments for the models
        kwargs: Keyword arguments for the models
        num_samples: Number of samples to use
        key: JAX random key

    Returns:
        Dictionary mapping model names to dictionaries of comparison metrics
    """
    # Generate random key if not provided
    rng_key = jax.random.PRNGKey(0) if key is None else key

    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Initialize results dictionary
    results = {}

    # Compute metrics for each model
    for model_name, (model_fn, inference_state) in models.items():
        # Split key for this model
        rng_key, subkey = jax.random.split(rng_key)

        # Get posterior samples
        posterior_samples = inference_state.posterior_samples

        # Compute log likelihood
        log_likelihoods = compute_log_likelihood(
            model=model_fn,
            posterior_samples=posterior_samples,
            args=args,
            kwargs=kwargs,
            num_samples=num_samples,
            key=subkey,
        )

        # Compute WAIC
        waic, waic_se, p_waic = compute_waic(log_likelihoods)

        # Compute LOO
        loo, loo_se, p_loo = compute_loo(log_likelihoods)

        # Store results
        results[model_name] = {
            "log_likelihood": float(jnp.mean(log_likelihoods)),
            "waic": waic,
            "waic_se": waic_se,
            "p_waic": p_waic,
            "loo": loo,
            "loo_se": loo_se,
            "p_loo": p_loo,
        }

    # Compute model weights
    model_names = list(results.keys())
    waics = jnp.array([results[name]["waic"] for name in model_names])
    min_waic = jnp.min(waics)
    delta_waics = waics - min_waic
    weights = jnp.exp(-0.5 * delta_waics)
    weights = weights / jnp.sum(weights)

    # Add weights to results
    for i, name in enumerate(model_names):
        results[name]["weight"] = float(weights[i])

    return results
