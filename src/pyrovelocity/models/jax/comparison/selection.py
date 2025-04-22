"""
Model selection utilities for PyroVelocity JAX/NumPyro implementation.

This module provides utilities for selecting models, including:

- select_best_model: Select the best model based on information criteria
- cross_validate: Perform cross-validation for model selection
- compute_predictive_performance: Compute predictive performance metrics
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype

from pyrovelocity.models.jax.comparison.comparison import compare_models
from pyrovelocity.models.jax.core.state import InferenceState


@beartype
def select_best_model(
    models: Dict[str, Tuple[Callable, InferenceState]],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    criterion: str = "waic",
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Select the best model based on information criteria.

    Args:
        models: Dictionary mapping model names to tuples of (model_fn, inference_state)
        args: Positional arguments for the models
        kwargs: Keyword arguments for the models
        criterion: Criterion to use for model selection ('waic' or 'loo')
        num_samples: Number of samples to use
        key: JAX random key

    Returns:
        Tuple of (best_model_name, comparison_results)
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Compare models
    comparison_results = compare_models(
        models=models,
        args=args,
        kwargs=kwargs,
        num_samples=num_samples,
        key=key,
    )

    # Select best model based on criterion
    if criterion == "waic":
        # Lower WAIC is better
        best_model_name = min(
            comparison_results.keys(),
            key=lambda name: comparison_results[name]["waic"],
        )
    elif criterion == "loo":
        # Lower LOO is better
        best_model_name = min(
            comparison_results.keys(),
            key=lambda name: comparison_results[name]["loo"],
        )
    elif criterion == "weight":
        # Higher weight is better
        best_model_name = max(
            comparison_results.keys(),
            key=lambda name: comparison_results[name]["weight"],
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return best_model_name, comparison_results


@beartype
def compute_predictive_performance(
    model: Callable,
    inference_state: InferenceState,
    test_args: Tuple = (),
    test_kwargs: Optional[Dict[str, Any]] = None,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute predictive performance metrics for a model on test data.

    Args:
        model: NumPyro model function
        inference_state: Inference state containing posterior samples
        test_args: Positional arguments for the model with test data
        test_kwargs: Keyword arguments for the model with test data
        num_samples: Number of samples to use
        key: JAX random key

    Returns:
        Dictionary of performance metrics
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Create a dictionary with a single model
    models = {"model": (model, inference_state)}

    # Compare the model against itself on test data
    comparison_results = compare_models(
        models=models,
        args=test_args,
        kwargs=test_kwargs,
        num_samples=num_samples,
        key=key,
    )

    # Return the performance metrics for the model
    return comparison_results["model"]


@beartype
def cross_validate(
    model_fn: Callable,
    data_splits: List[Tuple[Tuple, Dict[str, Any], Tuple, Dict[str, Any]]],
    inference_fn: Callable,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation for model selection.

    Args:
        model_fn: NumPyro model function
        data_splits: List of (train_args, train_kwargs, test_args, test_kwargs) tuples
        inference_fn: Function to perform inference, should return InferenceState
        num_samples: Number of samples to use
        key: JAX random key

    Returns:
        Dictionary of cross-validation metrics
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize results
    results = {
        "log_likelihood": [],
        "waic": [],
        "loo": [],
        "p_waic": [],
        "p_loo": [],
    }

    # Perform cross-validation
    for i, (train_args, train_kwargs, test_args, test_kwargs) in enumerate(data_splits):
        # Split key for this fold
        key, subkey = jax.random.split(key)

        # Perform inference on training data
        inference_state = inference_fn(
            model_fn, train_args, train_kwargs, key=subkey
        )

        # Compute performance on test data
        performance = compute_predictive_performance(
            model=model_fn,
            inference_state=inference_state,
            test_args=test_args,
            test_kwargs=test_kwargs,
            num_samples=num_samples,
            key=subkey,
        )

        # Store results
        for metric, value in performance.items():
            if metric in results:
                results[metric].append(value)

    # Compute mean and standard deviation for each metric
    summary = {}
    for metric, values in results.items():
        summary[f"{metric}_mean"] = float(jnp.mean(jnp.array(values)))
        summary[f"{metric}_std"] = float(jnp.std(jnp.array(values)))

    # Combine results and summary
    return {**results, **summary}
