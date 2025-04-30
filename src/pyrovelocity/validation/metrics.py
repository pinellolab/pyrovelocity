"""
PyroVelocity validation metrics.

This module provides metrics for measuring the similarity between different
implementations of PyroVelocity (legacy, modular, and JAX).

The metrics include:
- Parameter comparison metrics (MSE, correlation, KL divergence, Wasserstein distance)
- Velocity comparison metrics (MSE, correlation, cosine similarity, magnitude similarity)
- Uncertainty comparison metrics (MSE, correlation, distribution similarity)
- Performance comparison metrics (training time ratio, inference time ratio, memory usage ratio)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype import beartype
from scipy import stats

# Basic metrics

@beartype
def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the mean squared error between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Mean squared error
    """
    return float(np.mean((x - y) ** 2))


@beartype
def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the correlation between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Correlation coefficient
    """
    return float(np.corrcoef(x.flatten(), y.flatten())[0, 1])


@beartype
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Cosine similarity
    """
    return float(np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x) * np.linalg.norm(y)))


@beartype
def kl_divergence(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the KL divergence between two arrays.

    Args:
        x: First array (probability distribution)
        y: Second array (probability distribution)
        epsilon: Small constant to avoid division by zero

    Returns:
        KL divergence
    """
    # Ensure arrays are probability distributions
    x = x / np.sum(x)
    y = y / np.sum(y)

    # Add small constant to avoid division by zero
    x = x + epsilon
    y = y + epsilon

    # Renormalize
    x = x / np.sum(x)
    y = y / np.sum(y)

    # Calculate KL divergence
    return float(np.sum(x * np.log(x / y)))


@beartype
def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Wasserstein distance between two arrays.

    Args:
        x: First array
        y: Second array

    Returns:
        Wasserstein distance
    """
    return float(stats.wasserstein_distance(x.flatten(), y.flatten()))


# Parameter comparison metrics

@beartype
def compute_parameter_metrics(
    params1: Dict[str, np.ndarray],
    params2: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for comparing model parameters.

    Args:
        params1: First set of parameters
        params2: Second set of parameters

    Returns:
        Dictionary of parameter metrics
    """
    # Initialize metrics dictionary
    metrics = {}

    # Compute metrics for each parameter
    for param in params1.keys():
        if param in params2:
            # Convert parameters to numpy arrays
            p1 = np.array(params1[param])
            p2 = np.array(params2[param])

            # Compute metrics
            metrics[param] = {
                "mse": mean_squared_error(p1, p2),
                "correlation": correlation(p1, p2),
                "kl_divergence": kl_divergence(np.abs(p1), np.abs(p2)),
                "wasserstein_distance": wasserstein_distance(p1, p2),
            }

    return metrics


# Velocity comparison metrics

@beartype
def compute_velocity_metrics(
    velocity1: Union[np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor, jnp.ndarray]]],
    velocity2: Union[np.ndarray, Dict[str, Union[np.ndarray, torch.Tensor, jnp.ndarray]]]
) -> Dict[str, float]:
    """
    Compute metrics for comparing velocity estimates.

    Args:
        velocity1: First velocity estimate, either a numpy array or a dictionary with a 'velocity' key
        velocity2: Second velocity estimate, either a numpy array or a dictionary with a 'velocity' key

    Returns:
        Dictionary of velocity metrics
    """
    # Handle dictionary input
    if isinstance(velocity1, dict) and 'velocity' in velocity1:
        velocity1 = velocity1['velocity']
    if isinstance(velocity2, dict) and 'velocity' in velocity2:
        velocity2 = velocity2['velocity']

    # Convert velocities to numpy arrays
    if isinstance(velocity1, torch.Tensor):
        v1 = velocity1.detach().cpu().numpy()
    elif isinstance(velocity1, jnp.ndarray):
        v1 = np.array(velocity1)
    else:
        v1 = np.array(velocity1)

    if isinstance(velocity2, torch.Tensor):
        v2 = velocity2.detach().cpu().numpy()
    elif isinstance(velocity2, jnp.ndarray):
        v2 = np.array(velocity2)
    else:
        v2 = np.array(velocity2)

    # Compute metrics
    metrics = {
        "mse": mean_squared_error(v1, v2),
        "correlation": correlation(v1, v2),
        "cosine_similarity": cosine_similarity(v1, v2),
        "magnitude_similarity": 1.0 - np.abs(np.linalg.norm(v1) - np.linalg.norm(v2)) / (np.linalg.norm(v1) + np.linalg.norm(v2)),
    }

    return metrics


# Uncertainty comparison metrics

@beartype
def compute_uncertainty_metrics(
    uncertainty1: np.ndarray,
    uncertainty2: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for comparing uncertainty estimates.

    Args:
        uncertainty1: First uncertainty estimate
        uncertainty2: Second uncertainty estimate

    Returns:
        Dictionary of uncertainty metrics
    """
    # Convert uncertainties to numpy arrays
    u1 = np.array(uncertainty1)
    u2 = np.array(uncertainty2)

    # Compute metrics
    metrics = {
        "mse": mean_squared_error(u1, u2),
        "correlation": correlation(u1, u2),
        "distribution_similarity": 1.0 - wasserstein_distance(u1, u2) / (np.mean(u1) + np.mean(u2)),
    }

    return metrics


# Performance comparison metrics

@beartype
def compute_performance_metrics(
    performance1: Dict[str, float],
    performance2: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute metrics for comparing performance.

    Args:
        performance1: First performance metrics
        performance2: Second performance metrics

    Returns:
        Dictionary of performance comparison metrics
    """
    # Compute ratios
    metrics = {}

    # Training time ratio
    if "training_time" in performance1 and "training_time" in performance2:
        metrics["training_time_ratio"] = performance2["training_time"] / performance1["training_time"]

    # Inference time ratio
    if "inference_time" in performance1 and "inference_time" in performance2:
        metrics["inference_time_ratio"] = performance2["inference_time"] / performance1["inference_time"]

    # Memory usage ratio
    if "memory_usage" in performance1 and "memory_usage" in performance2:
        metrics["memory_usage_ratio"] = performance2["memory_usage"] / performance1["memory_usage"]

    return metrics
